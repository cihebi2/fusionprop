from celery import shared_task, current_task
from .models import PredictionJob
try:
    from protein_predictor import ProteinPredictorManager
except ImportError:
    print("ERROR: protein_predictor library not found. Ensure it's in the Python path.")
    ProteinPredictorManager = None

import time
import json
import uuid
import threading # Import threading for locking
import gc # Import garbage collector
import torch # <<< ADDED IMPORT
from django.utils import timezone
import logging

logger = logging.getLogger(__name__) # Added logger instance

# <<< REVERTED IMPORTS to absolute from 'web.'
from protein_feature_extractor.feature_manager import FeatureManager
from protein_feature_extractor.config import get_default_feature_configs # Placeholder

# --- Configuration ---
# IDLE_TIMEOUT_SECONDS = 10 * 60  # 10 minutes - This is for the old ProteinPredictorManager idle check
# The FeatureManager's model_release_timeout will handle ESM2/ESMC specific timeout.

# --- Worker Process State (Not shared between different worker processes) ---
_predictor_manager_instance = None # Renamed to avoid confusion with the class name
_feature_manager_instance = None   # <<< ADDED: Global instance for FeatureManager
_last_completion_time = 0
_manager_lock = threading.Lock() # Lock to manage access to manager instances

# --- Helper Functions ---
def get_managers(): # Combined function to get both managers
    """Loads ProteinPredictorManager and FeatureManager if not already loaded."""
    global _predictor_manager_instance, _feature_manager_instance
    with _manager_lock:
        if _feature_manager_instance is None:
            print("FeatureManager not loaded. Initializing...")
            try:
                # TODO: Get timeout from Django settings or a more robust config system
                fm_timeout = 10 # Using your 10s test timeout
                fm_configs = get_default_feature_configs() # Get configs
                fm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                _feature_manager_instance = FeatureManager(
                    configs=fm_configs,
                    device=fm_device,
                    model_release_timeout=fm_timeout
                )
                print(f"FeatureManager initialized successfully with timeout {fm_timeout}s.")
            except Exception as e:
                print(f"Error initializing FeatureManager: {e}")
                _feature_manager_instance = None # Ensure it's None on error

        if _predictor_manager_instance is None:
            print("ProteinPredictorManager not loaded. Initializing...")
            if ProteinPredictorManager:
                try:
                    # <<< ADDED DIAGNOSTIC PRINTING
                    print(f"DIAGNOSTIC: ProteinPredictorManager module path: {ProteinPredictorManager.__module__} {getattr(ProteinPredictorManager, '__file__', 'N/A')}")
                    import inspect
                    print(f"DIAGNOSTIC: ProteinPredictorManager.__init__ signature: {inspect.signature(ProteinPredictorManager.__init__)}")
                    # <<< END DIAGNOSTIC PRINTING

                    # Pass the initialized FeatureManager instance and its timeout
                    _predictor_manager_instance = ProteinPredictorManager(
                        use_toxicity=True,
                        use_thermostability=True,
                        use_solubility=True,
                        feature_manager_instance=_feature_manager_instance, # Pass the instance
                        # ProteinPredictorManager will use _feature_manager_instance's timeout if it uses it for new FM
                        # Or, if PPM re-creates its own FM, we could pass fm_timeout here too.
                        # Based on current PPM changes, it uses the passed instance.
                    )
                    print("ProteinPredictorManager initialized successfully (using shared FeatureManager).")
                except Exception as e:
                    print(f"Error initializing ProteinPredictorManager: {e}")
                    _predictor_manager_instance = None # Ensure it remains None on error
            else:
                print("Skipping ProteinPredictorManager initialization due to import error.")
        
        return _predictor_manager_instance, _feature_manager_instance

def unload_managers(): # Modified to reflect new structure
    """Unloads the ProteinPredictorManager. FeatureManager unloads models via its own mechanism."""
    global _predictor_manager_instance, _last_completion_time
    # FeatureManager handles its own model unloading via ManagedModel.
    # We might not need to explicitly call shutdown on _feature_manager_instance here unless
    # we want to force-stop its checker threads during worker shutdown (which __del__ in ManagedModel tries to do).

    with _manager_lock:
        if _predictor_manager_instance is not None:
            print("Unloading ProteinPredictorManager...")
            # Specific cleanup for ProteinPredictorManager's *other* models (Tox, Thermo, Sol) if any.
            # ESM2/ESMC are handled by FeatureManager.
            try:
                # Example: if ProteinPredictorManager has a method to release its own non-ESM models
                # if hasattr(_predictor_manager_instance, 'release_non_esm_models'):
                #     _predictor_manager_instance.release_non_esm_models()

                del _predictor_manager_instance 
                _predictor_manager_instance = None
                # _last_completion_time = 0 # Reset timer for PPM specific models if needed
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        # Note: FeatureManager also calls this when its models are released.
                        # Calling it here again might be redundant but generally safe.
                        torch.cuda.empty_cache()
                        print("torch.cuda.empty_cache() called during ProteinPredictorManager unload.")
                except ImportError:
                    pass
                print("ProteinPredictorManager unloaded.")
            except Exception as e:
                print(f"Error during ProteinPredictorManager unload: {e}")
        
        # Do we need to do anything with _feature_manager_instance here?
        # Its models auto-release. If we want to stop its threads, we can:
        # if _feature_manager_instance is not None:
        #     _feature_manager_instance.shutdown() # Call the shutdown method we added
        #     _feature_manager_instance = None

def update_last_completion_time():
    """Updates the timestamp of the last completed task."""
    global _last_completion_time
    with _manager_lock:
        _last_completion_time = time.time()

# --- Celery Tasks ---

@shared_task(bind=True)
def run_prediction_task(self, job_id_str: str, sequences: list, names: list):
    """
    Celery task to run batch protein predictions. Loads manager on demand.
    """
    job_id = uuid.UUID(job_id_str)
    total_sequences = len(sequences)
    results_list = []

    # manager = None # Old way
    predictor_manager_to_use = None
    # feature_manager_to_use = None # Not directly used by task, but by predictor_manager

    try:
        job = PredictionJob.objects.get(job_id=job_id)
        job.status = 'PROCESSING'
        job.save()

        # Update task state for initial progress
        self.update_state(state='PROGRESS', meta={'current': 0, 'total': total_sequences, 'status': 'Initializing...'})

        # --- Load Managers On Demand ---
        # predictor_manager = get_predictor_manager() # Old way
        predictor_manager_to_use, _ = get_managers() # Get both, use predictor_manager_to_use

        if predictor_manager_to_use is None:
            raise RuntimeError("ProteinPredictorManager is not available or failed to initialize.")
        # --- End Load Managers ---

        print(f"Starting batch prediction for Job ID: {job_id} ({total_sequences} sequences)")
        start_time = time.time()

        # --- Batch Prediction Logic ---
        for i, (sequence, name) in enumerate(zip(sequences, names)):
            try:
                progress_status = f"Processing sequence {i+1}/{total_sequences} ('{name}')"
                self.update_state(state='PROGRESS', meta={'current': i + 1, 'total': total_sequences, 'status': progress_status})
                print(f"Job {job_id}: {progress_status}")

                # Use the loaded manager instance
                single_result = predictor_manager_to_use.predict_all(
                    sequence=sequence,
                    return_confidence=True,
                    cleanup_features=True
                )
                logger.info(f"Task: single_result for sequence '{name}': {single_result}")
                single_result['name'] = name
                single_result['sequence_preview'] = sequence[:10] + '...' if len(sequence) > 10 else sequence
                results_list.append(single_result)

            except Exception as e_single:
                print(f"Error processing sequence {name} for job {job_id}: {e_single}")
                results_list.append({
                    'name': name,
                    'sequence_preview': sequence[:10] + '...' if len(sequence) > 10 else sequence,
                    'error': str(e_single)
                })
        # --- End Batch Prediction ---

        end_time = time.time()
        print(f"Finished batch prediction for Job ID: {job_id}. Duration: {end_time - start_time:.2f} seconds.")

        logger.info(f"Task: final results_list before saving for job {job_id_str}: {results_list}")
        # Store results and mark as completed
        job.results = json.dumps(results_list)
        job.status = 'COMPLETED'
        job.completed_at = timezone.now()
        job.save()

        # --- Update activity time on success ---
        update_last_completion_time()
        # --- End Update ---

        return {'status': 'COMPLETED', 'job_id': job_id_str, 'count': len(results_list)}

    except PredictionJob.DoesNotExist:
        print(f"Error: PredictionJob with ID {job_id} not found.")
        # No job object to update, maybe log this?
        return {'status': 'FAILED', 'error': 'Job not found'}
    except Exception as e:
        print(f"Error running prediction task for job {job_id}: {e}")
        try:
            # Try to mark the job as FAILED in the database
            job = PredictionJob.objects.get(job_id=job_id)
            job.status = 'FAILED'
            job.error_message = str(e)
            job.completed_at = timezone.now() # Mark completion time even for failures
            job.save()
        except PredictionJob.DoesNotExist:
            pass # Job already gone or never existed properly
        except Exception as db_err:
            print(f"Additionally, failed to update job status in DB: {db_err}") # Log DB error

        # Update Celery task state to FAILED
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        # Optionally re-raise to have Celery record the traceback
        # raise e
        return {'status': 'FAILED', 'job_id': job_id_str, 'error': str(e)}
    # Note: 'finally' block is removed as update_last_completion_time is called on success path


@shared_task(name="predictor.tasks.check_idle_and_unload")
def check_idle_and_unload():
    """Periodic task to check worker idle time and unload non-ESM models if needed."""
    # This task now primarily concerns models managed directly by ProteinPredictorManager (Tox, Thermo, Sol)
    # as FeatureManager (handling ESM2/ESMC) has its own timeout mechanism.
    print("Running periodic check for idle worker (for ProteinPredictorManager non-ESM models)...")
    should_unload_ppm = False
    current_time = time.time()
    
    # Define IDLE_TIMEOUT_SECONDS for PPM specific models, if different from FeatureManager
    PPM_IDLE_TIMEOUT_SECONDS = 10 * 60 # Example: 10 minutes for Tox, Thermo, Sol models

    with _manager_lock:
        if _predictor_manager_instance is not None:
            if _last_completion_time > 0 and (current_time - _last_completion_time) > PPM_IDLE_TIMEOUT_SECONDS:
                print(f"ProteinPredictorManager idle time ({current_time - _last_completion_time:.0f}s) exceeds its threshold ({PPM_IDLE_TIMEOUT_SECONDS}s).")
                should_unload_ppm = True

    if should_unload_ppm:
        unload_managers() # This will unload PPM; FeatureManager handles its own.
