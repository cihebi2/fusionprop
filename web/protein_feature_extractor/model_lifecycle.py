import time
import threading
import torch
import logging

# <<< Use root logger for very basic initial logging, in case named logger isn't captured
root_logger_mm = logging.getLogger() 

logger = logging.getLogger(__name__)

class ManagedModel:
    """
    Manages the lifecycle of a model, including automatic loading,
    tracking activity, and releasing after a period of inactivity.
    """
    def __init__(self, load_function, model_name="UnnamedModel", release_timeout_seconds=600):
        """
        Args:
            load_function (callable): A function that takes no arguments and returns the loaded model.
            model_name (str): Name of the model for logging.
            release_timeout_seconds (int): Seconds of inactivity after which to release the model.
        """
        # <<< ADDED HIGH-VISIBILITY LOGGING
        root_logger_mm.warning(f' MANAGEDMODEL_INIT_CALLED: Name: {model_name}, Timeout: {release_timeout_seconds} ')
        self.load_function = load_function
        self.model_name = model_name
        self.release_timeout_seconds = release_timeout_seconds
        
        self._model_instance = None
        self._last_accessed_time = 0
        self._lock = threading.Lock()
        self._checker_thread = None
        self._stop_event = threading.Event()

        self.get() # Initial load and start checker
        logger.info(f"ManagedModel for {self.model_name} initialized.")

    def get(self):
        """Returns the model instance, loading it if necessary or if timed out."""
        with self._lock:
            self._last_accessed_time = time.time()
            if self._model_instance is None:
                logger.info(f"Model {self.model_name} is None or timed out, reloading.")
                self._load()
            
            if self._checker_thread is None or not self._checker_thread.is_alive():
                self._start_checker_thread()
            return self._model_instance

    def _load(self):
        """Loads the model using the provided load_function."""
        # Assumes _lock is already acquired
        logger.info(f"Loading model: {self.model_name}...")
        try:
            self._model_instance = self.load_function()
            self._last_accessed_time = time.time()
            logger.info(f"Model {self.model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}", exc_info=True)
            self._model_instance = None # Ensure it's None on failure
            # Optionally, re-raise or handle more gracefully

    def _release(self):
        """Releases the model and cleans up resources."""
        with self._lock:
            if self._model_instance is not None:
                logger.info(f"Releasing model: {self.model_name} due to inactivity.")
                # Let Python's garbage collector handle the model object itself
                logger.info(f"[_release] - Deleting model instance reference for {self.model_name}.")
                model_instance_for_del = self._model_instance # Hold ref temporarily if needed for specific cleanup
                del self._model_instance 
                self._model_instance = None
                logger.info(f"[_release] - Model instance reference for {self.model_name} deleted.")
                # Optionally, perform explicit cleanup on model_instance_for_del if needed before it's GC'd
                # del model_instance_for_del 
                try:
                    if torch.cuda.is_available():
                        logger.info(f"[_release] - Calling torch.cuda.empty_cache() for {self.model_name}.")
                        torch.cuda.empty_cache()
                        logger.info(f"CUDA cache emptied for {self.model_name}.")
                except Exception as e:
                    logger.error(f"Error emptying CUDA cache for {self.model_name}: {e}", exc_info=True)
                logger.info(f"Model {self.model_name} released.")
            # Stop the checker thread if the model is released
            # It will be restarted on next `get()` call
            if self._checker_thread and self._checker_thread.is_alive():
                 # No explicit stop needed for daemon thread that checks a simple condition
                 # but if it were more complex, we might signal it.
                 pass


    def _checker_loop(self):
        # <<< ADDED HIGH-VISIBILITY LOGGING
        root_logger_mm.warning(f' MANAGEDMODEL_CHECKER_LOOP_STARTED: Name: {self.model_name} ')
        logger.info(f"Checker thread started for {self.model_name}.")
        while not self._stop_event.is_set():
            try:
                # <<< ATTEMPT FINAL LOGGING ISOLATION >>>
                if self._model_instance is not None: 
                    logger.info(f'[MOD_LIFECYCLE_LOOP] Iteration for {self.model_name}: Model IS LOADED.') # 使用模块 logger.info
                else:
                    # 当模型未加载时，使用模块 logger.debug
                    logger.debug(f'[MOD_LIFECYCLE_LOOP] Iteration for {self.model_name}: Model is NOT loaded.')

                if self._model_instance is not None: # Only check for inactivity if model is loaded
                    with self._lock: # Ensure last_accessed_time is read consistently
                        inactive_time = time.time() - self._last_accessed_time
                    
                    if inactive_time > self.release_timeout_seconds:
                        logger.info(f"{self.model_name} inactive for {inactive_time:.2f}s. Threshold is {self.release_timeout_seconds}s.")
                        logger.info(f"[Checker Loop] - Inactivity detected for {self.model_name}. Attempting release.")
                        self._release()
                        # Once released, the loop can break as get() will restart it if needed.
                        # However, to keep the thread running for future reloads, we don't break.
                        # Or, we can stop it and let get() restart it.
                        # For now, let it keep running but check if model is None.
                
                # Check more frequently if model is loaded, less if not
                check_interval = 30 if self._model_instance is not None else self.release_timeout_seconds / 2
                
                # Wait for the check_interval or until stop_event is set
                stopped = self._stop_event.wait(timeout=max(10, check_interval)) # min 10s check
                if stopped:
                    break 
            except Exception as e:
                logger.error(f"Error in checker thread for {self.model_name}: {e}", exc_info=True)
                # Avoid busy-looping on continuous errors
                time.sleep(60)
        logger.info(f"Checker thread stopped for {self.model_name}.")

    def _start_checker_thread(self):
        # Assumes _lock is already acquired
        if self._checker_thread is None or not self._checker_thread.is_alive():
            self._stop_event.clear()
            self._checker_thread = threading.Thread(target=self._checker_loop, daemon=True)
            self._checker_thread.start()
            logger.info(f"Checker thread initiated for {self.model_name}.")
            
    def __del__(self):
        logger.info(f"ManagedModel {self.model_name} is being deleted. Stopping checker thread.")
        self._stop_event.set()
        if self._checker_thread and self._checker_thread.is_alive():
            self._checker_thread.join(timeout=5) # Wait for thread to stop
        self._release() # Ensure model is released on deletion

# Example Usage (Illustrative - will be integrated into FeatureManager)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    def my_dummy_model_loader():
        print("LOADER: Loading dummy model...")
        time.sleep(2) # Simulate load time
        model = "THIS IS A DUMMY MODEL"
        print("LOADER: Dummy model loaded.")
        return model

    # Release after 5 seconds for testing
    managed_dummy_model = ManagedModel(load_function=my_dummy_model_loader, model_name="DummyTestModel", release_timeout_seconds=5)
    
    print("Getting model first time...")
    model = managed_dummy_model.get()
    print(f"Got model: {model}\n")
    
    print("Waiting for 3 seconds...")
    time.sleep(3)
    model = managed_dummy_model.get() # Access within timeout
    print(f"Got model again: {model}\n")

    print("Waiting for 7 seconds (should trigger release and reload)...")
    time.sleep(7)
    
    print("Getting model after long wait...")
    model = managed_dummy_model.get() # Access after timeout
    print(f"Got model after timeout: {model}\n")

    print("Waiting for 7 seconds again to ensure it releases again...")
    time.sleep(7)
    print("Test finished. Model should be released by checker thread if not accessed.")
    
    # Cleanly stop the manager if it were a long-lived app component
    # In this example, __del__ will handle it upon script exit.
    # del managed_dummy_model
    # time.sleep(2) # Give time for thread to stop
    print("Script ending.") 