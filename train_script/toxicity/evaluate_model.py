import os
import json
import argparse
import torch
import pandas as pd
import numpy as np
import fcntl # For file locking (optional but good practice)
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_curve, auc,
    precision_score, recall_score, f1_score, confusion_matrix,
    matthews_corrcoef as mcc_score
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datetime import datetime
import time # Needed for retry logic in _save_aggregate_results

# Assuming train_12_2.py is in the same directory or accessible via PYTHONPATH
# Make sure train_12_2.py exists and contains these imports
from train_12_2 import (
    ExperimentConfig, Logger, FeatureManager, ProteinFeatureDataset,
    collate_protein_features, SingleModelClassifier, FusionModelClassifier,
    WeightedFusionClassifier, ModelConfig, ESM2Config, ESMCConfig, SPLMConfig
)

def parse_eval_args():
    """Parses command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained protein language model.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model (.pt file).")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the experiment config.json file from the training run.")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="Path to the test CSV data file (combined, with 'sequence' and 'label' columns).")
    parser.add_argument("--sequence_column", type=str, default="sequence",
                        help="Name of the column containing protein sequences in the test CSV.")
    parser.add_argument("--target_column", type=str, default="label",
                        help="Name of the column containing true labels in the test CSV (optional, for metrics).")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results (predictions, metrics).")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for data loading.")
    parser.add_argument("--feature_cache_size", type=int, default=1000,
                        help="Feature cache size for the dataset.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for evaluation (e.g., 'cuda:0', 'cpu'). Autodetects if None.")
    parser.add_argument("--aggregate_results_file", type=str, default=None,
                        help="Path to a single JSON file to aggregate results from multiple runs.")
    parser.add_argument("--run_identifier", type=str, default=None,
                        help="Unique identifier for this run, used as a key in the aggregate results file.")
    # Removed --no_header_test as the new shell script will create a file with a proper header.
    return parser.parse_args()

def _calculate_metrics(true_labels, pred_labels, probabilities, logger):
    """Calculates evaluation metrics."""
    if true_labels is None or len(true_labels) == 0:
        logger.warning("No true labels provided or empty true labels. Skipping metrics calculation.")
        return {"predictions_only": True, "mcc": np.nan, "auc": np.nan} # Ensure keys exist for summary

    metrics = {}
    try:
        metrics["accuracy"] = accuracy_score(true_labels, pred_labels)
        metrics["precision"] = precision_score(true_labels, pred_labels, zero_division=0)
        metrics["recall"] = recall_score(true_labels, pred_labels, zero_division=0)
        metrics["f1"] = f1_score(true_labels, pred_labels, zero_division=0)
        metrics["mcc"] = mcc_score(true_labels, pred_labels)

        if len(np.unique(true_labels)) > 1 and probabilities is not None and len(probabilities) == len(true_labels):
            metrics["auc"] = roc_auc_score(true_labels, probabilities)
            precision_vals, recall_vals, _ = precision_recall_curve(true_labels, probabilities)
            metrics["pr_auc"] = auc(recall_vals, precision_vals)
        else:
            logger.warning("Not enough data for AUC/PR-AUC (e.g., single class, no probabilities, or mismatched lengths).")
            metrics["auc"] = np.nan
            metrics["pr_auc"] = np.nan

        # Ensure all labels [0,1] are present for confusion_matrix or handle appropriately
        # If only one class in true_labels, confusion_matrix might error or give unexpected result
        # For simplicity, we assume binary classification context where [0,1] are possible.
        cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
        if cm.size == 4: # Ensure it's a 2x2 matrix
            tn, fp, fn, tp = cm.ravel()
            metrics["tn"] = int(tn)
            metrics["fp"] = int(fp)
            metrics["fn"] = int(fn)
            metrics["tp"] = int(tp)
        else:
            logger.warning(f"Could not compute full confusion matrix. Shape: {cm.shape}")
            metrics["tn"] = metrics["fp"] = metrics["fn"] = metrics["tp"] = np.nan

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        metrics["error"] = str(e)
        # Ensure default numeric keys exist if error occurs mid-calculation
        for key in ["mcc", "auc", "accuracy", "precision", "recall", "f1", "pr_auc", "tn", "fp", "fn", "tp"]:
            if key not in metrics: metrics[key] = np.nan
    return metrics

def _save_aggregate_results(aggregate_file, identifier, metrics_dict, model_path, config_path, logger):
    """Loads, updates, and saves the aggregate results JSON file."""
    if not identifier:
        logger.warning("No run_identifier provided. Skipping aggregate results saving.")
        return

    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(aggregate_file), exist_ok=True)

            # Lock the file before reading/writing
            with open(aggregate_file, 'a+') as f: # Open in append+read mode, create if not exists
                fcntl.flock(f, fcntl.LOCK_EX) # Exclusive lock
                f.seek(0) # Go to the beginning to read
                try:
                    content = f.read()
                    if not content:
                        all_results = {}
                    else:
                        all_results = json.loads(content)
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode JSON from {aggregate_file}. Starting fresh.")
                    all_results = {}

                # Add model/config info to the metrics being saved
                metrics_to_save = metrics_dict.copy()
                metrics_to_save['model_path'] = model_path
                metrics_to_save['config_path'] = config_path

                # Update the dictionary
                all_results[identifier] = metrics_to_save

                # Go back to the beginning, truncate, and write updated data
                f.seek(0)
                f.truncate()
                json.dump(all_results, f, indent=4)

                fcntl.flock(f, fcntl.LOCK_UN) # Unlock
            logger.info(f"Successfully updated aggregate results file: {aggregate_file}")
            return # Success, exit the loop
        except (IOError, BlockingIOError, Exception) as e:
            logger.error(f"Attempt {attempt + 1}/{retry_attempts}: Error updating aggregate file {aggregate_file}: {e}")
            if attempt < retry_attempts - 1:
                time.sleep(1) # Wait before retrying
            else:
                logger.error(f"Failed to update aggregate results file after {retry_attempts} attempts.")
                # Optionally, try saving locally as a fallback?
                # fallback_path = os.path.join(os.path.dirname(model_path), f"{identifier.replace('/', '_')}_metrics_fallback.json")
                # try: ... save metrics_to_save to fallback_path ...
        finally:
            # Ensure unlock even if errors occur within the 'with' block, though 'with' should handle it.
            # This is belt-and-suspenders.
            try:
                if 'f' in locals() and not f.closed:
                    fcntl.flock(f, fcntl.LOCK_UN)
            except Exception: # Ignore errors during final unlock attempt
                pass

def main_evaluate():
    """Main function for model evaluation."""
    args = parse_eval_args()

    os.makedirs(args.output_dir, exist_ok=True)
    # Logger setup (can be done after output_dir is confirmed)
    log_file = os.path.join(args.output_dir, "evaluation.log")
    logger = Logger(log_file=log_file, console=True)
    logger.info(f"Starting evaluation for model: {args.model_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Arguments: {vars(args)}")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading experiment configuration from: {args.config_path}")
    try:
        exp_config = ExperimentConfig.load_config(args.config_path)
        exp_config.training_config.feature_extraction_device = device
        exp_config.training_config.training_device = device
        logger.info("Experiment configuration loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load experiment configuration: {e}")
        return

    logger.info("Initializing FeatureManager...")
    try:
        feature_manager = FeatureManager(exp_config, logger)
        feature_manager.preload_models()
        logger.info("FeatureManager initialized and models preloaded.")
    except Exception as e:
        logger.error(f"Failed to initialize FeatureManager: {e}")
        feature_manager = None # Ensure cleanup doesn't fail
        return

    logger.info(f"Loading test data from: {args.test_csv}")
    try:
        test_df = pd.read_csv(args.test_csv)
        if not {args.sequence_column, args.target_column}.issubset(test_df.columns):
            logger.error(f"Test CSV must contain '{args.sequence_column}' and '{args.target_column}' columns. Found: {test_df.columns.tolist()}")
            if feature_manager: feature_manager.cleanup()
            return
        test_df[args.target_column] = pd.to_numeric(test_df[args.target_column], errors='coerce')
        # Ensure the sequence column is treated as string
        test_df[args.sequence_column] = test_df[args.sequence_column].astype(str)
        has_labels = True # Assume labels are present as per current design
        logger.info(f"Test data loaded: {len(test_df)} records.")
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        if feature_manager: feature_manager.cleanup()
        return

    logger.info("Creating test dataset and dataloader...")
    try:
        test_dataset = ProteinFeatureDataset(
            df=test_df,
            feature_manager=feature_manager,
            config=exp_config,
            target_col=args.target_column,
            sequence_col=args.sequence_column,
            cache_size=args.feature_cache_size,
            logger=logger
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_protein_features,
            pin_memory=True if device.type == 'cuda' else False
        )
        logger.info("Test dataset and dataloader created.")
    except Exception as e:
        logger.error(f"Failed to create dataset/dataloader: {e}")
        if feature_manager: feature_manager.cleanup()
        return

    logger.info(f"Loading trained model from: {args.model_path}")
    try:
        train_mode = exp_config.training_config.train_mode
        model_to_load_class = None
        if train_mode == "fusion":
            fusion_type = getattr(exp_config.training_config, 'fusion_type', 'default')
            model_to_load_class = WeightedFusionClassifier if fusion_type == "weighted" else FusionModelClassifier
        elif train_mode == "single":
            model_to_load_class = SingleModelClassifier
        else:
            logger.error(f"Unsupported train_mode '{train_mode}' in config.")
            if feature_manager: feature_manager.cleanup()
            return

        model = model_to_load_class.load_model(args.model_path, device=device)
        model.eval()
        logger.info(f"Model loaded successfully ({model_to_load_class.__name__}) and set to evaluation mode.")
    except Exception as e:
        logger.error(f"Failed to load trained model: {e}")
        if feature_manager: feature_manager.cleanup()
        return

    logger.info("Starting evaluation on the test set...")
    all_probs_list, all_labels_list = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch_data)
            all_probs_list.extend(torch.sigmoid(outputs).cpu().numpy())
            if has_labels and "labels" in batch_data:
                all_labels_list.extend(batch_data["labels"].cpu().numpy())
    logger.info("Evaluation finished.")

    all_probs_np = np.array(all_probs_list)
    pred_labels_np = (all_probs_np >= 0.5).astype(int)
    results_df = test_df[[args.sequence_column, args.target_column]].copy()
    results_df['predicted_probability'] = all_probs_np
    results_df['predicted_label'] = pred_labels_np

    true_labels_for_metrics = None
    if has_labels and all_labels_list:
        true_labels_np = np.array(all_labels_list)
        valid_indices = ~np.isnan(true_labels_np)
        true_labels_for_metrics = true_labels_np[valid_indices]
        pred_labels_for_metrics = pred_labels_np[valid_indices]
        probs_for_metrics = all_probs_np[valid_indices]
        if len(true_labels_for_metrics) > 0:
            logger.info(f"Calculating metrics for {len(true_labels_for_metrics)} samples with non-NaN labels.")
            metrics = _calculate_metrics(true_labels_for_metrics, pred_labels_for_metrics, probs_for_metrics, logger)
        else:
            logger.warning("No valid labels found for metrics calculation after filtering NaNs.")
            metrics = {"status": "no_valid_labels", "mcc": np.nan, "auc": np.nan}
    else:
        logger.info("No labels for metrics calculation or labels list is empty.")
        metrics = {"status": "predictions_only", "mcc": np.nan, "auc": np.nan}

    predictions_path = os.path.join(args.output_dir, "predictions.csv")
    try:
        results_df.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to: {predictions_path}")
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    # Convert numpy types to native Python types for JSON serialization
    serializable_metrics = {k: (float(v) if isinstance(v, (np.float32, np.float64, np.int32, np.int64)) else v) for k, v in metrics.items()}

    if args.aggregate_results_file:
        _save_aggregate_results(
            aggregate_file=args.aggregate_results_file,
            identifier=args.run_identifier,
            metrics_dict=serializable_metrics,
            model_path=args.model_path,
            config_path=args.config_path,
            logger=logger
        )
    else:
        # Save metrics locally if not aggregating
        try:
            with open(metrics_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=4)
            logger.info(f"Metrics saved to: {metrics_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics locally: {e}")

    logger.info("Metrics Summary:")
    for k, v in metrics.items(): logger.info(f"  {k}: {v:.4f}" if isinstance(v, (float, np.floating)) and not np.isnan(v) else f"  {k}: {v}")

    if feature_manager: feature_manager.cleanup()
    logger.info(f"Evaluation for model {args.model_path} completed. Results in: {args.output_dir}")
    return metrics # Return metrics for potential aggregation

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # Already set or not supported
    main_evaluate() 