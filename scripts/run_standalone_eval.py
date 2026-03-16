"""Standalone evaluation: run task evaluators on a predictions.json file."""
import argparse
import json
import logging

import torchmetrics

from olmo.util import prepare_cli_environment, log_metrics_to_console
from olmo.data.get_dataset import get_dataset_by_name
from olmo.eval.eval_utils import get_evaluator
from olmo.eval.evaluators import SavePredictions

log = logging.getLogger(__name__)


def run_eval(predictions_path, task, split="test"):
    """Run evaluation on a predictions file. Returns resolved metrics dict."""

    # 1. Load predictions
    with open(predictions_path) as f:
        predictions_json = json.load(f)
    log.info(f"Loaded {len(predictions_json)} predictions from {predictions_path}")

    # 2. Get evaluator config and build evaluators
    evaluator_config = get_evaluator(task)
    inf_evaluator = evaluator_config.build(default_save_dir=None)

    # Filter out SavePredictions — we only want real evaluators
    evaluators = [m for m in inf_evaluator.metrics if not isinstance(m, SavePredictions)]
    if not evaluators:
        log.warning(f"No evaluators for task {task}, skipping")
        return {}

    # 3. Load dataset and build metadata lookup
    log.info(f"Loading dataset: {task} split={split}")
    dataset = get_dataset_by_name(task, split)
    log.info(f"Dataset size: {len(dataset)}")

    metadata_by_id = {}
    for i in range(len(dataset)):
        ex = dataset.get(i, None)
        ex_id = ex.get('metadata', {}).get('example_id', str(i))
        metadata_by_id[ex_id] = ex['metadata']

    # 4. Match predictions to metadata
    matched_metadatas = []
    matched_preds = []
    for pred_entry in predictions_json:
        eid = pred_entry['example_id']
        if eid in metadata_by_id:
            matched_metadatas.append(metadata_by_id[eid])
            matched_preds.append(pred_entry['prediction'])
        else:
            log.warning(f"No metadata for example_id={eid}, skipping")

    log.info(f"Matched {len(matched_preds)}/{len(predictions_json)} predictions to metadata")

    if not matched_preds:
        log.warning("No predictions matched metadata, skipping evaluation")
        return {}

    # 5. Build predictions dict that evaluators expect
    predictions = {
        "predictions": matched_preds,
        "predictions_text": matched_preds,
    }

    # 6. Run evaluators
    all_metrics = {}
    for metric in evaluators:
        results = metric(matched_metadatas, predictions, step=None, tokenizer=None)
        all_metrics.update(results)

    # 7. Resolve metrics (MeanMetric -> float)
    resolved_metrics = {}
    for k in sorted(all_metrics):
        v = all_metrics[k]
        if isinstance(v, float):
            resolved_metrics[k] = v
        elif isinstance(v, torchmetrics.Metric):
            resolved_metrics[k] = v.compute().item()
        else:
            # Skip non-numeric metrics (HtmlTable, List, etc.)
            log.info(f"Skipping non-numeric metric {k}: {type(v).__name__}")

    # 8. Log to console
    log_metrics_to_console(task, resolved_metrics)

    return resolved_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on predictions.json")
    parser.add_argument("--predictions", required=True, help="Path to predictions.json")
    parser.add_argument("--task", required=True, help="Task name (e.g. mevis_point_track_per_frame_fps_6_sample_fps_1)")
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    args = parser.parse_args()

    prepare_cli_environment()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_eval(args.predictions, args.task, args.split)
