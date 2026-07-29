"""Standalone evaluation: run task evaluators on a predictions.json file."""
import argparse
import json
import logging
import os
from os.path import join, exists

import numpy as np
import torchmetrics

from olmo.util import prepare_cli_environment, log_metrics_to_console
from olmo.eval.eval_utils import get_evaluator
from olmo.eval.evaluators import SavePredictions
from olmo.eval.object_tracking_utils import points_from_masks

log = logging.getLogger(__name__)

MASK_SCALES = [("strict", 1.0)]#, ("lenient_2x", 2.0)]

CFC_VIDEO_FPS = 6


def build_metadata_from_masks_rle(masks_dir, video_fps=CFC_VIDEO_FPS, sampling_fps=None):
    """Build per-example_id metadata directly from precomputed MasksRLE files.

    Walks {masks_dir}/{example_id}/0.json (one qid per example). For each
    example, derives GT points by computing bbox centroids from each RLE mask.
    No COCO JSON or queries.json needed.

    Returns:
        metadata_by_id: dict keyed by example_id (basename of the dir).
    """
    from glob import glob as _glob

    metadata_by_id = {}
    n_skipped = 0
    example_dirs = sorted([d for d in _glob(join(masks_dir, "*")) if os.path.isdir(d)])

    for vdir in example_dirs:
        example_id = os.path.basename(vdir)
        for mask_file in sorted(_glob(join(vdir, "*.json"))):
            qid = os.path.splitext(os.path.basename(mask_file))[0]
            with open(mask_file) as f:
                masks = json.load(f)

            # Find a sample non-None RLE for h/w
            sample_rle = None
            for frame_list in masks.values():
                for rle in frame_list:
                    if rle is not None:
                        sample_rle = rle
                        break
                if sample_rle is not None:
                    break
            if sample_rle is None:
                n_skipped += 1
                continue

            height, width = sample_rle['size']
            # Derive GT points from the masks via the shared helper, so points and
            # masks share object-slot order (same rule as the eval dataset path).
            points = points_from_masks(masks, video_fps)

            entry = {
                'example_id': example_id,
                'w': width,
                'h': height,
                'video_fps': video_fps,
                'sampling_fps': sampling_fps,
                'video': example_id,
                'points': points,
                'initial_points': points,
                'masks': masks,
                'mask_id': [str(i) for i in range(len(masks))],
            }
            metadata_by_id[example_id] = entry

    log.info(f"Built metadata for {len(metadata_by_id)} entries from {masks_dir} "
             f"({n_skipped} mask files skipped — all-None)")
    return metadata_by_id


def build_metadata_from_dataset(task, split):
    """Build per-video metadata by loading the dataset class."""
    from olmo.data.get_dataset import get_dataset_by_name

    log.info(f"Loading dataset: {task} split={split}")
    dataset = get_dataset_by_name(task, split)
    if hasattr(dataset, 'is_eval'):
        dataset.is_eval = True
    log.info(f"Dataset size: {len(dataset)}")

    metadata_by_id = {}
    for i in range(len(dataset)):
        ex = dataset.get(i, None)
        ex_id = ex.get('metadata', {}).get('example_id', str(i))
        metadata_by_id[ex_id] = ex['metadata']

    return metadata_by_id


def _meta_width(meta):
    if 'w' in meta:
        return meta['w']
    if 'image_size' in meta:
        return meta['image_size'][0]
    return None

def classify_direction(track_points, width, threshold_frac=0.05):
    """Classify a track's net x-motion as 'left', 'right', or 'stationary'."""
    if len(track_points) < 2:
        return 'stationary'
    first_x = track_points[0][0]
    last_x = track_points[-1][0]
    dx = last_x - first_x
    threshold = width * threshold_frac
    if dx > threshold:
        return 'right'
    elif dx < -threshold:
        return 'left'
    return 'stationary'

def run_eval(predictions_path, task, split="test", overwrite=False, masks_dir=None, out_dir=None,
             sampling_fps=None):
    """Run evaluation on a predictions file. Returns and writes resolved metrics dict."""

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

    # 3. Build metadata lookup. --masks_dir bypasses the dataset class entirely
    # by reading precomputed MasksRLE files directly.
    if masks_dir:
        metadata_by_id = build_metadata_from_masks_rle(masks_dir, sampling_fps=sampling_fps)
    else:
        metadata_by_id = build_metadata_from_dataset(task, split)

    # 4. Match predictions to metadata.
    matched_metadatas = []
    matched_preds = []
    for pred_entry in predictions_json:
        eid = pred_entry['example_id']
        meta = metadata_by_id.get(eid)
        if meta is not None:
            matched_metadatas.append(meta)
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

    # 6. Run evaluators at each mask scale tier
    all_metrics = {}
    for prefix, scale in MASK_SCALES:
        log.info(f"Running eval with mask_scale={scale} ({prefix})")
        for metric in evaluators:
            results = metric(matched_metadatas, predictions, step=None, tokenizer=None, mask_scale=scale)
            for k, v in results.items():
                all_metrics[f"{prefix}/{k}"] = v

    # 8. Resolve metrics (MeanMetric -> float)
    resolved_metrics = {}
    for k in sorted(all_metrics):
        v = all_metrics[k]
        if isinstance(v, (int, float)):
            resolved_metrics[k] = v
        elif isinstance(v, torchmetrics.Metric):
            resolved_metrics[k] = v.compute().item()
        else:
            # Skip non-numeric metrics (HtmlTable, List, etc.)
            log.info(f"Skipping non-numeric metric {k}: {type(v).__name__}")

    # 9. Log to console
    log_metrics_to_console(task, resolved_metrics)

    # 10. Write metrics.json
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        metrics_path = join(out_dir, "metrics.json")
    else:
        metrics_path = predictions_path[:predictions_path.rfind('/')] + "/metrics.json"
    if (not os.path.exists(metrics_path)) or overwrite:
        with open(metrics_path, 'w') as f:
            json.dump(resolved_metrics, f, indent=2)
    else:
        log.info(f"Metrics file {metrics_path} already exists, skipping overwrite")

    return resolved_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on predictions.json")
    parser.add_argument("--predictions", required=True, help="Path to predictions.json")
    parser.add_argument("--task", required=True, help="Task name (e.g. cfc_track_eval_2fps)")
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    parser.add_argument("--masks_dir", default=None,
                        help="Path to MasksRLE/ dir. If provided, builds metadata directly from "
                             "{video_id}/{qid}.json files (bypasses dataset class). Otherwise, "
                             "loads dataset via --task/--split.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing metrics.json if it exists")
    parser.add_argument("--out_dir", default=None,
                        help="Directory to write metrics.json into. Defaults to dir of --predictions.")
    parser.add_argument("--sampling_fps", type=int, default=None,
                        help="Pred cadence (Hz) for GT subsampling. Used only with --masks_dir.")
    args = parser.parse_args()

    prepare_cli_environment()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_eval(args.predictions, args.task, args.split, args.overwrite,
             masks_dir=args.masks_dir, out_dir=args.out_dir, sampling_fps=args.sampling_fps)
