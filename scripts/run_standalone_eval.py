"""Standalone evaluation: run task evaluators on a predictions.json file."""
import argparse
import json
import logging
import os
from collections import Counter, defaultdict
from os.path import join, exists

import numpy as np
import torchmetrics

from olmo.util import prepare_cli_environment, log_metrics_to_console
from olmo.eval.eval_utils import get_evaluator
from olmo.eval.evaluators import SavePredictions

log = logging.getLogger(__name__)

MASK_SCALES = [("strict", 1.0), ("lenient_1.5x", 1.5), ("lenient_2x", 2.0)]

CFC_VIDEO_FPS = 6


def build_metadata_from_gt_json(gt_json_path, data_dir, video_fps=CFC_VIDEO_FPS):
    """Build per-video metadata from a COCO annotation JSON + precomputed MasksRLE.

    This bypasses the dataset class, allowing eval on any split as long as
    the COCO JSON and MasksRLE files exist.

    Returns:
        metadata_by_id: {video_id: metadata_dict}
    """
    from olmo.data.academic_video_track_datasets import TrackingDataset

    with open(gt_json_path) as f:
        coco = json.load(f)

    # Group images by video
    images_by_video = {}
    for img in coco['images']:
        vid = img['file_name'][:img['file_name'].rfind('_')]
        images_by_video.setdefault(vid, []).append(img)

    # Group annotations by image_id
    annots_by_image = {}
    for ann in coco['annotations']:
        annots_by_image.setdefault(ann['image_id'], []).append(ann)

    masks_dir = join(data_dir, "MasksRLE")

    metadata_by_id = {}
    for video_id, images in sorted(images_by_video.items()):
        # Sort images by frame index (last number before .jpg)
        images = sorted(images, key=lambda x: int(x['file_name'].replace('.jpg', '').rsplit('_', 1)[-1]))
        height, width = images[0]['height'], images[0]['width']
        n_frames = len(images)
        image_id_to_frame = {img['id']: idx for idx, img in enumerate(images)}

        # Collect annotations for this video
        video_annots = []
        for img in images:
            video_annots.extend(annots_by_image.get(img['id'], []))

        fish_ids = sorted({ann['track_id'] for ann in video_annots})
        fish_id_to_obj = {fish_id: idx for idx, fish_id in enumerate(fish_ids)}

        # Build GT points (PointTrack format)
        bbox_lookup = {}
        for ann in video_annots:
            frame_idx = image_id_to_frame.get(ann['image_id'])
            if frame_idx is not None:
                bbox_lookup[(frame_idx, ann['track_id'])] = ann['bbox']

        points = []
        for frame_idx in range(n_frames):
            frame_points = {}
            for fish_id in fish_ids:
                bbox = bbox_lookup.get((frame_idx, fish_id))
                if bbox is None:
                    continue
                x, y, bw, bh = bbox
                frame_points[fish_id_to_obj[fish_id]] = {
                    'point': [x + bw / 2, y + bh / 2],
                    'occluded': False,
                }
            points.append({
                'frame': frame_idx,
                'time': frame_idx / video_fps,
                'points': frame_points,
            })

        # Load precomputed masks
        masks_path = join(masks_dir, f"{video_id}.json")
        masks = {}
        if exists(masks_path):
            with open(masks_path) as f:
                masks = json.load(f)
        else:
            log.warning(f"No MasksRLE for {video_id}, mask-based metrics will be zero")

        metadata_by_id[video_id] = {
            'example_id': video_id,
            'w': width,
            'h': height,
            'video_fps': video_fps,
            'video': video_id,
            'points': points,
            'masks': masks,
            'mask_id': [str(i) for i in range(len(fish_ids))],
        }

    log.info(f"Built metadata for {len(metadata_by_id)} videos from {gt_json_path}")
    return metadata_by_id


def build_metadata_from_dataset(task, split):
    """Build per-video metadata by loading the dataset class."""
    from olmo.data.get_dataset import get_dataset_by_name

    log.info(f"Loading dataset: {task} split={split}")
    dataset = get_dataset_by_name(task, split)
    log.info(f"Dataset size: {len(dataset)}")

    metadata_by_id = {}
    for i in range(len(dataset)):
        ex = dataset.get(i, None)
        ex_id = ex.get('metadata', {}).get('example_id', str(i))
        metadata_by_id[ex_id] = ex['metadata']

    return metadata_by_id


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


def compute_count_metrics(matched_metadatas, matched_preds):
    """Compare per-direction track counts between pred and GT."""
    all_gt_counts = {'left': [], 'right': [], 'stationary': []}
    all_pred_counts = {'left': [], 'right': [], 'stationary': []}

    for meta, pred_points in zip(matched_metadatas, matched_preds):
        if 'w' in meta:
            w = meta['w']
        elif 'image_size' in meta:
            w = meta['image_size'][0]
        else:
            continue

        gt_tracks = meta.get('points', [])

        # GT: group by obj_id, get (x,y) sequence per track
        gt_by_obj = defaultdict(list)
        for frame_entry in gt_tracks:
            for obj_id, pt in frame_entry['points'].items():
                gt_by_obj[obj_id].append(tuple(pt['point']))

        gt_dirs = Counter(classify_direction(pts, w) for pts in gt_by_obj.values())

        # Pred: group by obj_id from [[obj_id, t, x, y], ...]
        pred_by_obj = defaultdict(list)
        if isinstance(pred_points, list):
            for obj_id, t, x, y in pred_points:
                pred_by_obj[int(obj_id)].append((x, y))

        pred_dirs = Counter(classify_direction(pts, w) for pts in pred_by_obj.values())

        for d in ['left', 'right', 'stationary']:
            all_gt_counts[d].append(gt_dirs.get(d, 0))
            all_pred_counts[d].append(pred_dirs.get(d, 0))

    metrics = {}
    for d in ['left', 'right', 'stationary']:
        gt = all_gt_counts[d]
        pred = all_pred_counts[d]
        metrics[f'count/{d}_gt_total'] = sum(gt)
        metrics[f'count/{d}_pred_total'] = sum(pred)
        if gt:
            metrics[f'count/{d}_abs_error_mean'] = float(np.mean([abs(p - g) for p, g in zip(pred, gt)]))
    return metrics


def run_eval(predictions_path, task, split="test", overwrite=False, gt_json=None, data_dir=None):
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

    # 3. Build metadata lookup
    if gt_json:
        metadata_by_id = build_metadata_from_gt_json(gt_json, data_dir)
    else:
        metadata_by_id = build_metadata_from_dataset(task, split)

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

    # 6. Run evaluators at each mask scale tier
    all_metrics = {}
    for prefix, scale in MASK_SCALES:
        log.info(f"Running eval with mask_scale={scale} ({prefix})")
        for metric in evaluators:
            results = metric(matched_metadatas, predictions, step=None, tokenizer=None, mask_scale=scale)
            for k, v in results.items():
                all_metrics[f"{prefix}/{k}"] = v

    # 7. Compute count metrics
    count_metrics = compute_count_metrics(matched_metadatas, matched_preds)
    log.info(f"Count metrics: {count_metrics}")

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

    # Add count metrics (already floats/ints)
    resolved_metrics.update(count_metrics)

    # 9. Log to console
    log_metrics_to_console(task, resolved_metrics)

    # 10. Write metrics.json
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
    parser.add_argument("--gt_json", default=None,
                        help="COCO annotation JSON for GT (bypasses dataset class for metadata)")
    parser.add_argument("--data_dir", default="data/video_datasets/video_track/CFC",
                        help="Dataset root dir containing MasksRLE/ (used with --gt_json)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing metrics.json if it exists")
    args = parser.parse_args()

    prepare_cli_environment()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_eval(args.predictions, args.task, args.split, args.overwrite,
             gt_json=args.gt_json, data_dir=args.data_dir)
