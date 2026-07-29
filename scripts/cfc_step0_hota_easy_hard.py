"""Step-0 (pre-correction) HOTA for the CFC real-correction easy/hard full sets.

Read-only sanity check that the -easy / -hard train/val jsonls hold the step-0
tracks we think they do: scores every trajectory's correction_step==0
annotations as predictions against the base-video GT masks in
MasksRLE/{video_name}/0.json, at the standard 2 fps eval cadence.

Reference: also prints the mean hota_before stored in the old-real predictions
(runs/cfc_all_real_llm_connector_vit/results/old-real/.../predictions.json),
which came from the same -hard (formerly -old) val prompts — the hard-val row
should match it closely.

Usage:
    python scripts/cfc_step0_hota_easy_hard.py [--files ...] [--limit N]
"""
import argparse
import json
import os
import sys
from os.path import abspath, dirname, join

os.environ.setdefault("MOLMO_DATA_DIR", "data")
sys.path.insert(0, dirname(dirname(abspath(__file__))))
sys.path.insert(0, dirname(abspath(__file__)))

from cfc_correction_common import (
    build_corrupted_pred_tracks, build_metadata_one, cfc_aggregate, eval_record_from_tracks,
)

MASKS_HOME = "data/video_datasets/video_track/CFC/MasksRLE"
ANNO_DIR = "data/video_datasets/video_track/CFC/caption_annotations"
VIDEO_FPS = 6
SAMPLING_FPS = 2

DEFAULT_FILES = [
    join(ANNO_DIR, "cfc_real_correction_full_train-easy.jsonl"),
    join(ANNO_DIR, "cfc_real_correction_full_val-easy.jsonl"),
    join(ANNO_DIR, "cfc_real_correction_full_train-hard.jsonl"),
    join(ANNO_DIR, "cfc_real_correction_full_val-hard.jsonl"),
]

OLD_REAL_REFERENCE = ("runs/cfc_all_real_llm_connector_vit/results/old-real/"
                      "cfc_correction_real_full_eval_2fps/validation-v2/predictions.json")


def score_file(path, limit=None):
    records = []
    n_videos = n_missing_mask = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            video_name = rec["video_name"]
            n_videos += 1
            if limit and n_videos > limit:
                n_videos -= 1
                break
            meta = build_metadata_one(MASKS_HOME, video_name, VIDEO_FPS, SAMPLING_FPS)
            if meta is None:
                n_missing_mask += 1
                continue
            meta["example_id"] = video_name

            images = sorted(rec["images"], key=lambda im: im["id"])
            image_id_to_frame = {im["id"]: i for i, im in enumerate(images)}
            stride = VIDEO_FPS // SAMPLING_FPS
            for traj in rec["trajectories"]:
                step0 = min(traj["correction_steps"], key=lambda s: s["correction_step"])
                pred_tracks = build_corrupted_pred_tracks(step0, image_id_to_frame, VIDEO_FPS)
                # the jsonls annotate every native frame; the prompts the model
                # actually sees (and hota_before) are on the 2 fps grid, so drop
                # the frames GT is subsampled away from
                pred_tracks = [p for p in pred_tracks if p["frame"] % stride == 0]
                records.append(eval_record_from_tracks(meta, pred_tracks))

    agg, n = cfc_aggregate(records)
    return agg, n, n_videos, n_missing_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max videos per file (for a quick pass)")
    args = parser.parse_args()

    print(f"{'file':<52} {'trajs':>6} {'videos':>6} {'no-mask':>7} "
          f"{'HOTA':>7} {'DetA':>7} {'AssA':>7} {'coco_f1':>8}")
    for path in args.files:
        agg, n, n_videos, n_missing = score_file(path, args.limit)
        name = os.path.basename(path)
        if n == 0:
            print(f"{name:<52} {'-':>6} {n_videos:>6} {n_missing:>7}  NO RECORDS SCORED")
            continue
        print(f"{name:<52} {n:>6} {n_videos:>6} {n_missing:>7} "
              f"{agg['HOTA']:>7.4f} {agg['DetA']:>7.4f} {agg['AssA']:>7.4f} "
              f"{agg['coco_f1']:>8.4f}")

    if os.path.exists(OLD_REAL_REFERENCE):
        with open(OLD_REAL_REFERENCE) as f:
            preds = json.load(f)
        befores = [p["hota_before"] for p in preds if p.get("hota_before") is not None]
        if befores:
            print(f"\nreference: mean hota_before over {len(befores)} old-real full-val "
                  f"predictions = {sum(befores) / len(befores):.4f}")
            print("(hard-val step0 HOTA above should be close to this)")


if __name__ == "__main__":
    main()
