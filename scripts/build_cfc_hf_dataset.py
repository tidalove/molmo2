"""Build and upload the CFC tracking + track-correction datasets to one HF repo.

Mirrors the allenai/molmo2-track-instruction layout: a single dataset repo with
one config per CFC dataset, train/validation splits, parquet under the hood.
Annotations are subsampled to 2 fps (native frames with frame % 3 == 0; frame
keeps its native 6 fps index, time = frame / 6). GT masks are embedded inline
as pycocotools RLE, one entry per kept frame (entry i <-> native frame 3*i).

Rerunnable: `push_to_hub` replaces each config's parquet shards in place, so
rerunning after editing source jsonls (or adding configs below) just updates
the hub repo. `--prune` additionally deletes hub configs that are no longer
defined here. See docs/cfc_hf_dataset.md for the full maintenance guide.

Usage:
    python scripts/build_cfc_hf_dataset.py --dry-run --max-rows 20
    python scripts/build_cfc_hf_dataset.py --repo-id tidalove/cfc-track-instruction
    python scripts/build_cfc_hf_dataset.py --configs cfc_track cfc_text --prune
"""
import argparse
import hashlib
import json
import logging
import os
import sys
from functools import lru_cache
from os.path import abspath, dirname, exists, join

os.environ.setdefault("MOLMO_DATA_DIR", "data")
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import datasets
from datasets import Dataset, DatasetDict, Features, Sequence, Value

from olmo.data.academic_video_track_datasets import (
    CFC,
    CFCCorrectionRealFull,
    CFCCorrectionRealNoInfo,
    CFCCorrectionRealVague,
    CFCCorrectionRealWrongOnly,
    CFCCorrectionSyntheticFull,
    CFCCorrectionSyntheticIncomplete,
    CFCCorrectionSyntheticNoInfo,
    CFCCorrectionSyntheticVague,
    CFCCorrectionSyntheticWrongOnly,
    CFCMultiTurn,
    CFCTargeted,
    CFCText,
    LocalTrackingDataset,
)

log = logging.getLogger("build_cfc_hf_dataset")

DEFAULT_REPO_ID = "tidalove/cfc-track-instruction"

# 2 fps annotations over 6 fps video: keep native frames 0, 3, 6, ...
VIDEO_FPS = 6
ANNOTATION_FPS = 2
KEEP_EVERY = VIDEO_FPS // ANNOTATION_FPS

V2_SPLITS = {"train": "all-rivers-train-v2", "validation": "all-rivers-val-v2"}

# config name (= existing DATASET_NAME) -> build spec.
#   family: row builder + schema ("track" | "target" | "correction" | "text")
#   cls: source class for SPLIT_TO_FILE / SPLIT_TO_FPS_SOURCE / ID_TAG
#   mask_mode: "video_gt" (base-video COCO GT, what the hardlinked MasksRLE hold),
#              "final_step" (GT = max correction step of the trajectory), or None
#   splits: hub split name -> local data_split
CONFIGS = {
    "cfc_track": dict(family="track", splits=V2_SPLITS),
    "cfc_target": dict(family="target", splits=V2_SPLITS),
    "cfc_synthetic_correction_full": dict(
        family="correction", cls=CFCCorrectionSyntheticFull, mask_mode="video_gt", splits=V2_SPLITS),
    "cfc_synthetic_correction_vague": dict(
        family="correction", cls=CFCCorrectionSyntheticVague, mask_mode="video_gt", splits=V2_SPLITS),
    "cfc_synthetic_correction_wrong_only": dict(
        family="correction", cls=CFCCorrectionSyntheticWrongOnly, mask_mode="video_gt", splits=V2_SPLITS),
    "cfc_synthetic_correction_no_info": dict(
        family="correction", cls=CFCCorrectionSyntheticNoInfo, mask_mode="video_gt", splits=V2_SPLITS),
    "cfc_synthetic_correction_incomplete": dict(
        family="correction", cls=CFCCorrectionSyntheticIncomplete, mask_mode="final_step", splits=V2_SPLITS),
    "cfc_correction_real_full": dict(
        family="correction", cls=CFCCorrectionRealFull, mask_mode="video_gt", splits=V2_SPLITS),
    "cfc_correction_real_wrong_only": dict(
        # no val jsonl exists for this tier — train only
        family="correction", cls=CFCCorrectionRealWrongOnly, mask_mode="video_gt",
        splits={"train": "all-rivers-train-v2"}),
    "cfc_correction_real_vague": dict(
        family="correction", cls=CFCCorrectionRealVague, mask_mode="video_gt", splits=V2_SPLITS),
    "cfc_correction_real_no_info": dict(
        family="correction", cls=CFCCorrectionRealNoInfo, mask_mode="video_gt", splits=V2_SPLITS),
    "cfc_text": dict(
        family="text", cls=CFCText, mask_mode=None,
        splits={"train": "kenai-train", "validation": "kenai-val"}),
}

# ── Schemas ────────────────────────────────────────────────────────────────
# float64 for coordinates/time so hub values match the source jsons exactly.

RLE = {"size": Sequence(Value("int32")), "counts": Value("string")}
POINT = {"id": Value("int32"), "point": Sequence(Value("float64")), "occluded": Value("bool")}
FRAME_TRAJ = {"frame": Value("int32"), "time": Value("float64"), "points": [POINT]}
MASKS = [{"object_id": Value("string"), "masks": [RLE]}]

_BASE = {
    "id": Value("string"),
    "video": Value("string"),
    "clip": Value("string"),
    "video_dataset": Value("string"),
    "task": Value("string"),
    "expression": Value("string"),
    "qid": Value("string"),
    "width": Value("int32"),
    "height": Value("int32"),
    "fps": Value("int32"),
    "sampling_fps": Value("int32"),
    "n_frames": Value("int32"),
    "start_frame": Value("int32"),
    "end_frame": Value("int32"),
    "mask_id": [Value("string")],
    "obj_id": [Value("string")],
}

TRACK_FEATURES = Features({
    **_BASE,
    "prepend": Value("string"),
    "anno_id": [Value("string")],
    "frame_trajectories": [FRAME_TRAJ],
    "masks": MASKS,
})

_TURN = {"correction_step": Value("int32"), "prompt": Value("string"),
         "frame_trajectories": [FRAME_TRAJ]}

CORRECTION_FEATURES = Features({
    **_BASE,
    "confidence": Value("string"),
    "turns": [_TURN],
    "masks": MASKS,
})

# text corrections have no GT masks (no MasksRLE exist for these ids locally)
TEXT_FEATURES = Features({
    **_BASE,
    "confidence": Value("string"),
    "turns": [_TURN],
})

FAMILY_FEATURES = {
    "track": TRACK_FEATURES,
    "target": TRACK_FEATURES,
    "correction": CORRECTION_FEATURES,
    "text": TEXT_FEATURES,
}

# ── Shared helpers ─────────────────────────────────────────────────────────

def subsample_frame_trajectories(frame_trajectories, keep_every=KEEP_EVERY):
    """Keep annotation frames at 2 fps. Native frame index and time are kept."""
    return [ft for ft in frame_trajectories if ft["frame"] % keep_every == 0]


def points_dict_to_list(points_dict):
    """{slot: {point, occluded}} (CFCMultiTurn form) -> sorted [{id, point, occluded}]."""
    return [
        {"id": int(slot), "point": [float(v) for v in p["point"]], "occluded": bool(p["occluded"])}
        for slot, p in sorted(points_dict.items())
    ]


@lru_cache(maxsize=4)
def load_coco_index(data_split):
    """Split COCO grouped per video. Mirrors CFC.load (academic_video_track_datasets.py:1104).

    Returns ({video_id: (sorted_images, video_annots)}, {video_id: fps}).
    """
    coco = CFC._load_coco_json(data_split)
    fps_map = {v["id"]: v["fps"] for v in coco.get("videos", [])}

    images_by_video = {}
    for img in coco["images"]:
        images_by_video.setdefault(img["file_name"][:img["file_name"].rfind("_")], []).append(img)
    annots_by_image = {}
    for ann in coco["annotations"]:
        annots_by_image.setdefault(ann["image_id"], []).append(ann)

    videos = {}
    for video_id, images in images_by_video.items():
        images = sorted(
            images,
            key=lambda x: int(x["file_name"][x["file_name"].rfind("_") + 1:x["file_name"].rfind(".")]))
        video_annots = []
        for img in images:
            video_annots.extend(annots_by_image.get(img["id"], []))
        videos[video_id] = (images, video_annots)
    return videos, fps_map


def build_masks(images, annotations, track_ids):
    """Inline RLE masks: [{object_id: slot, masks: [rle|None per kept frame]}].

    Slot order follows the given track_ids ordering; entry i corresponds to
    native frame KEEP_EVERY * i.
    """
    height, width = images[0]["height"], images[0]["width"]
    image_id_to_frame = {img["id"]: idx for idx, img in enumerate(images)}
    bbox_lookup = {}
    for ann in annotations:
        fidx = image_id_to_frame.get(ann["image_id"])
        if fidx is not None:
            bbox_lookup[(fidx, ann["track_id"])] = ann["bbox"]

    kept_frames = range(0, len(images), KEEP_EVERY)
    out = []
    for slot, tid in enumerate(track_ids):
        frame_masks = []
        for fidx in kept_frames:
            bbox = bbox_lookup.get((fidx, tid))
            frame_masks.append(
                None if bbox is None
                else LocalTrackingDataset._bbox_to_rle(bbox, height, width))
        out.append({"object_id": str(slot), "masks": frame_masks})
    return out


_video_gt_cache = {}

def video_gt_masks(data_split, video_id):
    """Base-video COCO GT masks (slot order = sorted video track_ids), memoized so
    the tiers sharing hardlinked MasksRLE also share computation here."""
    key = (data_split, video_id)
    if key not in _video_gt_cache:
        videos, _ = load_coco_index(data_split)
        if video_id not in videos:
            log.warning(f"video {video_id} not in COCO split {data_split}; empty masks")
            _video_gt_cache[key] = []
        else:
            images, annots = videos[video_id]
            track_ids = sorted({a["track_id"] for a in annots})
            _video_gt_cache[key] = build_masks(images, annots, track_ids)
    return _video_gt_cache[key]


def source_files(config_name):
    """Absolute source paths feeding a config, per split. Filenames come from the
    existing classes' SPLIT_TO_FILE (never glob — .bak/-old/incl/excl traps)."""
    spec = CONFIGS[config_name]
    files = {}
    for hub_split, data_split in spec["splits"].items():
        paths = []
        if spec["family"] in ("track", "target"):
            paths.append(CFC._get_anno_path(data_split))
            if spec["family"] == "target":
                paths.append(CFCTargeted._get_queries_path(data_split))
        else:
            cls = spec["cls"]
            paths.append(join(CFC.VIDEO_HOME, "caption_annotations", cls.SPLIT_TO_FILE[data_split]))
            fps_split = getattr(cls, "SPLIT_TO_FPS_SOURCE", {}).get(data_split, data_split)
            paths.append(join(CFC.VIDEO_HOME, "annotations", f"{fps_split}.json"))
        files[hub_split] = paths
    return files


def source_token(config_name, hub_split):
    """Hash of (path, size, mtime) of all inputs — passed as a gen_kwarg so the
    datasets fingerprint (and cache) invalidates when any source file changes."""
    h = hashlib.md5()
    for p in source_files(config_name)[hub_split]:
        st = os.stat(p)  # follows the kenai symlinks
        h.update(f"{p}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


# ── Row generators (one per family) ────────────────────────────────────────
# All share the signature (config_name, data_split, source_token, max_rows);
# source_token is unused inside — it only feeds the cache fingerprint.

def gen_track_rows(config_name, data_split, source_token=None, max_rows=None):
    videos, fps_map = load_coco_index(data_split)
    n = 0
    for video_id in sorted(videos):
        images, annots = videos[video_id]
        fps = fps_map.get(video_id) or CFC.VIDEO_FPS
        ex = CFC._build_video_annotation(video_id, images, annots, fps=fps)
        yield {
            "id": ex["id"],
            "video": video_id,
            "clip": video_id,
            "video_dataset": "cfc",
            "task": "track",
            "expression": ex["expression"],
            "qid": ex["qid"],
            "prepend": ex.get("prepend") or "",
            "width": ex["width"],
            "height": ex["height"],
            "fps": int(fps),
            "sampling_fps": ANNOTATION_FPS,
            "n_frames": len(images),
            "start_frame": 0,
            "end_frame": len(images) - 1,
            "mask_id": ex["mask_id"],
            "obj_id": ex["obj_id"],
            "anno_id": ex["anno_id"],
            "frame_trajectories": subsample_frame_trajectories(ex["frame_trajectories"]),
            "masks": video_gt_masks(data_split, video_id),
        }
        n += 1
        if max_rows and n >= max_rows:
            return


def gen_target_rows(config_name, data_split, source_token=None, max_rows=None):
    videos, fps_map = load_coco_index(data_split)
    queries = CFCTargeted._load_queries(CFCTargeted._get_queries_path(data_split))
    n = 0
    for video_id in sorted(videos):
        images, annots = videos[video_id]
        fps = fps_map.get(video_id) or CFC.VIDEO_FPS
        # default qid=0 "track all fish" + one row per query (mirrors CFCTargeted.load)
        per_video = [(CFCTargeted._build_video_annotation(video_id, images, annots, qid=0), None)]
        for q in queries.get(video_id, []):
            per_video.append((
                CFCTargeted._build_video_annotation(
                    video_id, images, annots,
                    qid=q["qid"], expression=q["expression"], target_ids=q["target_ids"]),
                sorted(q["target_ids"]),
            ))
        for ex, target_ids in per_video:
            if target_ids is None:
                masks = video_gt_masks(data_split, video_id)
            else:
                masks = build_masks(images, annots, target_ids)
            yield {
                "id": ex["id"],
                "video": video_id,
                "clip": video_id,
                "video_dataset": "cfc",
                "task": "track",
                "expression": ex["expression"],
                "qid": ex["qid"],
                "prepend": ex.get("prepend") or "",
                "width": ex["width"],
                "height": ex["height"],
                "fps": int(fps),
                "sampling_fps": ANNOTATION_FPS,
                "n_frames": len(images),
                "start_frame": 0,
                "end_frame": len(images) - 1,
                "mask_id": ex["mask_id"],
                "obj_id": ex["obj_id"],
                "anno_id": ex["anno_id"],
                "frame_trajectories": subsample_frame_trajectories(ex["frame_trajectories"]),
                "masks": masks,
            }
            n += 1
            if max_rows and n >= max_rows:
                return


def gen_correction_rows(config_name, data_split, source_token=None, max_rows=None):
    spec = CONFIGS[config_name]
    cls = spec["cls"]
    jsonl_path = join(CFC.VIDEO_HOME, "caption_annotations", cls.SPLIT_TO_FILE[data_split])
    fps_split = getattr(cls, "SPLIT_TO_FPS_SOURCE", {}).get(data_split, data_split)

    fps_map = {}
    coco_path = join(CFC.VIDEO_HOME, "annotations", f"{fps_split}.json")
    if exists(coco_path):
        with open(coco_path) as f:
            fps_map = {v["id"]: v["fps"] for v in json.load(f).get("videos", [])}

    n = 0
    with open(jsonl_path) as f:  # streamed — files reach 700MB
        for line in f:
            record = json.loads(line)
            video_id = record["video_name"]
            native_fps = fps_map.get(video_id, CFC.VIDEO_FPS)
            images = sorted(record["images"], key=lambda x: x["id"])
            for traj in record["trajectories"]:
                # raw (frame-indexed) prompts on the hub; loaders rewrite to time
                ex = CFCMultiTurn._build_video_annotation(video_id, images, traj["correction_steps"])
                example_id = f"{video_id}{cls.ID_TAG}_traj{traj['trajectory_id']}"
                turns = []
                for step, prompt in enumerate(ex["prompts_list"]):
                    kept = subsample_frame_trajectories(ex["points_list"][step])
                    turns.append({
                        "correction_step": step,
                        "prompt": prompt,
                        "frame_trajectories": [
                            {"frame": ft["frame"], "time": ft["time"],
                             "points": points_dict_to_list(ft["points"])}
                            for ft in kept
                        ],
                    })
                row = {
                    "id": example_id,
                    "video": video_id,
                    "clip": video_id,
                    "video_dataset": "cfc",
                    "task": "track",
                    "expression": ex["expression"],
                    "qid": example_id,
                    "confidence": str(record.get("confidence") or ""),
                    "width": ex["width"],
                    "height": ex["height"],
                    "fps": int(native_fps),
                    "sampling_fps": ANNOTATION_FPS,
                    "n_frames": len(images),
                    "start_frame": 0,
                    "end_frame": len(images) - 1,
                    "mask_id": ex["mask_id"],
                    "obj_id": ex["obj_id"],
                    "turns": turns,
                }
                if spec["mask_mode"] == "video_gt":
                    row["masks"] = video_gt_masks(fps_split, video_id)
                elif spec["mask_mode"] == "final_step":
                    gt_step = max(traj["correction_steps"], key=lambda s: s["correction_step"])
                    tids = sorted({a["track_id"] for a in gt_step["annotations"]})
                    row["masks"] = build_masks(images, gt_step["annotations"], tids)
                yield row
                n += 1
                if max_rows and n >= max_rows:
                    return


FAMILY_GENERATORS = {
    "track": gen_track_rows,
    "target": gen_target_rows,
    "correction": gen_correction_rows,
    "text": gen_correction_rows,  # same jsonl format, schema just drops masks
}

# ── Build / push ───────────────────────────────────────────────────────────

def build_config(config_name, max_rows=None, cache_dir=None):
    spec = CONFIGS[config_name]
    splits = {}
    for hub_split, data_split in spec["splits"].items():
        for p in source_files(config_name)[hub_split]:
            assert exists(p), f"[{config_name}/{hub_split}] missing source: {p}"
        ds = Dataset.from_generator(
            FAMILY_GENERATORS[spec["family"]],
            features=FAMILY_FEATURES[spec["family"]],
            gen_kwargs=dict(
                config_name=config_name,
                data_split=data_split,
                source_token=source_token(config_name, hub_split),
                max_rows=max_rows,
            ),
            cache_dir=cache_dir,
        )
        splits[hub_split] = ds
        log.info(f"[{config_name}/{hub_split}] {len(ds)} rows, "
                 f"{ds.data.nbytes / 1e6:.1f}MB in-arrow")
    return DatasetDict(splits)


def prune_stale_configs(api, repo_id, keep):
    """Delete hub config dirs (and their README yaml entries) not built anymore."""
    from huggingface_hub import DatasetCard
    files = api.list_repo_files(repo_id, repo_type="dataset")
    on_hub = {f.split("/")[0] for f in files if "/" in f and f.endswith(".parquet")}
    stale = sorted(on_hub - set(keep))
    if not stale:
        log.info("prune: nothing stale on the hub")
        return
    for name in stale:
        log.info(f"prune: deleting hub folder {name}/")
        api.delete_folder(path_in_repo=name, repo_id=repo_id, repo_type="dataset",
                          commit_message=f"Prune stale config {name}")
    card = DatasetCard.load(repo_id, repo_type="dataset")
    for key in ("configs", "dataset_info"):
        entries = card.data.get(key)
        if entries:
            card.data[key] = [e for e in entries if e.get("config_name") not in stale]
    card.push_to_hub(repo_id, repo_type="dataset",
                     commit_message=f"Prune stale configs: {', '.join(stale)}")


def upload_readme(repo_id):
    from huggingface_hub import DatasetCard
    card = DatasetCard.load(repo_id, repo_type="dataset")  # keeps autogen yaml
    card.text = README_BODY
    card.push_to_hub(repo_id, repo_type="dataset", commit_message="Update dataset card")
    log.info(f"README updated on {repo_id}")


README_BODY = f"""
# CFC Track & Track-Correction Instruction Data

Fish tracking and multi-turn track-correction annotations on the
[Caltech Fish Counting](https://github.com/visipedia/caltech-fish-counting)
sonar videos, in the
[Molmo2 video-track-instruction](https://huggingface.co/datasets/allenai/molmo2-track-instruction)
format. Companion to the Molmo2 codebase's `olmo/data/cfc_hf_datasets.py`
loader classes.

Videos are 6 fps clips; **annotations are stored at 2 fps**: only native frames
with `frame % 3 == 0` are annotated, `frame` keeps the native 6 fps index, and
`time = frame / 6`. Inline `masks` follow the same convention — mask entry `i`
corresponds to native frame `3 * i`.

## Configs

| config | format | splits |
|---|---|---|
| `cfc_track` | track (one row per clip, "track all fish") | train / validation |
| `cfc_target` | track (one row per query; qid `0` = all fish, others = referred subsets) | train / validation |
| `cfc_synthetic_correction_full` / `_vague` / `_wrong_only` / `_no_info` | correction, synthetically corrupted step 0 | train / validation |
| `cfc_synthetic_correction_incomplete` | correction, partial-fix target (GT = final step, NOT the full video GT) | train / validation |
| `cfc_correction_real_full` / `_vague` / `_no_info` | correction, real model-prediction step 0 | train / validation |
| `cfc_correction_real_wrong_only` | correction (real) | train only |
| `cfc_text` | text-only correction (no masks, no video input at train time) | train / validation |

## Schema

Track configs: `id, video, clip, video_dataset, task, expression, qid, prepend,
width, height, fps (6), sampling_fps (2), n_frames, start_frame, end_frame,
mask_id, obj_id, anno_id, frame_trajectories, masks`.

`frame_trajectories`: list of `{{frame, time, points: [{{id, point: [x, y], occluded}}]}}`.
Point `id` indexes into `mask_id`/`obj_id` slots.

Correction configs replace `prepend`/`anno_id`/`frame_trajectories` with
`confidence` and `turns`: list of
`{{correction_step, prompt, frame_trajectories}}` — a multi-turn conversation
where step 0 is the (possibly corrupted or model-predicted) starting tracks and
later steps are correction targets. **Prompts are stored raw with native frame
references**; loaders rewrite them to timestamps (`frame N` -> `{{N/fps}}s`).
Per-turn point `id`s are per-step slot indices (sorted track ids of that step).

`masks`: list of `{{object_id, masks: [RLE | null]}}` — pycocotools RLE
(`{{size: [h, w], counts}}`), `null` = object absent. For most correction
configs these are the base-video GT masks (slot order = sorted video track
ids, which may differ from per-turn point slots — consumers should match by
mask content, as the Molmo2 eval does). For `cfc_synthetic_correction_incomplete`
they encode the final correction step per trajectory.

## Usage

```python
import datasets
ds = datasets.load_dataset("{DEFAULT_REPO_ID}", "cfc_track", split="validation")
```

With the Molmo2 codebase, `olmo/data/cfc_hf_datasets.py` downloads all configs,
rehydrates local `MasksRLE/` files, and (given frames) encodes videos:
place frames at
`$MOLMO_DATA_DIR/video_datasets/video_track/CFC/JPEGImages/{{video_id}}/*.jpg`,
then call `download()` on any of the loader classes (or use the registered
`cfc_hf_*` dataset names).

Built by `scripts/build_cfc_hf_dataset.py`; maintenance guide in
`docs/cfc_hf_dataset.md`.
""".strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Subset of configs to build (default: all)")
    parser.add_argument("--public", action="store_true",
                        help="Create the repo public (default private)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build only; no hub interaction")
    parser.add_argument("--prune", action="store_true",
                        help="Delete hub configs no longer defined in CONFIGS")
    parser.add_argument("--out-dir", default=None,
                        help="save_to_disk() each built config here (inspection)")
    parser.add_argument("--local-cache", action="store_true",
                        help="Also save_to_disk() into the loader classes' runtime "
                             "cache (CFC/hf_annotations/{config}/{split}) so "
                             "cfc_hf_datasets.py works without hitting the hub")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap rows per split (smoke tests)")
    parser.add_argument("--cache-dir", default=None,
                        help="datasets cache dir for from_generator")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    config_names = args.configs or list(CONFIGS)
    unknown = set(config_names) - set(CONFIGS)
    assert not unknown, f"Unknown configs: {unknown}. Known: {list(CONFIGS)}"

    for name in config_names:
        dsd = build_config(name, max_rows=args.max_rows, cache_dir=args.cache_dir)
        if args.out_dir:
            dsd.save_to_disk(join(args.out_dir, name))
        if args.local_cache:
            for hub_split, ds in dsd.items():
                target = join(CFC.VIDEO_HOME, "hf_annotations", name, hub_split)
                ds.save_to_disk(target)
                log.info(f"[{name}/{hub_split}] cached to {target}")
        if not args.dry_run:
            dsd.push_to_hub(args.repo_id, config_name=name, private=not args.public)
            log.info(f"[{name}] pushed to {args.repo_id}")

    if not args.dry_run:
        from huggingface_hub import HfApi
        api = HfApi()
        if args.prune:
            prune_stale_configs(api, args.repo_id, keep=list(CONFIGS))
        upload_readme(args.repo_id)


if __name__ == "__main__":
    main()
