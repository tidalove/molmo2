# CFC HuggingFace dataset: build, upload, and maintenance

The CFC tracking + track-correction annotations are published as **one HF
dataset repo** (default `tidalove/cfc-track-instruction`, private) in the
molmo2 track-instruction style: one **config** per dataset, splits
`train`/`validation`, parquet with inline RLE masks. Three files own the whole
pipeline:

| file | role |
|---|---|
| `scripts/build_cfc_hf_dataset.py` | reads the source jsons → builds parquet → pushes to the hub (rerunnable) |
| `olmo/data/cfc_hf_datasets.py` | loader classes (`cfc_hf_*`) that download from the hub instead of local jsons |
| `scripts/verify_cfc_hf_dataset.py` | asserts old-vs-new parity + hub smoke test |

Loader names are registered in `olmo/data/get_dataset.py` (`cfc_hf_track`,
`cfc_hf_correction_real_full_eval_2fps`, …) — same naming scheme as the local
`cfc_*` registrations they mirror.

## Conventions baked into the release

- **Videos 6 fps, annotations 2 fps.** Only native frames with
  `frame % 3 == 0` carry annotations. `frame` keeps its native index,
  `time = frame / 6`, rows have `fps = 6`, `sampling_fps = 2`.
- **Masks inline.** `masks = [{object_id, masks: [RLE | null]}]`
  (pycocotools RLE, `counts` utf-8). Mask entry `i` ↔ native frame `3i`.
  `download()` rehydrates them into the usual local
  `MasksRLE/{id-or-video}/{qid}.json` layout, **write-if-missing** (never
  clobbers existing files).
- **Correction prompts are stored raw** (frame-indexed, exactly as in the
  jsonls). The loaders apply `replace_frames_with_time` at load, so runtime
  prompts match the local classes.
- **Filenames come from the source classes' `SPLIT_TO_FILE`** dicts in
  `academic_video_track_datasets.py` — the builder never globs (avoids
  `.bak` / `-old` / `_incl` / `_excl` traps).
- Video frames are *not* on the hub. A fresh machine puts frames at
  `$MOLMO_DATA_DIR/video_datasets/video_track/CFC/JPEGImages/{video_id}/*.jpg`
  and calls `download()` (encodes `CFC/videos/{video_id}.mp4` via ffmpeg).

## Rebuilding / re-uploading (the common case)

```bash
# smoke test (no hub interaction)
python scripts/build_cfc_hf_dataset.py --dry-run --max-rows 20

# full rebuild + push of everything (also refresh the local runtime cache)
python scripts/build_cfc_hf_dataset.py --local-cache

# just one config after editing its source jsonl
python scripts/build_cfc_hf_dataset.py --configs cfc_correction_real_full --local-cache

# remove hub configs that no longer exist in CONFIGS
python scripts/build_cfc_hf_dataset.py --prune
```

Notes:
- `push_to_hub` **replaces the config's parquet shards in place** — rerunning
  is the intended update path; nothing needs deleting first.
- The heavy build (json parsing + RLE encoding) belongs on a compute node.
  Trick used in practice: build on a compute node with
  `--dry-run --local-cache --cache-dir /path/shared/cache`, then rerun on the
  login node *without* `--dry-run` and the same `--cache-dir` — the second run
  hits the `from_generator` cache and only uploads. The cache auto-invalidates
  when a source file's size/mtime changes (see `source_token`).
- Loader-side caches live at `CFC/hf_annotations/{config}/{split}`
  (`save_to_disk` format). Loaders prefer this cache over the hub; delete a
  config's cache dir (or pass `overwrite_cache=True`) after a re-upload if the
  machine already had the old version.
- After any rebuild:
  `python scripts/verify_cfc_hf_dataset.py --configs <new_name>_eval_2fps` and
  optionally `--hub`.

## Adding a new dataset

### 1. Track format (per-clip COCO GT, like `cfc_track`)

Source: a per-split COCO json in `CFC/annotations/{split}.json` with
`videos` (per-clip fps), `images` (`file_name = {video_id}_{frame}.jpg`),
`annotations` (bbox + `track_id`).

1. In `build_cfc_hf_dataset.py` add to `CONFIGS`:
   ```python
   "my_track": dict(family="track", splits={"train": "my-train", "validation": "my-val"}),
   ```
   The `track` family builds rows via `CFC._build_video_annotation` from
   `annotations/{data_split}.json`; nothing else needed if your COCO follows
   the CFC layout. (Different annotation dir/schema → add a generator, see §4.)
2. In `cfc_hf_datasets.py` add a loader:
   ```python
   class MyTrackHF(CFCTrackHF):
       DATASET_NAME = "cfc_hf_my_track"
       HF_CONFIG = "my_track"
   ```
3. Register `cfc_hf_my_track` (+ `_eval_2fps`) in `get_dataset.py`.
4. Rebuild: `python scripts/build_cfc_hf_dataset.py --configs my_track --local-cache`.

### 2. Queries format (referred subsets, like `cfc_target`)

Source: the COCO above **plus** a queries json
(`{video_id: [{qid, expression, target_ids}]}`) in `caption_annotations/`.

Same steps with `family="target"`. The `target` family resolves the queries
file via `CFCTargeted._get_queries_path`; a *different* queries file means
either extending that classmethod's mapping in the source class or giving the
`CONFIGS` entry an explicit generator (§4). Loader subclasses `CFCTargetedHF`.

### 3. Correction format (multi-turn, like `cfc_correction_real_*`)

Source: a jsonl in `caption_annotations/`, one record per video:
`{video_name, confidence, images, categories,
trajectories: [{trajectory_id, correction_steps: [{correction_step, prompt,
annotations}]}]}`.

1. Define the source class in `academic_video_track_datasets.py` as usual
   (subclass `CFCCorrection`/`CFCCorrectionReal`, set `DATASET_NAME`,
   `ID_TAG`, `SPLIT_TO_FILE`) — the builder only reads its `SPLIT_TO_FILE`,
   `SPLIT_TO_FPS_SOURCE`, and `ID_TAG`.
2. Add to `CONFIGS`:
   ```python
   "my_correction": dict(family="correction", cls=MyCorrectionClass,
                         mask_mode="video_gt", splits=V2_SPLITS),
   ```
   `mask_mode`: `"video_gt"` when GT = the base video's COCO annotations
   (synthetic-complete and real tiers — what the hardlinked MasksRLE hold);
   `"final_step"` when GT = the trajectory's last correction step
   (incomplete-style, where the target is *not* the full video GT);
   `None` for no masks (text-style).
3. Loader:
   ```python
   class MyCorrectionHF(CFCCorrectionHFBase):
       DATASET_NAME = "cfc_hf_my_correction"
       HF_CONFIG = "my_correction"
   ```
   Train-only tier? Set `SPLIT_MAP = {"train-v2": "train", "train": "train"}`
   (see `CFCCorrectionRealWrongOnlyHF`).
4. Register + rebuild as above. Add the pair to `PAIRS` in
   `verify_cfc_hf_dataset.py` if a local twin exists.

### 4. A format that doesn't fit the three families

Add a generator function in `build_cfc_hf_dataset.py` with the signature
`(config_name, data_split, source_token=None, max_rows=None)`, a `Features`
schema (reuse `RLE` / `POINT` / `FRAME_TRAJ` / `MASKS` and `_BASE`), and wire
both into `FAMILY_GENERATORS` / `FAMILY_FEATURES` under a new family name.
Also extend `source_files()` so cache invalidation and existence checks see
your inputs.

## What's on the hub (as of 2026-07)

12 configs: `cfc_track`, `cfc_target`,
`cfc_synthetic_correction_{full,vague,wrong_only,no_info,incomplete}`,
`cfc_correction_real_{full,wrong_only,vague,no_info}` (wrong_only train-only),
`cfc_text` (kenai text corrections, no masks, no videos). Eel and
kenai-channel are deliberately excluded.
