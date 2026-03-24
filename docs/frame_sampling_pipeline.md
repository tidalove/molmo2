# Frame Sampling Pipeline for LocalTrackingDataset

How video frames get loaded, filtered, and turned into prompts for tracking tasks (CFC, PanAf, etc.). Covers train vs eval, what the model sees vs what it predicts on, and how `max_frames`, `max_fps`, and `sampling_fps` interact.

## TL;DR

There are **three distinct fps concepts** in the pipeline:

1. **`VIDEO_FPS`** — the native fps of the encoded `.mp4` (e.g. CFC=24, PanAf=18). Set offline, never changes at runtime.
2. **Loaded fps (`target_fps`)** — how many frames the model actually *sees*. Determined at load time by `TimeSampler` based on `candidate_sampling_fps`, `max_frames`, and video duration. Varies per-video.
3. **`sampling_fps`** — the fps of the *text output* the model is asked to produce. Controls `_sample_at_fps` in the formatter and the "{fps}" in the prompt. Set by the dataset task (1 for `_eval_1fps`, 8 for `_eval_8fps`, resolved from `target_fps` for training). **Capped at `max_output_fps` (default 2)** since the model was trained on ≤2fps output.

The model always sees **more frames than it predicts on** (or equal). Loaded fps >= sampling_fps.

---

## End-to-End Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Dataset.get(idx)                                                │
│    - Builds annotation point-tracks at VIDEO_FPS (all native frames)│
│    - Sets sampling_fps: explicit value (eval) or None (training)   │
│    - Constructs sampler_overrides: {frame_sample_mode, candidates} │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│ 2. VideoLoader (TimeSampler in "fps" mode)                         │
│    - Picks highest candidate fps that fits within max_frames       │
│    - Decodes that many frames from the .mp4                        │
│    - Returns VideoFrames with target_fps and timestamps            │
│    → This determines WHAT THE MODEL SEES                           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│ 3. DataFormatter.format_video_object_track_points()                │
│    a. _filter_frames_to_video: align annotations → loaded frames   │
│    b. Resolve sampling_fps: explicit value, or None → target_fps   │
│    c. Cap sampling_fps at max_output_fps (default 2)               │
│    d. _sample_at_fps: subsample annotations to sampling_fps grid   │
│    e. Build prompt: "Track {label} at {fps} FPS"                   │
│    → This determines WHAT THE MODEL PREDICTS ON                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Dataset — Annotation and Overrides

### `_build_video_annotation` (per-dataset classmethod)

Each `LocalTrackingDataset` subclass builds annotation dicts with `frame_trajectories` containing point-tracks at every native frame. The annotation's `sampling_fps` is set to `VIDEO_FPS` (e.g. 24 for CFC) — this is just the annotation's native granularity, not the actual sampling target.

### `LocalTrackingDataset.get()` (~line 676)

```python
ex = dict(self.data[idx])              # shallow copy
ex['sampling_fps'] = self.sampling_fps  # override annotation value
message_list = self._create_message_list(ex)  # reads ex['sampling_fps']
```

The key line is `ex['sampling_fps'] = self.sampling_fps`. This ensures the message_list (which the formatter reads) gets:
- **Eval 1fps**: `sampling_fps=1`
- **Eval 8fps**: `sampling_fps=8`
- **Training**: `sampling_fps=None` (resolved later from loaded video)

### Sampler overrides

When `use_fps_sampling=True` (default), the dataset attaches per-example sampler overrides:

```python
metadata['sampler_overrides'] = {
    'frame_sample_mode': 'fps',
    'candidate_sampling_fps': self._get_candidate_fps(video_fps),
    'min_fps': self.sampling_fps or 1,
}
```

These override the base `TimeSampler` config (which defaults to `uniform_last_frame` mode) to use `fps` mode with dataset-specific candidates.

### `get_candidate_sampling_fps(video_fps, sampling_fps)` (~line 213)

Generates candidate fps values: multiples of `sampling_fps` that evenly divide `video_fps`, capped at `MAX_VIDEO_FPS=10`.

Examples:
- `get_candidate_sampling_fps(24, 1)` → `[1, 2, 3, 4, 6, 8]`
- `get_candidate_sampling_fps(24, 8)` → `[8]`
- `get_candidate_sampling_fps(18, 1)` → `[1, 2, 3, 6, 9]`
- `get_candidate_sampling_fps(6, 1)` → `[1, 2, 3, 6]`

**Note:** `min_fps` in the overrides is silently ignored by `TimeSampler` (it has no `min_fps` field). It only affects `FrameSampler`. The `_sampler_with_overrides` function filters overrides to valid sampler fields.

---

## Step 2: Video Loading — What the Model Sees

### Base sampler config (from model config)

```yaml
mm_preprocessor:
  max_frames: 384          # or 128 in some SFT configs
  frame_sample_mode: uniform_last_frame   # overridden to "fps" by tracking datasets
  candidate_sampling_fps: [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 16.0]
  max_fps: [2.0]           # only used by uniform_last_frame mode
  time_sampling: true      # → builds TimeSampler (not FrameSampler)
```

Because `time_sampling: true`, a `TimeSampler` is built. Tracking datasets override `frame_sample_mode` to `"fps"` via `sampler_overrides`.

### TimeSampler modes

#### `"uniform_last_frame"` (default, used by non-tracking tasks)

```python
if max_fps is not None:
    max_duration = (max_frames - 1) / max_fps
    if max_duration < duration:
        times = np.linspace(0, duration, num=max_frames, endpoint=True)
    else:
        times = np.arange(0.0, stop=duration, step=1/max_fps)
        times = np.concatenate([times, [duration]])
return None, times, None   # target_fps is always None
```

- `max_fps` controls the maximum temporal density. With `max_fps=[2.0]` and `max_frames=384`, videos up to `383/2 = 191.5s` get frames at 2fps. Longer videos get uniformly spaced frames.
- **`target_fps` is always `None`** in this mode.

#### `"fps"` (used by tracking datasets via sampler_overrides)

```python
sampling_fps = candidate_sampling_fps[0]   # start with lowest
for candidate_fps in candidate_sampling_fps[1:]:
    if max_frames / candidate_fps < duration:
        break
    sampling_fps = candidate_fps
times = np.arange(0, max_frames) / sampling_fps
times = times[times < duration]
return sampling_fps, times, None   # target_fps = selected fps
```

- Picks the **highest** candidate fps where the frame budget (`max_frames / fps`) still spans the full video duration.
- **`target_fps` is the selected fps** — this is what flows to the formatter.
- `max_fps` from the base config is **ignored** in this mode (it's only read in `uniform_last_frame`).

### How `max_frames` affects loaded fps

The fps selection depends on `max_frames / candidate_fps >= duration`. With different `max_frames`:

**CFC example (25s clip, candidates=[1,2,3,4,6,8]):**

| `max_frames` | Selected fps | Frames loaded | Reasoning |
|--------------|-------------|---------------|-----------|
| 384 | 8 | 200 | 384/8=48s ≥ 25s |
| 128 | 4 | 100 | 128/4=32s ≥ 25s, but 128/6=21.3s < 25s |
| 64 | 2 | 50 | 64/2=32s ≥ 25s, but 64/3=21.3s < 25s |
| 16 | 1 | 16 | budget too small for anything higher |

**Shorter CFC clip (8s, candidates=[1,2,3,4,6,8]):**

| `max_frames` | Selected fps | Frames loaded |
|--------------|-------------|---------------|
| 384 | 8 | 64 |
| 128 | 8 | 64 | 128/8=16s ≥ 8s |
| 64 | 8 | 64 | 64/8=8s ≥ 8s |
| 16 | 1 | 8 | budget too small |

So **shorter videos get higher fps** because the same frame budget covers them at denser sampling.

### `max_fps` vs `candidate_sampling_fps`

These control **different sampling modes** and do not interact:

| Parameter | Used by | Controls |
|-----------|---------|----------|
| `max_fps` | `uniform_last_frame` mode | Maximum temporal density (hard cap on fps) |
| `candidate_sampling_fps` | `fps` mode | Pool of fps values to pick from based on budget |

For tracking tasks, `sampler_overrides` switch to `fps` mode, so **`max_fps` from the model config has no effect**. The loaded fps is entirely determined by `candidate_sampling_fps` and `max_frames`.

---

## Step 3: Formatter — What the Model Predicts On

After loading, `ExamplePreprocessor` attaches the loaded `VideoFrames` object to the example (`multimodal_preprocessor.py:259`):
```python
example["video"] = video  # VideoFrames with .timestamps and .target_fps
```

Then the formatter's `_format_example` method sets up video info for the message (`data_formatter.py:1874`):
```python
if hasattr(video, "timestamps"):
    message["video"] = {"timestamps": video.timestamps, "target_fps": video.target_fps}
```

### `format_video_object_track_points` (~line 1419)

Three stages:

#### Stage A: `_filter_frames_to_video` (~line 1366)

Aligns raw annotation frames (at native VIDEO_FPS) to the actually-loaded timestamps. Keeps annotation frames within `eps=0.01s` of a loaded timestamp. This drops annotations for frames the model never saw.

Example: CFC has 599 annotation frames at 24fps. Video loaded 200 frames at 8fps. After filtering: 200 annotation frames remain (one per loaded frame).

#### Stage B: Resolve `sampling_fps` and apply output cap (~line 1432)

```python
sampling_fps = example["sampling_fps"]
if sampling_fps is None:
    video_info = example.get("video", {})
    sampling_fps = video_info.get("target_fps")

# Cap output fps (DataFormatter.max_output_fps, default=2)
if self.max_output_fps is not None and sampling_fps is not None and sampling_fps > self.max_output_fps:
    sampling_fps = self.max_output_fps
```

Resolution chain:
- **Eval**: explicit value (1 or 8) from dataset → used directly
- **Training**: `None` from dataset → resolves to `target_fps` from loaded video (e.g. 4 or 8, depends on `max_frames` and duration)

Then the **`max_output_fps` cap** is applied. The model was trained on ≤2fps output (Mevis 1-2fps, Molmo2VideoTrack 1-2fps), so `DataFormatter.max_output_fps` defaults to 2. This caps the resolved `sampling_fps` — the model still *sees* all loaded frames, but only *predicts* at the capped rate. A warning is logged when capping occurs.

To override for high-fps eval, set `max_output_fps` in the config or via CLI:
```bash
# eval.py supports dotlist overrides
torchrun ... launch_scripts/eval.py Molmo2-8B --task cfc_track_eval_8fps \
  --model.data_formatter.max_output_fps=8
```

#### Stage C: `_sample_at_fps` (~line 1396)

Generates a timestamp grid at `1/sampling_fps` intervals, then keeps only annotation frames near grid points.

```python
sampling_interval = 1.0 / sampling_fps
first_grid_point = ceil(start_time / sampling_interval) * sampling_interval
target_times = arange(first_grid_point, end_time + 1e-6, sampling_interval)
return _filter_frames_to_video(frames_data, target_times)
```

When `sampling_fps == target_fps` (loaded fps), this is a **natural no-op** — the grid matches the loaded frames exactly.

### Prompt construction (~line 1506)

```python
if sampling_fps and sampling_fps > 0:
    prompt_keywords["fps"] = str(int(sampling_fps))
```

- If `fps` is present: uses template like `"Track {label} at {fps} FPS"`
- If `fps` is absent (sampling_fps is None/0) or fps=="2" (50% of time): uses template like `"Track {label}."`

---

## Train vs Eval Summary

### Eval (`cfc_track_eval_1fps`, `cfc_track_eval_2fps`, `cfc_track_eval_8fps`)

| | 1fps eval | 2fps eval | 8fps eval (cap overridden) |
|-|-----------|-----------|---------------------------|
| `self.sampling_fps` | 1 | 2 | 8 |
| `candidate_sampling_fps` | [1,2,3,4,6,8] | [1,2,3,4,6,8] | [8] |
| Loaded fps (25s clip, max_frames=384) | 8 | 8 | 8 |
| Frames model sees | ~200 | ~200 | ~200 |
| After `max_output_fps` cap | 1 (under cap) | 2 (at cap) | 8 (requires override) |
| Annotation frames in output | ~25 | ~50 | ~200 |
| Prompt says | "at 1 FPS" | "at 2 FPS" | "at 8 FPS" |

**Key insight for eval:** The model sees all loaded frames (visual input), but only predicts points at the `sampling_fps` grid. For 1fps eval, the model sees 200 frames but outputs predictions at 25 timestamps. The 8fps eval requires `--model.data_formatter.max_output_fps=8` to bypass the default 2fps cap.

### Training (`cfc_track`)

| | max_frames=384 (25s clip) | max_frames=128 (25s clip) | max_frames=128 (8s clip) |
|-|---------------------------|---------------------------|--------------------------|
| `self.sampling_fps` | None | None | None |
| `candidate_sampling_fps` | [1,2,3,4,6,8] | [1,2,3,4,6,8] | [1,2,3,4,6,8] |
| Loaded fps (`target_fps`) | 8 | 4 | 8 |
| Frames model sees | ~200 | ~100 | ~64 |
| `sampling_fps` resolves to | 8 → **2** (capped) | 4 → **2** (capped) | 8 → **2** (capped) |
| Annotation frames in output | ~50 | ~50 | ~16 |
| Prompt says | "at 2 FPS" | "at 2 FPS" | "at 2 FPS" |

**Key insight for training:** `sampling_fps=None` resolves to `target_fps`, then gets capped by `max_output_fps` (default 2). The model sees all loaded frames but only predicts at ≤2fps — matching what it was pretrained on. To train at higher output fps, set `model_cfg.data_formatter.max_output_fps` in `sft.py`.

---

## TrackingDataset (Mevis, etc.) — Comparison

`TrackingDataset` (the HF-based parent class) works differently from `LocalTrackingDataset`:

- Per-example `sampling_fps` comes from the HF dataset (not hardcoded to VIDEO_FPS)
- `load()` filters the HF dataset to only examples where `sampling_fps == self.sampling_fps`
- `get()` passes `ex['sampling_fps']` directly (no override logic needed)
- No `None` case — `sampling_fps` is always an explicit integer from the HF data

This is correct because the HF dataset already has per-example `sampling_fps` values that match different annotation granularities.

---

## Per-Dataset Reference

| Dataset | `VIDEO_FPS` | `candidates(fps=1)` | `candidates(fps=8)` | Max loaded fps (384 frames, 25s) |
|---------|-------------|---------------------|---------------------|----------------------------------|
| CFC | 24 | [1,2,3,4,6,8] | [8] | 8 |
| PanAf | 18 | [1,2,3,6,9] | — | 9 |
| Mevis | 6 | [1,2,3,6] | — | 6 |

---

## Debugging

Use `scripts/debug_frame_sampling.py` to trace the pipeline:

```bash
# See what frames are loaded and how annotations are filtered
MOLMO_DATA_DIR=data python scripts/debug_frame_sampling.py cfc_track_eval_1fps --split validation
MOLMO_DATA_DIR=data python scripts/debug_frame_sampling.py cfc_track_eval_8fps --split validation
MOLMO_DATA_DIR=data python scripts/debug_frame_sampling.py cfc_track --split train

# Use a specific max_frames to simulate SFT config
MOLMO_DATA_DIR=data python scripts/debug_frame_sampling.py cfc_track --split train --max_frames 128
```

Output shows: raw annotation count, loaded frame count + target_fps, post-filter count, post-sample count, resolved sampling_fps, and per-object trajectory summaries.
