# Frame Sampling Pipeline for Video Tracking Eval

How video frames get sampled, loaded, and selected when running tracking evaluation tasks like `cfc_track_eval_1fps` or `mevis_track_eval_1fps`.

## TL;DR

1. **Encode** — Offline: raw frames are converted to an `.mp4` at the dataset's native `VIDEO_FPS` (`encode_frames_to_video`, `academic_video_track_datasets.py:118-178`).
2. **Sample** — At load time: a `TimeSampler` (configured via `sampler_overrides` from the dataset) decides which timestamps to decode from the `.mp4` (`video_loader.py:126-182`).
3. **Filter** — In the formatter: annotation point-tracks are aligned to the actually-loaded timestamps (`_filter_frames_to_video`), then downsampled to the target `sampling_fps` (`_sample_at_fps`) (`data_formatter.py:1366-1417`).
4. **Predict** — The model sees one image-patch block per loaded frame, prefixed with a compact timestamp (e.g. `"0.0"`, `"0.5"`), and outputs point predictions only for the frames that survived filtering (`video_preprocessor.py:183-205`).

---

## Step 1: Video Encoding (offline)

Each dataset converts raw image frames to H.264 `.mp4` files at the dataset's `VIDEO_FPS`.

**`encode_frames_to_video`** (`academic_video_track_datasets.py:118-178`):
- Accepts `frames_dir`, `output_path`, `fps`, and optional `native_fps` for subsampling.
- If `native_fps != fps`, keeps every `round(native_fps / fps)`-th frame before encoding.
- Writes an ffmpeg concat-demuxer filelist and encodes with `libx264` at the target `fps`.

**Per-dataset VIDEO_FPS values** (set as class constants):

| Dataset | `VIDEO_FPS` | Source |
|---------|-------------|--------|
| Mevis   | 6           | `academic_video_track_datasets.py:1307` |
| CFC     | 24          | `academic_video_track_datasets.py:1044` |
| PanAf   | 18          | `academic_video_track_datasets.py:728` |

The encoded `.mp4`'s internal fps matches `VIDEO_FPS`. This is the fps that the video decoder reports at load time.

---

## Step 2: Frame Sampling at Load Time

When `ExamplePreprocessor.__call__` processes an example with a `"video"` key, it calls `video_preprocessor.load_video()` which delegates to `VideoLoader.__call__` (`video_loader.py:304-365`).

### Base sampler config (from `Molmo2-8B/config.yaml:148-168`)

```yaml
mm_preprocessor:
  max_frames: 384
  frame_sample_mode: uniform_last_frame
  candidate_sampling_fps: [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 16.0]
  max_fps: [2.0]
  time_sampling: true       # → builds a TimeSampler (not FrameSampler)
  loading_method: torchcodec_exact
```

Because `time_sampling: true`, a `TimeSampler` is built (`video_loader.py:291-292`).

### Sampler overrides

Tracking datasets override the sampler on a per-example basis via `metadata['sampler_overrides']`. These overrides are applied in `_sampler_with_overrides` (`video_loader.py:106-123`), which uses `dataclasses.replace` to swap only fields that exist on the sampler.

Both `TrackingDataset.get()` (`academic_video_track_datasets.py:529-534`) and `LocalTrackingDataset.get()` (`academic_video_track_datasets.py:691-697`) set:

```python
metadata['sampler_overrides'] = {
    'frame_sample_mode': 'fps',              # override: switch from uniform_last_frame → fps
    'candidate_sampling_fps': [...],         # override: dataset-specific candidates
    'min_fps': ex['sampling_fps'],           # NOTE: ignored for TimeSampler (field doesn't exist)
}
```

The `candidate_sampling_fps` list comes from `get_candidate_sampling_fps(video_fps, self.sampling_fps or 1)` (`academic_video_track_datasets.py:213-225`), which generates multiples of `sampling_fps` that evenly divide `video_fps`, up to `MAX_VIDEO_FPS` (10).

**Important:** The `min_fps` override is silently dropped for `TimeSampler` because `TimeSampler` has no `min_fps` field. It only exists on `FrameSampler` (`video_loader.py:192`). The `_sampler_with_overrides` function filters to `valid_fields` (`video_loader.py:109-113`).

### TimeSampler in `fps` mode (`video_loader.py:171-180`)

```python
elif self.frame_sample_mode == "fps":
    sampling_fps = self.candidate_sampling_fps[0]
    for candidate_fps in self.candidate_sampling_fps[1:]:
        if max_frames / candidate_fps < duration:
            break
        sampling_fps = candidate_fps
    times = np.arange(0, max_frames) / sampling_fps
    times = times[times < duration]
    return sampling_fps, times, None
```

It picks the **highest** candidate fps whose frame budget (`max_frames / fps`) still spans the full video duration. This maximizes temporal density while covering the whole clip.

### Worked example: CFC (`cfc_track_eval_1fps`)

- `VIDEO_FPS = 24`, `self.sampling_fps = 1`
- `get_candidate_sampling_fps(24, 1)` → `[1, 2, 3, 4, 6, 8]` (multiples of 1 dividing 24, capped at 10)
- Typical CFC clip: 128 frames at 24fps = **5.33s** duration
- TimeSampler iterates candidates:
  - fps=1: budget=384s > 5.33s ✓ → sampling_fps=1
  - fps=2: budget=192s > 5.33s ✓ → sampling_fps=2
  - ...
  - fps=8: budget=48s > 5.33s ✓ → **sampling_fps=8**
- Result: `times = [0, 0.125, 0.25, ..., 5.25]` → **~42 frames loaded**

### Worked example: Mevis (`mevis_track_eval_1fps`)

- `VIDEO_FPS = 6`, `self.sampling_fps = 1`
- `get_candidate_sampling_fps(6, 1)` → `[1, 2, 3, 6]`
- Typical Mevis clip: 120 frames at 6fps = **20s** duration
- TimeSampler iterates:
  - fps=1: budget=384s > 20s ✓ → sampling_fps=1
  - fps=2: budget=192s > 20s ✓ → sampling_fps=2
  - ...
  - fps=6: budget=64s > 20s ✓ → **sampling_fps=6**
- Result: `times = [0, 0.167, 0.333, ..., 19.83]` → **~120 frames loaded**

---

## Step 3: Annotation Filtering in the Formatter

After the video is loaded, the formatter (`data_formatter.py`) processes the example's `message_list`. For tracking tasks, `format_video_object_track_points` (`data_formatter.py:1419-1515`) runs two filtering stages:

### Stage A: `_filter_frames_to_video` (`data_formatter.py:1366-1394`)

Aligns annotation frames to the **actually loaded** video timestamps. For each annotation frame, finds the closest loaded timestamp within `eps=1e-2` seconds. Drops any annotation frames that don't match a loaded frame.

This is necessary because the annotations may describe frames at the native video fps (e.g. every frame at 24fps), but the video loader only sampled a subset.

### Stage B: `_sample_at_fps` (`data_formatter.py:1396-1417`)

Generates a 1/`sampling_fps`-spaced timestamp grid, then calls `_filter_frames_to_video` to keep only the annotation frames closest to grid points.

```python
sampling_interval = 1.0 / sampling_fps
first_grid_point = np.ceil(start_time / sampling_interval) * sampling_interval
target_times = np.arange(first_grid_point, end_time + 1e-6, sampling_interval)
return self._filter_frames_to_video(frames_data, target_times)
```

### The critical field: `sampling_fps` in the message

The `sampling_fps` value used in `_sample_at_fps` comes from the **message_list** (`data_formatter.py:1427`):

```python
sampling_fps = example["sampling_fps"]
```

This is set per-example by `_create_message_list` (`academic_video_track_datasets.py:480-483`):

```python
message_list = [{
    "sampling_fps": ex['sampling_fps'],  # from the annotation data
    ...
}]
```

For **Mevis**, `ex['sampling_fps']` comes from the HF dataset and is already filtered to match `self.sampling_fps` during `load()` (`academic_video_track_datasets.py:444-447`). So for `mevis_track_eval_1fps`, `ex['sampling_fps'] = 1`.

For **CFC**, `ex['sampling_fps']` is hardcoded to `VIDEO_FPS = 24` in `_build_video_annotation` (`academic_video_track_datasets.py:1155`). CFC's overridden `load()` (`academic_video_track_datasets.py:1057-1082`) does not filter by `self.sampling_fps`. So for `cfc_track_eval_1fps`, `ex['sampling_fps'] = 24` — regardless of the `_1fps` suffix.

---

## Step 4: What the Model Sees

### Timestamp prefixes

`TokenIndexingVideoPreprocessor.__call__` (`video_preprocessor.py:183-205`) prepends each frame's patch tokens with a text timestamp. With the default `time_mode = "per-frame-compact"`:

```python
prefix = f"{frame_time:.1f}"   # e.g. "0.0", "0.5", "1.0"
```

### Prompt text

The prompt template includes the target fps (from the message's `sampling_fps`) and the object label. For `video_point_track_per_frame` style:

```
Track {label} at {fps} fps.    # e.g. "Track fish at 24 fps."
```

The model's output is expected to contain point coordinates at timestamps matching the `sampling_fps` grid.

### Prediction frames

The model outputs predictions for the frames that survived both `_filter_frames_to_video` and `_sample_at_fps`. In correct operation (Mevis), these are the 1fps-grid frames. In the CFC bug case, these are all loaded frames (~8fps), because `_sample_at_fps(data, 24)` is a no-op.

---

## Summary Comparison Table

| Aspect | Mevis `_track_eval_1fps` | CFC `_track_eval_1fps` |
|--------|--------------------------|------------------------|
| `VIDEO_FPS` | 6 | 24 |
| `self.sampling_fps` | 1 | 1 |
| `ex['sampling_fps']` (annotation) | 1 (from HF, filtered) | **24** (hardcoded) |
| `candidate_sampling_fps` override | `[1, 2, 3, 6]` | `[1, 2, 3, 4, 6, 8]` |
| Video loader actual fps | up to 6fps | up to 8fps |
| `_sample_at_fps` target | 1fps grid | **24fps grid (no-op)** |
| Prompt says | "at 1 fps" | **"at 24 fps"** |
| Model predicts at | 1fps | **whatever was loaded (~8fps)** |
| Correct 1fps behavior? | Yes | **No** |

---

## Bug: CFC `_1fps` Doesn't Actually Filter to 1fps

### Symptom

Running `cfc_track_eval_1fps` produces predictions at the video loader's sampling rate (~8fps for short clips), not at 1fps. The prompt also incorrectly says "at 24 fps".

### Root cause

CFC's `_build_video_annotation` hardcodes `"sampling_fps": cls.VIDEO_FPS` (24) for every example (`academic_video_track_datasets.py:1155`). Unlike `TrackingDataset.load()`, CFC's overridden `load()` method (`academic_video_track_datasets.py:1057-1082`) does **not** call `super().load()` and therefore skips the filtering at lines 444-447 that would reject examples where `sampling_fps != self.sampling_fps`.

The result is that `self.sampling_fps = 1` (from the constructor) only affects the `candidate_sampling_fps` override for the video loader, but **never propagates** to the per-example `sampling_fps` field that the formatter uses for `_sample_at_fps`.

### Flow comparison

**Mevis (correct):**
```
self.sampling_fps=1
  → load() filters HF dataset to ex['sampling_fps']==1
  → message_list gets sampling_fps=1
  → formatter._sample_at_fps(data, 1) → 1fps grid ✓
```

**CFC (broken):**
```
self.sampling_fps=1
  → load() overridden, no filtering, ex['sampling_fps']=24
  → message_list gets sampling_fps=24
  → formatter._sample_at_fps(data, 24) → no-op ✗
```

### Suggested fix

In `LocalTrackingDataset.get()` (or `CFC.get()`), override the per-example `sampling_fps` when `self.sampling_fps` is set:

```python
# In LocalTrackingDataset.get() or CFC.get(), after building message_list:
effective_sampling_fps = self.sampling_fps if self.sampling_fps is not None else ex['sampling_fps']

# Use effective_sampling_fps instead of ex['sampling_fps'] in:
# 1. message_list[0]['sampling_fps']
# 2. The returned dict's 'sampling_fps' and 'fps' fields
```

This would make `_sample_at_fps` receive the correct target fps (1), and the prompt would correctly say "at 1 fps".
