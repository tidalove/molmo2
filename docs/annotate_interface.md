# Interactive Track-Correction Annotation Interface

Web UI for annotating CFC fish-tracking videos by iteratively refining tracks —
manually (point-and-click) or by prompting the trained Molmo2 correction model
(vLLM). Every refinement is a step in a **branching tree** per video; you can
click back to any earlier step, branch a new prompt off it, and delete dead
branches.

## Quick start

```bash
# on an A100 node, repo root, molmo2 env
MOLMO_DATA_DIR=data python viewer/annotate_app.py            # port 6006, loads model in background
# or GPU-free (manual edits, browsing, export still work):
MOLMO_DATA_DIR=data python viewer/annotate_app.py --no-model

# from your laptop:
ssh -L 6006:localhost:6006 <a100-node>   # then open http://localhost:6006
```

The model (`--model_dir`, default `runs/cfc_all_real_llm_connector_vit/step300-hf`)
loads in a background thread (~2.5 min); the UI is usable immediately. The
sidebar pill shows `loading… / ready / error` — a CUDA OOM shows up there as an
error instead of crashing the app; reallocate with more memory and restart
(sessions survive restarts).

## CLI

| Flag | Default | Meaning |
|---|---|---|
| `--data_dir` | `data/video_datasets/video_track/CFC` | needs `videos/`, `JPEGImages/`, `annotations/` |
| `--session_dir` | `viewer/annotate_sessions` | per-video tree JSONs + `exports/` |
| `--model_dir` | `runs/cfc_all_real_llm_connector_vit/step300-hf` | HF checkpoint for vLLM |
| `--no-model` | off | never touch vLLM/GPU |
| `--export_box_size` | `20` | bbox side (px) written around each point on export |
| `--port` / `--host` | `6006` / `0.0.0.0` | |

## Workflow

1. **Load videos** (setup panel): tick videos from the searchable list (backed by
   `data_dir/videos/*.mp4`; if the dir is missing the list is empty and you paste
   exact names instead), pick a predictions source, **Confirm & Load**. Frames and
   metadata are only read at this point.
2. **Source formats** (auto-detected):
   - `predictions.json` (vLLM eval output) — reconstructed as a 2-node chain:
     root = step0 tracks parsed from the stored `input` chat, child = the model's
     `prediction`. So you build on top of the existing correction step.
   - COCO json (GT or tracker preds) — single root node, points = bbox centroids.
   - trajectory-annotation jsonl (`caption_annotations/*.jsonl`) — the
     `correction_steps` become a linear chain; a dropdown appears if a video has
     multiple trajectories.
3. **Iterate** (viewer):
   - The **tree** (sidebar) shows one box per step with its prompt. Click = select
     that step (canvas shows its tracks; save persists the selection). Hover → ✕
     deletes the step **and everything after it** (confirm dialog). Root can't be
     deleted.
   - **Manual edit**: `✏️ Edit points` → Add / Move / Delete modes (keys `a m d`),
     `＋ New track` allocates a fresh id, drag to move. `Save as new step` creates
     a `manual` node with prompt `"User added x points, deleted y points, moved z
     points"` — no model call.
   - **Model prompt**: type into the box, `Run model`. The request = multi-turn
     chat where the **selected** step's tracks are the previous assistant turn and
     your text is the next user turn (single previous step only — the context
     window can't fit more; steps are never chained). Job runs in the background
     (~2 min on an A100); a ⏳ ghost node shows in the tree; the UI stays usable.
     "frame N" phrases are rewritten to seconds like the training data.
   - Track chips: click = active track (edit target), double-click = hide/show.
     Slider/←→ scrub frames; "Only frames with points" filters the slider.
   - **Zoom/pan**: scroll wheel zooms toward the cursor (1×–24×); pan with
     middle-drag, Space+drag, or plain drag when not in edit mode. Keys:
     `+`/`-` zoom at center, `0` (or the `1×` button) resets. The view persists
     across frame scrubs and tree clicks; point editing works at any zoom —
     markers and hit radii stay constant on screen.
4. **Export**: with a leaf selected, `Export path → jsonl` appends the root→leaf
   chain as one standard trajectory-annotation jsonl line
   (`{session_dir}/exports/{video}_export.jsonl`), compatible with
   `CFCMultiTurn`-style loaders: bboxes are fixed-size boxes centered on each
   point so centroids round-trip exactly; images are rebuilt with
   `id = frame_idx + 1`.

## Persistence / crash recovery

Every mutation atomically rewrites `{session_dir}/{video}.json` (tmp + rename).
On restart, sessions are listed in the sidebar and `/api/load` resumes them
(tick "Reinitialize" to rebuild from the source instead). Running model jobs do
NOT survive a restart — the resulting node is only written on success.

## Architecture

```
viewer/annotate_app.py   FastAPI endpoints + State singleton (this file = API docs)
viewer/annotate.html     single vanilla-JS page (canvas editor, tree, prompt box)
viewer/track_io.py       pure data layer: source loaders, session schema,
                         conversions, export (coordinate conventions documented
                         in its module docstring; NO vllm imports)
viewer/inference.py      ModelRunner: vLLM load + one-correction-step inference;
                         reuses scripts/run_vllm.py's build_multi_turn_chat and
                         DataFormatter config
```

Canonical coordinates everywhere in the app: **pixels + native-fps frame
indices**. The model's 0-1000 scale / seconds exist only at the model boundary
(`track_io.parse_tracks_text`, `inference.build_correction_chat`).

The prompt path is regression-tested against the eval pipeline: rebuilding the
chat for an eel example reproduces the `input` string stored in
`runs/.../cfc_correction_eel_even_eval_2fps/eel/predictions.json` byte-for-byte.
Gotcha if you touch it: the final turn's `points` must be non-empty, otherwise
`DataFormatter.format_video_object_track_points` ignores the `question` and
substitutes a template prompt (data_formatter.py:1446).

## Frames

Displayed frames come from pre-extracted JPEGs (`JPEGImages/{video}/{video}_{i}.jpg`)
— no video decoding in the app. Model inference reads the actual mp4
(`videos/{video}.mp4`) through vLLM's molmo2 video-loader backend, exactly like
the eval pipeline (max_fps=2, sampling_fps from the session).
