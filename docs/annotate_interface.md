# Interactive Track-Correction Annotation Interface

Web UI for annotating CFC fish-tracking videos by iteratively refining tracks —
manually (point-and-click) or by prompting the trained Molmo2 correction model
(vLLM). Every refinement is a step in a **branching tree** per video; you can
click back to any earlier step, branch a new prompt off it, and delete dead
branches.

## Quick start

```bash
# on an A100 node, repo root, molmo2 env
MOLMO_DATA_DIR=data python viewer/annotate_app.py            # port 6006; GPU untouched until you load a model

# from your laptop:
ssh -L 6006:localhost:6006 <a100-node>   # then open http://localhost:6006
```

The server starts GPU-free. The **model is picked in the UI** (setup step 1,
exact path or filesystem browse; `--model_dir` only prefills the input) and
loads in a background thread (~2.5 min) once you press **Load model** — keep
picking predictions/videos meanwhile. The sidebar pill shows
`not loaded / loading… / ready / error` — a CUDA OOM shows up there as an error
instead of crashing the app; reallocate with more memory and restart (sessions
survive restarts). **One checkpoint per process**: switching checkpoints
requires a server restart (vLLM can't reliably free GPU memory in-process).

**Comparing two checkpoints on the same videos**: run two server instances
(second one with `--port 6007`, tunnel both ports) sharing the default
`--session_dir`, load a different checkpoint in each, and give each a distinct
**session tag** (setup step 4 — prefilled from the checkpoint name). Each tag
gets its own session tree per video, listed side by side in either sidebar, so
you can toggle between the two models' trees on the same video. Don't open the
*same* tagged session for editing in both instances at once — each keeps an
in-memory copy and the last save wins.

## CLI

| Flag | Default | Meaning |
|---|---|---|
| `--data_dir` | `data/video_datasets/video_track/CFC` | needs `videos/`, `JPEGImages/`, `annotations/` |
| `--session_dir` | `viewer/annotate_sessions` | per-session tree JSONs + `exports/` |
| `--model_dir` | `runs/cfc_all_real_llm_connector_vit/step300-hf` | default checkpoint path prefilled in the UI (nothing loads at startup) |
| `--export_box_size` | `20` | bbox side (px) written around each point on export |
| `--port` / `--host` | `6006` / `0.0.0.0` | |

## Workflow

1. **Setup panel**, in order:
   1. **Model** — checkpoint path (prefilled from `--model_dir`) or Browse
      (pick a *directory*; it must contain `config.json`), then **Load model**.
      Loading runs in the background while you continue.
   2. **Predictions (optional)** — pick a source file; after **Inspect**, the
      video list below is *filtered to the videos present in that file*. Or tick
      **Start without predictions**: each video gets a single empty root node and
      the first tracks come from the **▶ Track all fish** button (initial-tracking
      inference) or from manual points.
   3. **Videos** — tick from the searchable list (backed by `data_dir/videos/*.mp4`;
      if the dir is missing the list is empty and you paste exact names instead).
   4. **Confirm & Load** — the optional **session tag** (prefilled from the
      loaded checkpoint) lets the same video carry several parallel session
      trees; sessions are keyed by *(video, tag)*, so resuming matches both.
      Clear the tag for the plain untagged session (legacy session files are
      untagged). Frames and metadata are only read at this point.
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
     `Run model` is disabled while the selected step has **no tracks** — an empty
     context would make a degenerate correction prompt; use **▶ Track all fish**
     instead, which runs single-turn initial-tracking inference (same prompt path
     as the `cfc_track_eval` pipeline) and adds a `model` node under the empty step.
   - Track chips: click = active track (edit target), double-click = hide/show.
     Slider/←→ scrub frames; "Only frames with points" filters the slider.
   - **Zoom/pan**: scroll wheel zooms toward the cursor (1×–24×); pan with
     middle-drag, Space+drag, or plain drag when not in edit mode. Keys:
     `+`/`-` zoom at center, `0` (or the `1×` button) resets. The view persists
     across frame scrubs and tree clicks; point editing works at any zoom —
     markers and hit radii stay constant on screen.
4. **Export**: with a leaf selected, `Export path → jsonl` appends the root→leaf
   chain as one standard trajectory-annotation jsonl line
   (`{session_dir}/exports/{video}__{tag}_export.jsonl`), compatible with
   `CFCMultiTurn`-style loaders: bboxes are fixed-size boxes centered on each
   point so centroids round-trip exactly; images are rebuilt with
   `id = frame_idx + 1`.

## Persistence / crash recovery

Every mutation atomically rewrites `{session_dir}/{video}__{tag}.json`
(`{video}.json` when untagged; tmp + rename).
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
viewer/inference.py      ModelRunner: vLLM load + inference (run_correction /
                         run_tracking); reuses scripts/run_vllm.py's
                         build_multi_turn_chat and DataFormatter config
```

Canonical coordinates everywhere in the app: **pixels + native-fps frame
indices**. The model's 0-1000 scale / seconds exist only at the model boundary
(`track_io.parse_tracks_text`, `inference.build_correction_chat`).

The prompt path is regression-tested against the eval pipeline: rebuilding the
chat for an eel example reproduces the `input` string stored in
`runs/.../cfc_correction_eel_even_eval_2fps/eel/predictions.json` byte-for-byte.
Gotcha if you touch it: with empty `points`,
`DataFormatter.format_video_object_track_points` ignores the `question` and
substitutes a template prompt (data_formatter.py:1446). In the *correction*
chat the final turn therefore reuses the parent tracks as its `points`; in the
*initial-tracking* chat (`build_track_chat`) `points=[]` is deliberate — the
substituted template is exactly the `cfc_track_eval` prompt (verified
byte-for-byte against the `input` field of
`runs/.../cfc_track_eval_2fps/validation-v2/predictions.json`).

## Frames

Displayed frames come from pre-extracted JPEGs (`JPEGImages/{video}/{video}_{i}.jpg`)
— no video decoding in the app. Model inference reads the actual mp4
(`videos/{video}.mp4`) through vLLM's molmo2 video-loader backend, exactly like
the eval pipeline (max_fps=2, sampling_fps from the session).
