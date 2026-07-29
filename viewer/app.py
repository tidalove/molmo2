"""FastAPI viewer for Molmo2 prediction JSONs (or GT-only).

Lists predictions grouped by video. Click a prompt to view its predicted
points overlaid on the video frames at the original frame rate.

Two modes:
  --predictions: load model predictions.json
  --annotations: GT-only mode — synthesize prompts from a COCO split json plus
                 caption_annotations/queries.json; predictions are absent.

Usage:
    python viewer/app.py \
        --predictions runs/cfc_targeted_easy/results/val-easy/predictions.json \
        --data_dir data/video_datasets/video_track/CFC \
        --port 8000

    python viewer/app.py \
        --annotations data/video_datasets/video_track/CFC/annotations/kenai-val-easy.json \
        --data_dir data/video_datasets/video_track/CFC \
        --port 8000
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _render  # noqa: E402


VIEWER_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "viewer.html")

app = FastAPI(title="Molmo2 Predictions Viewer")


CFC_DEFAULT_FPS = 10
CFC_DEFAULT_EXPRESSION = "fish"  # qid=0 "track all fish" prompt


class State:
    predictions_path: str | None = None
    compare_path: str | None = None
    annotations_path: str | None = None
    gt_coco_path: str | None = None
    captions_path: str | None = None
    queries_path: str | None = None
    data_dir: str | None = None
    gt_only: bool = False
    text_mode: bool = False
    _preds: list | None = None
    _by_video: dict | None = None
    _by_id: dict | None = None
    _frames_by_video: dict | None = None
    _masks_cache: dict = {}  # example_id -> {obj_idx: [rle_or_None per frame]}
    _gt_by_video: dict | None = None  # {video: {frame_idx: [[x,y,w,h], ...]}}
    _captions: dict | None = None  # {video_id: caption_str}


_INITIAL_RE = re.compile(
    r"<\|im_start\|>assistant\s*\n(.*?)<\|im_end\|>", re.DOTALL)
_USER_RE = re.compile(
    r"<\|im_start\|>user\s*\n(.*?)<\|im_end\|>", re.DOTALL)
_SPECIAL_TOK_RE = re.compile(r"<\|[^|]*\|>")


def _strip_special_tokens(s) -> str:
    if not isinstance(s, str):
        return ""
    return _SPECIAL_TOK_RE.sub("", s).strip()


def _extract_initial_pred(input_str: str) -> str | None:
    m = _INITIAL_RE.search(input_str or "")
    return _strip_special_tokens(m.group(1)) if m else None


def _extract_correction_text(input_str: str) -> str | None:
    matches = _USER_RE.findall(input_str or "")
    if len(matches) < 2:
        return None
    return _strip_special_tokens(matches[-1])


def _load_gt_bboxes() -> dict:
    """Build {video: {frame_idx: [[x,y,w,h], ...]}} from a COCO json."""
    if not State.gt_coco_path:
        return {}
    with open(State.gt_coco_path) as f:
        coco = json.load(f)
    image_id_to_video_frame = {}
    for img in coco.get("images", []):
        base = img["file_name"]
        if base.endswith(".jpg"):
            base = base[:-4]
        try:
            video, frame_idx_str = base.rsplit("_", 1)
            frame_idx = int(frame_idx_str)
        except (ValueError, IndexError):
            continue
        image_id_to_video_frame[img["id"]] = (video, frame_idx)
    out: dict = defaultdict(lambda: defaultdict(list))
    for ann in coco.get("annotations", []):
        vf = image_id_to_video_frame.get(ann["image_id"])
        if vf is None:
            continue
        video, frame_idx = vf
        out[video][frame_idx].append(ann["bbox"])
    return {v: dict(frames) for v, frames in out.items()}


def _build_pseudo_preds_from_annotations() -> list:
    """Synthesize a predictions-shaped list from COCO annotations + queries.json."""
    with open(State.annotations_path) as f:
        coco = json.load(f)
    if State.queries_path:
        queries_path = Path(State.queries_path)
    else:
        queries_path = None
    queries = {}
    if queries_path and queries_path.exists():
        with open(queries_path) as f:
            queries = json.load(f)
    else:
        print(f"WARN: {queries_path} is None or not found; only qid=0 prompts will be shown")

    images_by_video: dict[str, list] = {}
    for img in coco["images"]:
        vid = img["file_name"][:img["file_name"].rfind("_")]
        images_by_video.setdefault(vid, []).append(img)

    preds = []
    for video_id, images in sorted(images_by_video.items()):
        first = images[0]
        w, h = first["width"], first["height"]
        prompts = [(0, CFC_DEFAULT_EXPRESSION)]
        for q in queries.get(video_id, []):
            prompts.append((q["qid"], q["expression"]))
        for qid, expression in prompts:
            preds.append({
                "example_id": f"{video_id}__{qid}",
                "video": video_id,
                "expression": expression,
                "prompt": "",
                "generated_text": "",
                "w": w,
                "h": h,
                "video_fps": CFC_DEFAULT_FPS,
                "prediction": [],
            })
    return preds


def _load():
    if State._preds is None:
        if State.gt_only:
            State._preds = _build_pseudo_preds_from_annotations()
        else:
            with open(State.predictions_path) as f:
                State._preds = json.load(f)

        # Auto-detect text-correction mode (CFCv1Text outputs)
        State.text_mode = (
            len(State._preds) > 0
            and "input" in State._preds[0]
            and "_traj" in State._preds[0].get("example_id", "")
        )
        if State.text_mode:
            for p in State._preds:
                if not p.get("video"):
                    ex_id = p["example_id"]
                    if "_traj" in ex_id:
                        p["video"] = ex_id.rsplit("_traj", 1)[0]
                    else:
                        p["video"] = ex_id
                p["initial_pred"] = _extract_initial_pred(p.get("input", ""))
                p["correction_text"] = _extract_correction_text(p.get("input", ""))

        # Optional comparison predictions (e.g. the no_info tier of the same eval): per example,
        # ndh_diff = norm_delta_hota(main) - norm_delta_hota(compare). Both values are precomputed
        # floats stored in the jsons (written natively at eval time) — no HOTA compute here.
        if State.compare_path:
            with open(State.compare_path) as f:
                cmp_ndh = {c["example_id"]: c.get("norm_delta_hota") for c in json.load(f)}
            for p in State._preds:
                cv = cmp_ndh.get(p["example_id"])
                pv = p.get("norm_delta_hota")
                p["compare_norm_delta_hota"] = cv
                p["ndh_diff"] = (pv - cv) if (pv is not None and cv is not None) else None

        by_video = defaultdict(list)
        by_id = {}
        for p in State._preds:
            by_video[p["video"]].append(p)
            by_id[p["example_id"]] = p
        State._by_video = dict(by_video)
        State._by_id = by_id
        State._frames_by_video = {}
        State._gt_by_video = _load_gt_bboxes()
        if State.captions_path:
            with open(State.captions_path) as f:
                raw = json.load(f)
            State._captions = {
                k: (v.get("caption", "") if isinstance(v, dict) else str(v))
                for k, v in raw.items()
            }
    return State._preds


def _list_video_frames(video: str) -> list[int]:
    if video in State._frames_by_video:
        return State._frames_by_video[video]
    frames = _render.list_video_frames(State.data_dir, video)
    State._frames_by_video[video] = frames
    return frames


def _unique_obj_ids(preds: list) -> set:
    """Union of obj_ids across all predictions for one video."""
    ids = set()
    for p in preds:
        pred = p["prediction"]
        if isinstance(pred, list):
            for obj_id, _t, _x, _y in pred:
                ids.add(str(obj_id))
        elif isinstance(pred, str):
            from olmo.preprocessing.point_formatter import extract_tracks
            tracks = extract_tracks(pred, p["w"], p["h"], p["video_fps"],
                                    format="video_point_track_per_frame")
            for tr in tracks:
                for pid in tr["points"]:
                    ids.add(str(pid))
    return ids


def _get_queries() -> dict:
    if hasattr(State, "_queries_cache"):
        return State._queries_cache
    path = State.queries_path or (
        Path(State.data_dir) / "caption_annotations" / "queries.json")
    path = Path(path)
    if not path.exists():
        State._queries_cache = {}
    else:
        with open(path) as f:
            State._queries_cache = json.load(f)
    return State._queries_cache


def _resolve_qid(p: dict) -> int | None:
    """Derive MasksRLE qid for a prediction.

    Priority: example_id suffix after `__`; CFC default "fish" → 0;
    expression match in queries.json; else fallback to 0.
    """
    ex_id = p["example_id"]
    if "__" in ex_id:
        try:
            return int(ex_id.rsplit("__", 1)[1])
        except ValueError:
            pass
    expression = (p.get("expression") or "").strip()
    if expression == CFC_DEFAULT_EXPRESSION:
        return 0
    for q in _get_queries().get(p.get("video", ""), []):
        if q.get("expression") == expression:
            return int(q.get("qid", 0))
    return 0


def _load_masks(p: dict):
    """Load MasksRLE/{video}/{qid}.json; return raw dict or None if missing.

    Correction-style examples (example_id contains `_traj`) have per-trajectory GT
    keyed by the full example_id (MasksRLE/{example_id}/0.json), not the raw video
    name. The raw-video mask only coincides for hardlinked families; for encoded GT
    (e.g. synthetic_incomplete, step-1) it is a different mask — so always key these
    by example_id.
    """
    ex_id = p["example_id"]
    if ex_id in State._masks_cache:
        return State._masks_cache[ex_id]
    if "_traj" in ex_id:
        State._masks_cache[ex_id] = _render.load_masks_for(State.data_dir, ex_id, 0)
        return State._masks_cache[ex_id]
    if "__" in ex_id:
        video, _ = ex_id.rsplit("__", 1)
    else:
        video = p.get("video", ex_id)
    qid = _resolve_qid(p)
    if qid is None:
        State._masks_cache[ex_id] = None
        return None
    State._masks_cache[ex_id] = _render.load_masks_for(State.data_dir, video, qid)
    return State._masks_cache[ex_id]


def _mask_frames_with_gt(masks: dict) -> list[int]:
    return _render.mask_frames_with_gt(masks)


def _render_mask_png(masks: dict, frame_idx: int, w: int, h: int):
    return _render.render_mask_png(masks, frame_idx, w, h)


def _parse_prediction(p: dict, which: str = "prediction") -> dict:
    """Return {frame_idx: [{obj_id, x, y}, ...]} for one prediction.

    `which` ∈ {"prediction", "initial_pred"}.
    """
    pred = p.get(which)
    if pred is None or pred == "":
        return {}
    video_fps = p["video_fps"]
    frames = defaultdict(list)
    if isinstance(pred, list):
        for obj_id, t, x, y in pred:
            frame = round(float(t) * video_fps)
            frames[frame].append({"obj_id": int(obj_id), "x": float(x), "y": float(y)})
    elif isinstance(pred, str):
        from olmo.preprocessing.point_formatter import extract_tracks
        tracks = extract_tracks(pred, p["w"], p["h"], video_fps,
                                format="video_point_track_per_frame")
        for tr in tracks:
            f = tr["frame"]
            for pid, pdata in tr["points"].items():
                frames[f].append({"obj_id": int(pid),
                                  "x": float(pdata["point"][0]),
                                  "y": float(pdata["point"][1])})
    return {int(k): v for k, v in frames.items()}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(VIEWER_HTML_PATH) as f:
        return f.read()


def _n_fish_from_gt(video: str) -> int:
    path = Path(State.data_dir) / "MasksRLE" / video / "0.json"
    if not path.exists():
        return 0
    with open(path) as f:
        return len(json.load(f))


@app.get("/api/predictions")
def list_predictions():
    _load()
    out = []
    for video in sorted(State._by_video):
        preds = State._by_video[video]
        items = []
        for p in preds:
            items.append({
                "example_id": p["example_id"],
                "expression": p.get("expression", ""),
                "correction_text": p.get("correction_text", "") or "",
                "norm_delta_hota": p.get("norm_delta_hota"),
                "ndh_diff": p.get("ndh_diff"),
            })
        n_fish = _n_fish_from_gt(video) if State.gt_only else len(_unique_obj_ids(preds))
        # Per-video normalized ΔHOTA: mean of per-trajectory values (excluding n/a, i.e. before==1
        # or unannotated). None -> sorts to the bottom in the viewer. Cheap arithmetic on stored
        # floats; HOTA itself is precomputed during the post-eval plot step, never here.
        ndh_vals = [p["norm_delta_hota"] for p in preds
                    if p.get("norm_delta_hota") is not None]
        diff_vals = [p["ndh_diff"] for p in preds if p.get("ndh_diff") is not None]
        out.append({
            "video": video,
            "items": items,
            "n_prompts": len(items),
            "n_fish": n_fish,
            "n_frames": len(_list_video_frames(video)),
            "norm_delta_hota": (sum(ndh_vals) / len(ndh_vals)) if ndh_vals else None,
            "ndh_diff": (sum(diff_vals) / len(diff_vals)) if diff_vals else None,
        })
    return out


@app.get("/api/meta")
def get_meta():
    _load()
    return {
        "gt_only": State.gt_only,
        "text_mode": State.text_mode,
        "has_captions": State._captions is not None,
        "has_hota": any("hota_before" in p for p in (State._preds or [])),
        "has_ndh_diff": any(p.get("ndh_diff") is not None for p in (State._preds or [])),
    }


@app.get("/api/predictions/{example_id}")
def get_prediction(example_id: str):
    _load()
    p = State._by_id.get(example_id)
    if p is None:
        raise HTTPException(404, f"example_id {example_id} not found")
    frames_with_preds = _parse_prediction(p, which="prediction")
    available_frames = _list_video_frames(p["video"])

    payload = {
        "example_id": example_id,
        "video": p["video"],
        "expression": p.get("expression", ""),
        "prompt": p.get("prompt", ""),
        "generated_text": p.get("generated_text", ""),
        "w": p["w"],
        "h": p["h"],
        "video_fps": p["video_fps"],
        "frames_with_preds": {str(k): v for k, v in frames_with_preds.items()},
        "available_frames": available_frames,
        "caption": (State._captions or {}).get(p["video"], ""),
        "hota_before": p.get("hota_before"),
        "hota_after": p.get("hota_after"),
        "norm_delta_hota": p.get("norm_delta_hota"),
        "compare_norm_delta_hota": p.get("compare_norm_delta_hota"),
        "ndh_diff": p.get("ndh_diff"),
    }

    masks = _load_masks(p)
    mask_frames = _mask_frames_with_gt(masks) if masks else []
    payload["mask_frames"] = mask_frames

    if State.text_mode:
        frames_with_initial = _parse_prediction(p, which="initial_pred")
        gt_frames = (State._gt_by_video or {}).get(p["video"], {})
        payload.update({
            "text_mode": True,
            "frames_with_initial_pred": {str(k): v for k, v in frames_with_initial.items()},
            "gt_bboxes_by_frame": {str(k): v for k, v in gt_frames.items()},
            "correction_text": p.get("correction_text", "") or "",
            "initial_text": _strip_special_tokens(p.get("initial_pred", "")),
            "prediction_text": _strip_special_tokens(p.get("prediction", "")),
        })
    else:
        payload["text_mode"] = False

    return payload


@app.get("/api/masks/{example_id}/{frame_idx}")
def get_mask(example_id: str, frame_idx: int):
    _load()
    p = State._by_id.get(example_id)
    if p is None:
        raise HTTPException(404, f"example_id {example_id} not found")
    masks = _load_masks(p)
    if not masks:
        raise HTTPException(404, "no MasksRLE for example")
    png = _render_mask_png(masks, frame_idx, p["w"], p["h"])
    if png is None:
        raise HTTPException(404, f"no mask at frame {frame_idx}")
    return Response(content=png, media_type="image/png")


@app.get("/api/frame/{video}/{frame_idx}")
def get_frame(video: str, frame_idx: int):
    p = Path(State.data_dir) / "JPEGImages" / video / f"{video}_{frame_idx}.jpg"
    if not p.exists():
        raise HTTPException(404, f"frame not found: {p}")
    return FileResponse(p, media_type="image/jpeg")


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--predictions", help="Path to predictions.json")
    src.add_argument("--annotations",
                     help="Path to a COCO split json (GT-only mode)")
    parser.add_argument("--compare_predictions", default=None,
                        help="Second predictions.json (e.g. the no_info tier of the same eval); "
                             "enables sorting by ΔΔHOTA = norm_delta_hota(main) - norm_delta_hota(compare), "
                             "matched by example_id")
    parser.add_argument("--queries", help="Path to a queries.json (GT-only mode)")
    parser.add_argument("--captions", default=None,
                        help="Path to a captions JSON (video_id -> {caption, rounds})")
    parser.add_argument("--data_dir", required=False,
                        help="Dataset root dir containing JPEGImages/ and MasksRLE/",
                        default="data/video_datasets/video_track/CFC")
    parser.add_argument("--gt_coco", default=None,
                        help="Path to a COCO json with GT bboxes (text-mode). "
                             "Optional; if provided, GT bboxes overlay on frames.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()
    State.predictions_path = args.predictions
    State.compare_path = args.compare_predictions
    State.annotations_path = args.annotations
    State.gt_coco_path = args.gt_coco
    State.data_dir = args.data_dir
    State.gt_only = args.predictions is None
    State.queries_path = args.queries
    State.captions_path = args.captions
    if State.gt_only:
        print(f"Annotations: {args.annotations} (GT-only mode)")
    else:
        print(f"Predictions: {args.predictions}")
    if args.compare_predictions:
        print(f"Compare:     {args.compare_predictions} (ΔΔHOTA sort)")
    if args.captions:
        print(f"Captions:    {args.captions}")
    print(f"Data dir:    {args.data_dir}")
    print(f"Listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
