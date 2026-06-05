"""FastAPI app for reviewing/editing AI-generated correction captions.

Loads predictions (for point/mask display) + corrections JSON; lets user
confirm or edit the `correction` text per video and appends to an output
jsonl: one `{"video_id": "...", "caption": "..."}` per line.

Usage:
    python viewer/edit_app.py \\
        --predictions runs/cfc_track/results/validation-v2/predictions.json \\
        --corrections data/.../caption_annotations/all-rivers-val-v2-vision-corrections-real-text.json \\
        --output data/.../caption_annotations/all-rivers-val-v2-reviewed.jsonl \\
        --port 6007
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _render  # noqa: E402

EDIT_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "edit.html")

CFC_DEFAULT_EXPRESSION = "fish"

app = FastAPI(title="Caption Review")


class State:
    predictions_path: str | None = None
    corrections_path: str | None = None
    output_path: str | None = None
    data_dir: str | None = None
    queries_path: str | None = None
    _preds_by_video: dict | None = None
    _corrections: dict | None = None
    _reviewed: set | None = None
    _latest_edit: dict | None = None
    _frames_cache: dict = {}
    _masks_cache: dict = {}
    _queries_cache: dict | None = None


def _get_queries() -> dict:
    if State._queries_cache is not None:
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


def _resolve_qid(p: dict) -> int:
    ex_id = p.get("example_id", "")
    if "__" in ex_id:
        try:
            return int(ex_id.rsplit("__", 1)[1])
        except ValueError:
            pass
    return _render.resolve_qid_from_expression(
        p.get("expression", ""), p.get("video", ""),
        _get_queries(), default_expression=CFC_DEFAULT_EXPRESSION,
    )


def _parse_prediction(p: dict) -> dict:
    """Return {frame_idx: [{obj_id, x, y}, ...]}."""
    pred = p.get("prediction")
    if pred is None or pred == "":
        return {}
    video_fps = p["video_fps"]
    frames = defaultdict(list)
    if isinstance(pred, list):
        for obj_id, t, x, y in pred:
            frame = round(float(t) * video_fps)
            frames[frame].append({"obj_id": int(obj_id),
                                  "x": float(x), "y": float(y)})
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


def _list_video_frames(video: str) -> list[int]:
    if video in State._frames_cache:
        return State._frames_cache[video]
    frames = _render.list_video_frames(State.data_dir, video)
    State._frames_cache[video] = frames
    return frames


def _load_masks_for_video(video: str, p: dict) -> dict | None:
    if video in State._masks_cache:
        return State._masks_cache[video]
    qid = _resolve_qid(p) if p else 0
    State._masks_cache[video] = _render.load_masks_for(State.data_dir, video, qid)
    return State._masks_cache[video]


def _load():
    if State._preds_by_video is not None:
        return
    with open(State.predictions_path) as f:
        preds = json.load(f)
    State._preds_by_video = {}
    for p in preds:
        v = p.get("video")
        if v and v not in State._preds_by_video:
            State._preds_by_video[v] = p

    with open(State.corrections_path) as f:
        State._corrections = json.load(f)

    State._reviewed = set()
    State._latest_edit = {}
    out = Path(State.output_path)
    if out.exists():
        with open(out) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    vid = obj["video_id"]
                    State._reviewed.add(vid)
                    State._latest_edit[vid] = obj.get("caption", "")
                except (json.JSONDecodeError, KeyError):
                    continue
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()


@app.get("/", response_class=HTMLResponse)
def index():
    with open(EDIT_HTML_PATH) as f:
        return f.read()


@app.get("/api/meta")
def get_meta():
    _load()
    confidences = sorted({
        (c.get("confidence") or "").strip()
        for c in State._corrections.values()
        if c.get("confidence")
    })
    return {
        "output_path": State.output_path,
        "confidences": confidences,
        "n_total": sum(1 for v in State._corrections if v in State._preds_by_video),
        "n_reviewed": len(State._reviewed),
    }


@app.get("/api/videos")
def list_videos():
    _load()
    out = []
    for vid, entry in State._corrections.items():
        if vid not in State._preds_by_video:
            continue
        p = State._preds_by_video[vid]
        out.append({
            "video_id": vid,
            "expression": p.get("expression", ""),
            "confidence": entry.get("confidence", ""),
            "reviewed": vid in State._reviewed,
            "n_frames": len(_list_video_frames(vid)),
        })
    out.sort(key=lambda x: x["video_id"])
    return out


@app.get("/api/video/{video_id}")
def get_video(video_id: str):
    _load()
    entry = State._corrections.get(video_id)
    if entry is None:
        raise HTTPException(404, f"video_id {video_id} not in corrections")
    p = State._preds_by_video.get(video_id)
    if p is None:
        raise HTTPException(404, f"video_id {video_id} not in predictions")
    frames_with_preds = _parse_prediction(p)
    available_frames = _list_video_frames(video_id)
    masks = _load_masks_for_video(video_id, p)
    mask_frames = _render.mask_frames_with_gt(masks)
    original = entry.get("correction", "")
    current = State._latest_edit.get(video_id, original)
    return {
        "video_id": video_id,
        "expression": p.get("expression", ""),
        "w": p["w"],
        "h": p["h"],
        "video_fps": p["video_fps"],
        "original_correction": original,
        "current_caption": current,
        "confidence": entry.get("confidence", ""),
        "reasoning": entry.get("reasoning", ""),
        "reviewed": video_id in State._reviewed,
        "frames_with_preds": {str(k): v for k, v in frames_with_preds.items()},
        "available_frames": available_frames,
        "mask_frames": mask_frames,
    }


@app.get("/api/frame/{video}/{frame_idx}")
def get_frame(video: str, frame_idx: int):
    p = Path(State.data_dir) / "JPEGImages" / video / f"{video}_{frame_idx}.jpg"
    if not p.exists():
        raise HTTPException(404, f"frame not found: {p}")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/masks/{video}/{frame_idx}")
def get_mask(video: str, frame_idx: int):
    _load()
    p = State._preds_by_video.get(video)
    if p is None:
        raise HTTPException(404, f"video {video} not loaded")
    masks = _load_masks_for_video(video, p)
    if not masks:
        raise HTTPException(404, "no MasksRLE for video")
    png = _render.render_mask_png(masks, frame_idx, p["w"], p["h"])
    if png is None:
        raise HTTPException(404, f"no mask at frame {frame_idx}")
    return Response(content=png, media_type="image/png")


class SavePayload(BaseModel):
    video_id: str
    caption: str


@app.post("/api/save")
def save(payload: SavePayload):
    _load()
    if payload.video_id not in State._corrections:
        raise HTTPException(400, f"unknown video_id {payload.video_id}")
    line = json.dumps({"video_id": payload.video_id, "caption": payload.caption},
                      ensure_ascii=False)
    with open(State.output_path, "a") as f:
        f.write(line + "\n")
    State._reviewed.add(payload.video_id)
    State._latest_edit[payload.video_id] = payload.caption
    return {"ok": True, "reviewed_count": len(State._reviewed)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                        help="Path to predictions.json")
    parser.add_argument("--corrections", required=True,
                        help="Path to vision-corrections JSON "
                             "(video_id -> {correction, confidence, reasoning})")
    parser.add_argument("--output", required=True,
                        help="Output jsonl path (will be created/appended)")
    parser.add_argument("--data_dir", default="data/video_datasets/video_track/CFC",
                        help="Dataset root with JPEGImages/ and MasksRLE/")
    parser.add_argument("--queries", default=None,
                        help="Optional queries.json for qid resolution")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6007)
    args = parser.parse_args()
    State.predictions_path = args.predictions
    State.corrections_path = args.corrections
    State.output_path = args.output
    State.data_dir = args.data_dir
    State.queries_path = args.queries
    print(f"Predictions: {args.predictions}")
    print(f"Corrections: {args.corrections}")
    print(f"Output:      {args.output}")
    print(f"Data dir:    {args.data_dir}")
    print(f"Listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
