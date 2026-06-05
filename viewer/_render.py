"""Stateless render/IO helpers shared by viewer apps."""
import colorsys
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

MASK_OVERLAY_ALPHA = 140


def obj_color(obj_id) -> tuple[int, int, int]:
    try:
        idx = int(obj_id)
    except (ValueError, TypeError):
        idx = abs(hash(str(obj_id)))
    hue = (idx * 0.6180339887) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def list_video_frames(data_dir: str | Path, video: str) -> list[int]:
    video_dir = Path(data_dir) / "JPEGImages" / video
    if not video_dir.exists():
        return []
    frames = []
    for jpg in video_dir.glob(f"{video}_*.jpg"):
        try:
            frames.append(int(jpg.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    frames.sort()
    return frames


def load_masks_for(data_dir: str | Path, video: str, qid: int) -> dict | None:
    path = Path(data_dir) / "MasksRLE" / video / f"{qid}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def mask_frames_with_gt(masks: dict | None) -> list[int]:
    if not masks:
        return []
    n = max(len(v) for v in masks.values())
    has = [False] * n
    for frames in masks.values():
        for i, m in enumerate(frames):
            if m is not None:
                has[i] = True
    return [i for i, b in enumerate(has) if b]


def render_mask_png(masks: dict | None, frame_idx: int, w: int, h: int):
    if not masks:
        return None
    rgba = None
    for obj_id, frames in masks.items():
        if frame_idx >= len(frames) or frames[frame_idx] is None:
            continue
        rle = frames[frame_idx]
        if isinstance(rle.get("counts"), str):
            rle = {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
        decoded = mask_utils.decode(rle).astype(bool)
        if rgba is None:
            rgba = np.zeros((decoded.shape[0], decoded.shape[1], 4), dtype=np.uint8)
        r, g, b = obj_color(obj_id)
        rgba[decoded, 0] = r
        rgba[decoded, 1] = g
        rgba[decoded, 2] = b
        rgba[decoded, 3] = MASK_OVERLAY_ALPHA
    if rgba is None:
        return None
    img = Image.fromarray(rgba, mode="RGBA")
    if img.size != (w, h):
        img = img.resize((w, h), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def resolve_qid_from_expression(
    expression: str,
    video: str,
    queries: dict,
    default_expression: str | None = None,
) -> int:
    expression = (expression or "").strip()
    if default_expression and expression == default_expression:
        return 0
    for q in queries.get(video, []):
        if q.get("expression") == expression:
            return int(q.get("qid", 0))
    return 0
