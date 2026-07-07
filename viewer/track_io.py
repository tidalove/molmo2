"""Data layer for the interactive track-correction annotation app (annotate_app.py).

No vLLM / GPU imports here — this module must stay usable on a CPU-only node.

Coordinate & time conventions (single source of truth)
------------------------------------------------------
* Canonical track storage everywhere in the app:
      tracks = {track_id(str): {frame_idx(str): [x_px, y_px]}}
  i.e. pixel-space floats indexed by NATIVE-fps frame index.
* The model's 0-1000 coordinate scale and seconds timestamps exist only at the
  model boundary:
    - parsing model output: UnifiedPointFormatter.extract_tracks() unscales to
      pixels and maps time -> frame via frame = round(time * video_fps).
    - building model input: DataFormatter scales pixels back to 0-1000 using
      the turn's width/height; time = frame / video_fps (see
      tracks_to_frame_trajectories, mirroring CFCMultiTurn._build_video_annotation).
* Trajectory-annotation jsonl stores COCO bboxes; points are bbox centroids
  [x + w/2, y + h/2]. On export we write a fixed-size box centered on each
  point so the centroid round-trips exactly through the existing loaders.

Session file schema (one JSON per session under --session_dir)
---------------------------------------------------------------
A video can have several parallel sessions distinguished by an optional "tag"
(e.g. one per model checkpoint, for side-by-side comparison). The session id
(= filename stem) is "{video}__{tag}", or just "{video}" when untagged —
legacy untagged files keep working unchanged.
{
  "version": 1, "video": ..., "tag": "", "video_path": ...,
  "width": int, "height": int, "n_frames": int,
  "video_fps": float, "sampling_fps": float, "expression": "fish",
  "source": {"kind": "predictions|coco|trajectory_jsonl", "path": ..., "trajectory_id": ...},
  "root_prompt": "track all fish",
  "next_node_id": int, "next_track_id": int, "selected_node_id": str,
  "nodes": {node_id: {"id", "parent_id", "kind": "root|loaded|model|manual",
                      "prompt", "created_at", "model_raw_output", "tracks"}}
}
"""
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_ROOT_PROMPT = "track all fish"
DEFAULT_EXPRESSION = "fish"

_FORMATTER = None


def get_point_formatter():
    """html-v2 point formatter (matches the trained Molmo2 config). Lazy so
    importing this module stays fast; no vllm involved."""
    global _FORMATTER
    if _FORMATTER is None:
        from olmo.preprocessing.point_formatter import UnifiedPointFormatter
        _FORMATTER = UnifiedPointFormatter.build_for_format("html-v2")
    return _FORMATTER


# ---------------------------------------------------------------------------
# tracks <-> other representations
# ---------------------------------------------------------------------------

def parse_tracks_text(text, width, height, video_fps):
    """`<tracks coords="...">` text -> canonical tracks dict (pixel coords)."""
    point_tracks = get_point_formatter().extract_tracks(text, width, height, video_fps)
    return extracted_to_tracks(point_tracks)


def extracted_to_tracks(point_tracks):
    """extract_tracks() output ([{time, frame, points:{id:{point}}}]) -> canonical."""
    tracks = {}
    for pt in point_tracks:
        frame = str(pt["frame"])
        for tid, info in pt["points"].items():
            tracks.setdefault(str(tid), {})[frame] = [float(info["point"][0]),
                                                      float(info["point"][1])]
    return tracks


def clip_tracks(tracks, n_frames):
    """Drop points at frames >= n_frames (model timestamps can round to one
    frame past the end, e.g. 60.0s -> frame 120 of a 120-frame video; such
    points would be silently lost by the jsonl loaders on export round-trip)."""
    out, dropped = {}, 0
    for tid, tmap in tracks.items():
        kept = {f: p for f, p in tmap.items() if int(f) < n_frames}
        dropped += len(tmap) - len(kept)
        if kept:
            out[tid] = kept
    if dropped:
        log.warning("clip_tracks: dropped %d out-of-range point(s)", dropped)
    return out


def _sorted_track_ids(tracks):
    """Sort ids numerically when possible (matches sorted(track_ids) of ints
    in CFCMultiTurn._build_video_annotation)."""
    def key(tid):
        try:
            return (0, int(tid))
        except ValueError:
            return (1, tid)
    return sorted(tracks, key=key)


def tracks_to_frame_trajectories(tracks, n_frames, video_fps):
    """Canonical tracks -> the `points` list of a multi_turn_messages turn.

    Mirrors CFCMultiTurn._build_video_annotation: one entry per frame
    0..n_frames-1, obj slots = enumerate(sorted track ids), time = frame/fps.
    """
    track_ids = _sorted_track_ids(tracks)
    tid_to_obj = {tid: idx for idx, tid in enumerate(track_ids)}
    frame_trajectories = []
    for frame_idx in range(n_frames):
        points_dict = {}
        for tid in track_ids:
            pt = tracks[tid].get(str(frame_idx))
            if pt is None:
                continue
            points_dict[tid_to_obj[tid]] = {"point": [pt[0], pt[1]], "occluded": False}
        frame_trajectories.append({
            "frame": frame_idx,
            "time": frame_idx / video_fps,
            "points": points_dict,
        })
    return frame_trajectories


def tracks_to_annotations(tracks, box_size=20.0):
    """Canonical tracks -> jsonl-style COCO annotations (fixed-size boxes
    centered on each point; centroid round-trips exactly). image_id convention:
    frame_idx + 1 (matches images built by export_jsonl)."""
    anns = []
    for slot, tid in enumerate(_sorted_track_ids(tracks)):
        try:
            int_tid = int(tid)
        except ValueError:
            int_tid = slot + 1
        for frame_str, (x, y) in sorted(tracks[tid].items(), key=lambda kv: int(kv[0])):
            anns.append({
                "image_id": int(frame_str) + 1,
                "bbox": [x - box_size / 2, y - box_size / 2, box_size, box_size],
                "track_id": int_tid,
                "category_id": 1,
                "model_track_id": int_tid,
            })
    return anns


# ---------------------------------------------------------------------------
# source files: detection / inspection / loading
# ---------------------------------------------------------------------------

def detect_source(path):
    """Sniff a source file. Returns (kind, error): kind in
    {"predictions", "coco", "trajectory_jsonl"} or None with an error string.
    Never raises on bad input."""
    p = Path(path)
    if not p.exists():
        return None, f"file not found: {path}"
    if p.is_dir():
        return None, f"is a directory: {path}"
    try:
        if p.suffix == ".jsonl":
            with open(p) as f:
                first = f.readline()
            obj = json.loads(first)
            if isinstance(obj, dict) and "video_name" in obj and "trajectories" in obj:
                return "trajectory_jsonl", None
            return None, "jsonl first line lacks video_name/trajectories"
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, list):
            if data and isinstance(data[0], dict) and "prediction" in data[0]:
                return "predictions", None
            return None, "json list entries lack a 'prediction' field"
        if isinstance(data, dict):
            if "video_name" in data and "trajectories" in data:
                return "trajectory_jsonl", None
            if "images" in data and "annotations" in data:
                return "coco", None
        return None, "unrecognized json structure (want predictions / COCO / trajectory jsonl)"
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        return None, f"could not parse: {e}"


def inspect_source(path):
    """Summary of a source file for the UI: kind + per-video info (drives the
    trajectory dropdown for jsonl sources)."""
    kind, err = detect_source(path)
    if err:
        return {"kind": None, "error": err}
    out = {"kind": kind, "error": None, "videos": [], "trajectories": []}
    try:
        if kind == "predictions":
            with open(path) as f:
                entries = json.load(f)
            out["videos"] = sorted({e.get("video") for e in entries if e.get("video")})
        elif kind == "coco":
            with open(path) as f:
                coco = json.load(f)
            vids = {v["id"] for v in coco.get("videos", [])}
            if not vids:
                for img in coco.get("images", []):
                    clip, _, frame = _split_frame_filename(img.get("file_name", ""))
                    if clip:
                        vids.add(clip)
            out["videos"] = sorted(vids)
        else:  # trajectory_jsonl
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    video = rec.get("video_name")
                    if not video:
                        continue
                    out["videos"].append(video)
                    for traj in rec.get("trajectories", []):
                        steps = sorted(traj.get("correction_steps", []),
                                       key=lambda s: s.get("correction_step", 0))
                        out["trajectories"].append({
                            "video": video,
                            "trajectory_id": traj.get("trajectory_id", 0),
                            "n_steps": len(steps),
                            "prompts": [s.get("prompt", "") for s in steps],
                        })
            out["videos"].sort()
    except (OSError, json.JSONDecodeError) as e:
        return {"kind": kind, "error": f"could not inspect: {e}"}
    return out


# turn parser for the `input` field stored in predictions.json
_IM_TURN_RE = re.compile(r"<\|im_start\|>(user|assistant)\s*\n(.*?)(?:<\|im_end\|>|\Z)",
                         re.S)


def _parse_chat_input(input_text):
    """predictions.json `input` -> (user_texts, assistant_texts)."""
    users, assistants = [], []
    for role, text in _IM_TURN_RE.findall(input_text or ""):
        text = text.replace("<|video|>", "").strip()
        if role == "user":
            users.append(text)
        elif text:
            assistants.append(text)
    return users, assistants


def _split_frame_filename(file_name):
    """'{clip}_{frame}.jpg' -> (clip, frame_str_ok, frame_int)."""
    stem = file_name[:-4] if file_name.endswith(".jpg") else file_name
    clip, _, frame_str = stem.rpartition("_")
    try:
        return clip, True, int(frame_str)
    except ValueError:
        return None, False, -1


def load_source_for_video(path, kind, video, trajectory_id=None):
    """Load one video's step chain from a source file.

    Returns (steps, meta):
      steps: [{"prompt", "kind", "tracks", "model_raw_output"}] in chain order
             (steps[0] is the root).
      meta:  {"width", "height", "video_fps", "n_frames", "expression"} —
             entries may be None; caller fills fallbacks (get_video_meta).
    Raises ValueError with a user-facing message on failure.
    """
    if kind == "predictions":
        return _load_predictions_source(path, video)
    if kind == "coco":
        return _load_coco_source(path, video)
    if kind == "trajectory_jsonl":
        return _load_trajectory_source(path, video, trajectory_id)
    raise ValueError(f"unknown source kind: {kind}")


def _load_predictions_source(path, video):
    with open(path) as f:
        entries = json.load(f)
    matches = [e for e in entries if e.get("video") == video]
    if not matches:
        raise ValueError(f"video {video!r} not found in predictions file")
    entry = matches[0]
    if len(matches) > 1:
        log.warning("multiple predictions entries for %s; using the first (%s)",
                    video, entry.get("example_id"))
    w, h = entry.get("w"), entry.get("h")
    video_fps = entry.get("video_fps") or entry.get("sampling_fps")
    if not (w and h and video_fps):
        raise ValueError(f"predictions entry for {video} lacks w/h/video_fps")

    steps = []
    users, assistants = _parse_chat_input(entry.get("input", ""))
    if users and assistants:
        # Reconstruct the full 1-step trajectory: root = step0 tracks from the
        # input's assistant turn, child = the model's prediction.
        steps.append({
            "prompt": users[0] or DEFAULT_ROOT_PROMPT,
            "kind": "root",
            "tracks": parse_tracks_text(assistants[0], w, h, video_fps),
            "model_raw_output": None,
        })
        correction_prompt = users[1] if len(users) > 1 else "(correction)"
        steps.append({
            "prompt": correction_prompt,
            "kind": "model",
            "tracks": parse_tracks_text(entry.get("prediction", ""), w, h, video_fps),
            "model_raw_output": entry.get("prediction"),
        })
    else:
        log.warning("predictions entry for %s has no parseable input chat; "
                    "loading prediction as the root node", video)
        steps.append({
            "prompt": DEFAULT_ROOT_PROMPT,
            "kind": "root",
            "tracks": parse_tracks_text(entry.get("prediction", ""), w, h, video_fps),
            "model_raw_output": entry.get("prediction"),
        })
    meta = {"width": w, "height": h, "video_fps": video_fps, "n_frames": None,
            "expression": entry.get("expression") or DEFAULT_EXPRESSION}
    return steps, meta


def _load_coco_source(path, video):
    with open(path) as f:
        coco = json.load(f)
    prefix = f"{video}_"
    img_to_frame, w, h = {}, None, None
    for img in coco.get("images", []):
        fn = img.get("file_name", "")
        if not fn.startswith(prefix):
            continue
        clip, ok, frame = _split_frame_filename(fn)
        if ok and clip == video:
            img_to_frame[img["id"]] = frame
            if w is None:
                w, h = img.get("width"), img.get("height")
    if not img_to_frame:
        raise ValueError(f"video {video!r} not found in COCO file (no matching images)")

    video_fps = None
    for v in coco.get("videos", []):
        if v.get("id") == video:
            video_fps = v.get("fps")
            break

    tracks = {}
    for ann in coco.get("annotations", []):
        frame = img_to_frame.get(ann.get("image_id"))
        if frame is None:
            continue
        x, y, bw, bh = ann["bbox"]
        tid = str(ann.get("track_id", 0))
        tracks.setdefault(tid, {})[str(frame)] = [x + bw / 2, y + bh / 2]

    steps = [{"prompt": DEFAULT_ROOT_PROMPT, "kind": "root",
              "tracks": tracks, "model_raw_output": None}]
    n_frames = max(img_to_frame.values()) + 1 if img_to_frame else None
    meta = {"width": w, "height": h, "video_fps": video_fps,
            "n_frames": n_frames, "expression": DEFAULT_EXPRESSION}
    return steps, meta


def _load_trajectory_source(path, video, trajectory_id=None):
    record = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("video_name") == video:
                record = rec
                break
    if record is None:
        raise ValueError(f"video {video!r} not found in trajectory jsonl")

    trajectories = record.get("trajectories", [])
    if not trajectories:
        raise ValueError(f"no trajectories for video {video!r}")
    if trajectory_id is None:
        if len(trajectories) > 1:
            raise ValueError(
                f"video {video!r} has {len(trajectories)} trajectories; "
                "pick a trajectory_id")
        traj = trajectories[0]
    else:
        traj = next((t for t in trajectories
                     if t.get("trajectory_id") == trajectory_id), None)
        if traj is None:
            raise ValueError(f"trajectory_id {trajectory_id} not found for {video!r}")

    images = sorted(record.get("images", []), key=lambda im: im["id"])
    if not images:
        raise ValueError(f"no images for video {video!r}")
    image_id_to_frame = {img["id"]: idx for idx, img in enumerate(images)}
    w, h = images[0].get("width"), images[0].get("height")

    steps = []
    ordered = sorted(traj.get("correction_steps", []),
                     key=lambda s: s.get("correction_step", 0))
    for i, cs in enumerate(ordered):
        tracks = {}
        for ann in cs.get("annotations", []):
            frame = image_id_to_frame.get(ann.get("image_id"))
            if frame is None:
                continue
            x, y, bw, bh = ann["bbox"]
            tid = str(ann.get("track_id", 0))
            tracks.setdefault(tid, {})[str(frame)] = [x + bw / 2, y + bh / 2]
        steps.append({
            "prompt": cs.get("prompt", ""),
            "kind": "root" if i == 0 else "loaded",
            "tracks": tracks,
            "model_raw_output": None,
        })
    meta = {"width": w, "height": h, "video_fps": None,
            "n_frames": len(images), "expression": DEFAULT_EXPRESSION}
    return steps, meta


# ---------------------------------------------------------------------------
# video metadata fallbacks
# ---------------------------------------------------------------------------

_FPS_MAPS = {}


def _fps_map(data_dir):
    """{video_id: fps} lazily merged over all {data_dir}/annotations/*.json."""
    key = str(data_dir)
    if key in _FPS_MAPS:
        return _FPS_MAPS[key]
    fps = {}
    ann_dir = Path(data_dir) / "annotations"
    if ann_dir.is_dir():
        for jf in sorted(ann_dir.glob("*.json")):
            try:
                with open(jf) as f:
                    coco = json.load(f)
                for v in coco.get("videos", []):
                    if isinstance(v, dict) and "id" in v and v.get("fps"):
                        fps.setdefault(v["id"], v["fps"])
            except (json.JSONDecodeError, OSError, MemoryError):
                continue
    _FPS_MAPS[key] = fps
    return fps


def get_video_meta(data_dir, video, meta=None):
    """Fill missing width/height/video_fps/n_frames.

    Order: source-provided meta > frame-0 JPEG dims / JPEGImages count >
    annotations/*.json fps > av probe of the mp4 > fps default 2.0 + warning.
    """
    meta = dict(meta or {})
    frames_dir = Path(data_dir) / "JPEGImages" / video

    if not (meta.get("width") and meta.get("height")):
        f0 = frames_dir / f"{video}_0.jpg"
        if f0.exists():
            from PIL import Image
            with Image.open(f0) as im:
                meta["width"], meta["height"] = im.size

    if not meta.get("n_frames"):
        if frames_dir.is_dir():
            n = sum(1 for _ in frames_dir.glob(f"{video}_*.jpg"))
            meta["n_frames"] = n or None

    if not meta.get("video_fps"):
        meta["video_fps"] = _fps_map(data_dir).get(video)
    if not meta.get("video_fps"):
        mp4 = Path(data_dir) / "videos" / f"{video}.mp4"
        if mp4.exists():
            try:
                import av
                with av.open(str(mp4)) as container:
                    rate = container.streams.video[0].average_rate
                    if rate:
                        meta["video_fps"] = float(rate)
            except Exception as e:  # noqa: BLE001 - probe is best-effort
                log.warning("av probe failed for %s: %s", mp4, e)
    if not meta.get("video_fps"):
        log.warning("no fps found for %s; defaulting to 2.0", video)
        meta["video_fps"] = 2.0

    meta.setdefault("expression", DEFAULT_EXPRESSION)
    return meta


# ---------------------------------------------------------------------------
# session build / persistence
# ---------------------------------------------------------------------------

def _max_int_track_id(tracks):
    best = 0
    for tid in tracks:
        try:
            best = max(best, int(tid))
        except ValueError:
            continue
    return best


def sanitize_tag(tag):
    """Filesystem-safe session tag ([\\w.-] only); '' when empty/None."""
    return re.sub(r"[^\w.-]+", "-", (tag or "").strip()).strip("-")


def session_id(video, tag):
    """Session id = filename stem. Untagged sessions keep the legacy name."""
    tag = sanitize_tag(tag)
    return f"{video}__{tag}" if tag else video


def build_session(video, video_path, meta, source, steps, tag=""):
    """New session dict from a loaded step chain (steps[0] = root)."""
    now = datetime.now().isoformat(timespec="seconds")
    nodes, parent_id = {}, None
    max_tid = 0
    for i, step in enumerate(steps):
        nid = str(i)
        nodes[nid] = {
            "id": nid,
            "parent_id": parent_id,
            "kind": step["kind"] if i > 0 else "root",
            "prompt": step["prompt"],
            "created_at": now,
            "model_raw_output": step.get("model_raw_output"),
            "tracks": step["tracks"],
        }
        max_tid = max(max_tid, _max_int_track_id(step["tracks"]))
        parent_id = nid
    return {
        "version": 1,
        "video": video,
        "tag": sanitize_tag(tag),
        "video_path": str(video_path),
        "width": meta["width"],
        "height": meta["height"],
        "n_frames": meta["n_frames"],
        "video_fps": meta["video_fps"],
        "sampling_fps": meta.get("sampling_fps") or meta["video_fps"],
        "expression": meta.get("expression", DEFAULT_EXPRESSION),
        "source": source,
        "root_prompt": steps[0]["prompt"] or DEFAULT_ROOT_PROMPT,
        "next_node_id": len(steps),
        "next_track_id": max_tid + 1,
        "selected_node_id": str(len(steps) - 1),
        "nodes": nodes,
    }


def add_node(session, parent_id, kind, prompt, tracks, model_raw_output=None):
    """Append a node under parent_id, select it, and return it (caller saves)."""
    if parent_id not in session["nodes"]:
        raise ValueError(f"unknown parent node {parent_id!r}")
    nid = str(session["next_node_id"])
    node = {
        "id": nid,
        "parent_id": parent_id,
        "kind": kind,
        "prompt": prompt,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_raw_output": model_raw_output,
        "tracks": tracks,
    }
    session["nodes"][nid] = node
    session["next_node_id"] += 1
    session["next_track_id"] = max(session["next_track_id"],
                                   _max_int_track_id(tracks) + 1)
    session["selected_node_id"] = nid
    return node


def delete_subtree(session, node_id):
    """Remove node_id and all descendants. Returns deleted ids. Root refuses."""
    nodes = session["nodes"]
    if node_id not in nodes:
        raise ValueError(f"unknown node {node_id!r}")
    if nodes[node_id]["parent_id"] is None:
        raise ValueError("cannot delete the root node")
    children = {}
    for nid, n in nodes.items():
        children.setdefault(n["parent_id"], []).append(nid)
    doomed, stack = [], [node_id]
    while stack:
        nid = stack.pop()
        doomed.append(nid)
        stack.extend(children.get(nid, []))
    parent = nodes[node_id]["parent_id"]
    for nid in doomed:
        del nodes[nid]
    if session["selected_node_id"] in set(doomed) or \
            session["selected_node_id"] not in nodes:
        session["selected_node_id"] = parent
    return doomed


def path_to_root(session, node_id):
    """[root, ..., node_id] chain of node dicts."""
    nodes = session["nodes"]
    if node_id not in nodes:
        raise ValueError(f"unknown node {node_id!r}")
    chain = []
    cur = node_id
    while cur is not None:
        chain.append(nodes[cur])
        cur = nodes[cur]["parent_id"]
    return list(reversed(chain))


def session_path(session_dir, video):
    return Path(session_dir) / f"{video}.json"


def save_session(session, path):
    """Atomic write: {path}.tmp then os.replace (survives a mid-write kill)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(session, f)
    os.replace(tmp, path)


def load_session(path):
    with open(path) as f:
        return json.load(f)


def list_sessions(session_dir):
    """[{sid, video, tag, path, n_nodes, selected_node_id}] for every session
    on disk. sid = filename stem (authoritative; legacy files = video name)."""
    out = []
    sdir = Path(session_dir)
    if not sdir.is_dir():
        return out
    for jf in sorted(sdir.glob("*.json")):
        try:
            s = load_session(jf)
            out.append({"sid": jf.stem, "video": s["video"],
                        "tag": s.get("tag") or "", "path": str(jf),
                        "n_nodes": len(s["nodes"]),
                        "selected_node_id": s.get("selected_node_id")})
        except (json.JSONDecodeError, KeyError, OSError):
            log.warning("skipping unreadable session file %s", jf)
    return out


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def export_jsonl(session, leaf_node_id, out_path, box_size=20.0):
    """Append the root->leaf path as one standard correction_steps jsonl line.

    images are rebuilt with id = frame_idx + 1 and file_name
    '{video}_{frame_idx}.jpg' so the existing loaders' sorted-by-id -> frame
    mapping reproduces our frame indices exactly.
    """
    chain = path_to_root(session, leaf_node_id)
    video = session["video"]
    n_frames = session["n_frames"]
    images = [{"id": i + 1, "file_name": f"{video}_{i}.jpg",
               "height": session["height"], "width": session["width"]}
              for i in range(n_frames)]
    record = {
        "video_name": video,
        "images": images,
        "categories": [{"id": 1, "name": session.get("expression", "fish"),
                        "supercategory": ""}],
        "trajectories": [{
            "trajectory_id": 0,
            "correction_steps": [
                {"correction_step": i,
                 "prompt": node["prompt"],
                 "annotations": tracks_to_annotations(node["tracks"], box_size)}
                for i, node in enumerate(chain)
            ],
        }],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return {"path": str(out_path), "n_steps": len(chain)}


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import tempfile

    data_dir = os.environ.get("MOLMO_DATA_DIR", "data")
    cfc = Path(data_dir) / "video_datasets" / "video_track" / "CFC"
    jsonl = cfc / "caption_annotations" / "cfc_correction_eel_even.jsonl"
    if not jsonl.exists():
        sys.exit(f"self-check needs {jsonl}")

    with open(jsonl) as f:
        rec = json.loads(f.readline())
    video = rec["video_name"]
    kind, err = detect_source(jsonl)
    assert kind == "trajectory_jsonl", (kind, err)

    steps, meta = load_source_for_video(jsonl, kind, video)
    meta = get_video_meta(cfc, video, meta)
    print(f"video={video} steps={len(steps)} meta={meta}")
    assert len(steps) == 2 and steps[0]["prompt"] == "track all fish"

    session = build_session(video, cfc / "videos" / f"{video}.mp4", meta,
                            {"kind": kind, "path": str(jsonl), "trajectory_id": 0},
                            steps)
    with tempfile.TemporaryDirectory() as td:
        sp = session_path(td, video)
        save_session(session, sp)
        session2 = load_session(sp)
        assert session2 == session

        out = export_jsonl(session2, session2["selected_node_id"],
                           Path(td) / "export.jsonl")
        steps2, _ = load_source_for_video(Path(td) / "export.jsonl",
                                          "trajectory_jsonl", video)
        for s_orig, s_re in zip(steps, steps2):
            assert s_orig["prompt"] == s_re["prompt"]
            assert set(s_orig["tracks"]) == set(s_re["tracks"])
            for tid in s_orig["tracks"]:
                for fr, pt in s_orig["tracks"][tid].items():
                    pt2 = s_re["tracks"][tid][fr]
                    assert abs(pt[0] - pt2[0]) < 1e-6 and abs(pt[1] - pt2[1]) < 1e-6, \
                        (tid, fr, pt, pt2)
        print(f"export round-trip OK ({out})")

    # predictions.json chain reconstruction, if the default run exists
    preds = Path("runs/cfc_all_real_llm_connector_vit/results/"
                 "cfc_correction_eel_even_eval_2fps/eel/predictions.json")
    if preds.exists():
        kind, err = detect_source(preds)
        assert kind == "predictions", (kind, err)
        info = inspect_source(preds)
        v0 = info["videos"][0]
        steps, meta = load_source_for_video(preds, kind, v0)
        assert len(steps) == 2 and steps[0]["kind"] == "root"
        n0 = sum(len(v) for v in steps[0]["tracks"].values())
        n1 = sum(len(v) for v in steps[1]["tracks"].values())
        print(f"predictions chain OK for {v0}: root {len(steps[0]['tracks'])} tracks/"
              f"{n0} pts -> step1 {len(steps[1]['tracks'])} tracks/{n1} pts "
              f"(prompts: {steps[0]['prompt']!r} -> {steps[1]['prompt']!r})")
    print("track_io self-check passed")
