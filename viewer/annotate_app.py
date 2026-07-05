"""Interactive track-correction annotation app.

Load CFC videos + initial track predictions, then iteratively refine tracks by
manual point edits or by prompting the trained Molmo2 correction model (vLLM).
Each refinement is a node in a branching tree per video; sessions autosave to
--session_dir after every mutation and survive restarts.

Usage:
    # GPU-free (browse / manual edits / tree / export):
    python viewer/annotate_app.py --no-model

    # with the correction model (loads in a background thread at startup):
    python viewer/annotate_app.py \
        --model_dir runs/cfc_all_real_llm_connector_vit/step300-hf

    # then tunnel:  ssh -L 6006:localhost:6006 <a100-node>

See docs/annotate_interface.md for the full guide and viewer/track_io.py for
data formats + coordinate conventions.
"""
import argparse
import logging
import sys
import threading
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _render  # noqa: E402
import track_io  # noqa: E402

log = logging.getLogger("annotate")

HTML_PATH = Path(__file__).resolve().parent / "annotate.html"
DEFAULT_MODEL_DIR = "runs/cfc_all_real_llm_connector_vit/step300-hf"

app = FastAPI(title="Track Correction Annotator")


class State:
    data_dir: Path = None
    session_dir: Path = None
    model_dir: str | None = None
    export_box_size: float = 20.0
    sessions: dict = {}          # video -> session dict (mirrors disk)
    jobs: dict = {}              # job_id -> {status, video, parent_id, prompt, node?, error?}
    runner = None                # inference.ModelRunner | None
    _frames_cache: dict = {}
    _lock = threading.Lock()     # guards sessions + jobs mutations


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _videos_dir() -> Path:
    return State.data_dir / "videos"


def _list_frames(video: str) -> list[int]:
    if video not in State._frames_cache:
        State._frames_cache[video] = _render.list_video_frames(State.data_dir, video)
    return State._frames_cache[video]


def _get_session(video: str) -> dict:
    s = State.sessions.get(video)
    if s is None:
        p = track_io.session_path(State.session_dir, video)
        if p.exists():
            s = track_io.load_session(p)
            State.sessions[video] = s
    if s is None:
        raise HTTPException(404, f"no session for video {video!r}; load it first")
    return s


def _save(session: dict):
    track_io.save_session(session,
                          track_io.session_path(State.session_dir, session["video"]))


# ---------------------------------------------------------------------------
# static / config / browse
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    with open(HTML_PATH) as f:
        return f.read()


@app.get("/api/config")
def get_config():
    return {
        "data_dir": str(State.data_dir),
        "videos_dir": str(_videos_dir()),
        "session_dir": str(State.session_dir),
        "model_dir": State.model_dir,
        "model_enabled": State.runner is not None,
        "videos_listing_ok": _videos_dir().is_dir(),
        "export_box_size": State.export_box_size,
    }


@app.get("/api/browse/videos")
def browse_videos(q: str = "", limit: int = 500):
    """Filtered video basenames (no .mp4). Missing/odd dir -> empty list, no error."""
    vdir = _videos_dir()
    if not vdir.is_dir():
        return {"videos": [], "truncated": False, "listing_ok": False}
    q = q.lower()
    names = []
    truncated = False
    try:
        for p in sorted(vdir.iterdir()):
            if p.suffix != ".mp4":
                continue
            if q and q not in p.stem.lower():
                continue
            names.append(p.stem)
            if len(names) >= limit:
                truncated = True
                break
    except OSError:
        return {"videos": [], "truncated": False, "listing_ok": False}
    return {"videos": names, "truncated": truncated, "listing_ok": True}


@app.get("/api/browse/fs")
def browse_fs(path: str = ".", q: str = ""):
    """Directory listing for the source-file picker (json/jsonl only)."""
    p = Path(path).expanduser()
    if p.is_file():
        return {"path": str(p), "dirs": [], "files": [p.name], "is_file": True}
    if not p.is_dir():
        return {"path": str(p), "dirs": [], "files": [], "is_file": False,
                "error": "not a directory"}
    q = q.lower()
    dirs, files = [], []
    try:
        for child in sorted(p.iterdir()):
            if child.name.startswith("."):
                continue
            if q and q not in child.name.lower():
                continue
            if child.is_dir():
                dirs.append(child.name)
            elif child.suffix in (".json", ".jsonl"):
                files.append(child.name)
            if len(dirs) + len(files) >= 500:
                break
    except OSError as e:
        return {"path": str(p), "dirs": [], "files": [], "is_file": False,
                "error": str(e)}
    return {"path": str(p.resolve()), "parent": str(p.resolve().parent),
            "dirs": dirs, "files": files, "is_file": False}


class InspectPayload(BaseModel):
    path: str


@app.post("/api/inspect_source")
def inspect_source(payload: InspectPayload):
    return track_io.inspect_source(payload.path)


# ---------------------------------------------------------------------------
# load / sessions
# ---------------------------------------------------------------------------

class LoadPayload(BaseModel):
    videos: list[str]
    source_path: str
    trajectory_id: int | None = None
    force_reinit: bool = False


@app.post("/api/load")
def load_videos(payload: LoadPayload):
    """Create (or resume) a session per video. First point where frames and
    metadata are touched — nothing is loaded before the user confirms."""
    kind, err = track_io.detect_source(payload.source_path)
    results = []
    for video in payload.videos:
        video = video.strip().removesuffix(".mp4")
        video = Path(video).name if "/" in video else video
        try:
            sp = track_io.session_path(State.session_dir, video)
            if sp.exists() and not payload.force_reinit:
                State.sessions[video] = track_io.load_session(sp)
                results.append({"video": video, "ok": True, "resumed": True,
                                "n_nodes": len(State.sessions[video]["nodes"])})
                continue
            if err:
                raise ValueError(f"bad source file: {err}")
            steps, meta = track_io.load_source_for_video(
                payload.source_path, kind, video, payload.trajectory_id)
            meta = track_io.get_video_meta(State.data_dir, video, meta)
            if not (meta.get("width") and meta.get("height")):
                raise ValueError("could not determine video dimensions "
                                 "(no source meta and no JPEGImages frame 0)")
            if not meta.get("n_frames"):
                frames = _list_frames(video)
                meta["n_frames"] = (max(frames) + 1) if frames else None
            if not meta.get("n_frames"):
                raise ValueError(f"no frames found under JPEGImages/{video}")
            session = track_io.build_session(
                video, _videos_dir() / f"{video}.mp4", meta,
                {"kind": kind, "path": payload.source_path,
                 "trajectory_id": payload.trajectory_id},
                steps)
            with State._lock:
                State.sessions[video] = session
                _save(session)
            results.append({"video": video, "ok": True, "resumed": False,
                            "n_nodes": len(session["nodes"])})
        except (ValueError, OSError) as e:
            results.append({"video": video, "ok": False, "error": str(e)})
    return {"sessions": results}


@app.get("/api/videos")
def list_videos():
    """Loaded + on-disk sessions."""
    seen = {}
    for s in track_io.list_sessions(State.session_dir):
        seen[s["video"]] = {"video": s["video"], "n_nodes": s["n_nodes"],
                            "selected_node_id": s["selected_node_id"],
                            "loaded": s["video"] in State.sessions}
    for v, s in State.sessions.items():
        seen[v] = {"video": v, "n_nodes": len(s["nodes"]),
                   "selected_node_id": s["selected_node_id"], "loaded": True}
    return sorted(seen.values(), key=lambda x: x["video"])


@app.get("/api/session/{video}")
def get_session(video: str):
    s = _get_session(video)
    return dict(s, available_frames=_list_frames(video))


@app.get("/api/frame/{video}/{frame_idx}")
def get_frame(video: str, frame_idx: int):
    p = State.data_dir / "JPEGImages" / video / f"{video}_{frame_idx}.jpg"
    if not p.exists():
        raise HTTPException(404, f"frame not found: {p}")
    return FileResponse(p, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# tree mutations
# ---------------------------------------------------------------------------

class SelectPayload(BaseModel):
    node_id: str


@app.post("/api/session/{video}/select")
def select_node(video: str, payload: SelectPayload):
    s = _get_session(video)
    if payload.node_id not in s["nodes"]:
        raise HTTPException(400, f"unknown node {payload.node_id!r}")
    with State._lock:
        s["selected_node_id"] = payload.node_id
        _save(s)
    return {"ok": True}


class ManualPayload(BaseModel):
    parent_id: str
    tracks: dict
    counts: dict = {}


@app.post("/api/session/{video}/node/manual")
def add_manual_node(video: str, payload: ManualPayload):
    s = _get_session(video)
    c = payload.counts
    prompt = (f"User added {c.get('added', 0)} points, "
              f"deleted {c.get('deleted', 0)} points, "
              f"moved {c.get('moved', 0)} points")
    with State._lock:
        try:
            node = track_io.add_node(s, payload.parent_id, "manual", prompt,
                                     payload.tracks)
        except ValueError as e:
            raise HTTPException(400, str(e))
        _save(s)
    return {"node": node}


@app.delete("/api/session/{video}/node/{node_id}")
def delete_node(video: str, node_id: str):
    s = _get_session(video)
    with State._lock:
        try:
            deleted = track_io.delete_subtree(s, node_id)
        except ValueError as e:
            raise HTTPException(400, str(e))
        _save(s)
    return {"ok": True, "deleted_ids": deleted,
            "selected_node_id": s["selected_node_id"]}


# ---------------------------------------------------------------------------
# model inference
# ---------------------------------------------------------------------------

class ModelPayload(BaseModel):
    parent_id: str
    prompt: str


@app.post("/api/session/{video}/node/model")
def add_model_node(video: str, payload: ModelPayload):
    if State.runner is None:
        raise HTTPException(409, "model disabled (--no-model); manual edits only")
    status = State.runner.status()
    if status["state"] != "ready":
        raise HTTPException(409, f"model not ready: {status['state']}"
                            + (f" ({status.get('error')})" if status.get("error") else ""))
    s = _get_session(video)
    if payload.parent_id not in s["nodes"]:
        raise HTTPException(400, f"unknown node {payload.parent_id!r}")
    if not payload.prompt.strip():
        raise HTTPException(400, "empty prompt")
    with State._lock:
        if any(j["status"] == "running" for j in State.jobs.values()):
            raise HTTPException(409, "a model job is already running")
        job_id = uuid.uuid4().hex[:12]
        State.jobs[job_id] = {"status": "running", "video": video,
                              "parent_id": payload.parent_id,
                              "prompt": payload.prompt}
    threading.Thread(target=_run_model_job,
                     args=(job_id, video, payload.parent_id, payload.prompt),
                     daemon=True).start()
    return {"job_id": job_id}


def _run_model_job(job_id: str, video: str, parent_id: str, prompt: str):
    job = State.jobs[job_id]
    try:
        session = State.sessions[video]
        parent = session["nodes"].get(parent_id)
        if parent is None:
            raise ValueError(f"parent node {parent_id} disappeared")
        raw_text, tracks = State.runner.run_correction(session, parent["tracks"],
                                                       prompt)
        with State._lock:
            if parent_id not in session["nodes"]:
                raise ValueError("parent node was deleted while the job ran")
            node = track_io.add_node(session, parent_id, "model", prompt,
                                     tracks, model_raw_output=raw_text)
            _save(session)
        job.update(status="done", node=node)
    except Exception as e:  # noqa: BLE001 - surfaced to the UI
        log.exception("model job %s failed", job_id)
        job.update(status="error", error=str(e))


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = State.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job (jobs do not survive restarts)")
    return job


@app.get("/api/model/status")
def model_status():
    if State.runner is None:
        return {"state": "disabled"}
    return State.runner.status()


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

class ExportPayload(BaseModel):
    leaf_node_id: str
    output_path: str | None = None


@app.post("/api/session/{video}/export")
def export_session(video: str, payload: ExportPayload):
    s = _get_session(video)
    out = payload.output_path or str(State.session_dir / "exports"
                                     / f"{video}_export.jsonl")
    try:
        res = track_io.export_jsonl(s, payload.leaf_node_id, out,
                                    box_size=State.export_box_size)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, **res}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default="data/video_datasets/video_track/CFC",
                        help="Dataset root with videos/, JPEGImages/, annotations/")
    parser.add_argument("--session_dir", default="viewer/annotate_sessions",
                        help="Where per-video session JSONs (the trees) live")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR,
                        help="HF checkpoint for the correction model")
    parser.add_argument("--no-model", action="store_true",
                        help="Run without vLLM/GPU (manual edits still work)")
    parser.add_argument("--export_box_size", type=float, default=20.0,
                        help="Side (px) of the fixed bbox written around each "
                             "point on jsonl export")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    State.data_dir = Path(args.data_dir)
    State.session_dir = Path(args.session_dir)
    State.session_dir.mkdir(parents=True, exist_ok=True)
    State.export_box_size = args.export_box_size
    State.model_dir = None if args.no_model else args.model_dir

    if State.model_dir:
        # Import (pulls vllm) and load in the background so the UI is usable
        # immediately; load failures (e.g. CUDA OOM) land in /api/model/status.
        def _boot():
            try:
                import inference
                State.runner = inference.ModelRunner(State.model_dir)
                State.runner.load()
            except Exception:  # noqa: BLE001
                log.exception("model boot failed")
                if State.runner is None:
                    import types
                    err = "import of inference/vllm failed; see server log"
                    State.runner = types.SimpleNamespace(
                        status=lambda: {"state": "error", "error": err},
                        run_correction=None)
        threading.Thread(target=_boot, daemon=True).start()
        print(f"Model:       {State.model_dir} (loading in background)")
    else:
        print("Model:       DISABLED (--no-model)")
    print(f"Data dir:    {State.data_dir}")
    print(f"Sessions:    {State.session_dir}")
    print(f"Listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
