"""Interactive track-correction annotation app.

Load CFC videos + initial track predictions, then iteratively refine tracks by
manual point edits or by prompting the trained Molmo2 correction model (vLLM).
Each refinement is a node in a branching tree per session; sessions autosave to
--session_dir after every mutation and survive restarts. A video can have
several parallel sessions distinguished by a tag (e.g. one per checkpoint when
comparing models across two app instances sharing the session dir).

Usage:
    python viewer/annotate_app.py
    # then tunnel:  ssh -L 6006:localhost:6006 <a100-node>

The model checkpoint is picked IN THE UI (setup step 1) and loads in a
background thread via POST /api/model/load; until then the app is fully
usable GPU-free (browse / manual edits / tree / export). One checkpoint per
process: switching checkpoints requires an app restart (vLLM cannot reliably
free GPU memory in-process).

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
    default_model_dir: str = DEFAULT_MODEL_DIR   # prefilled in the UI
    model_dir: str | None = None                 # set once /api/model/load runs
    export_box_size: float = 20.0
    sessions: dict = {}          # sid -> session dict (mirrors disk)
    jobs: dict = {}              # job_id -> {status, sid, parent_id, prompt, node?, error?}
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


def _get_session(sid: str) -> dict:
    s = State.sessions.get(sid)
    if s is None:
        p = track_io.session_path(State.session_dir, sid)
        if p.exists():
            s = track_io.load_session(p)
            State.sessions[sid] = s
    if s is None:
        raise HTTPException(404, f"no session {sid!r}; load it first")
    return s


def _save(session: dict):
    sid = track_io.session_id(session["video"], session.get("tag"))
    track_io.save_session(session, track_io.session_path(State.session_dir, sid))


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
        "default_model_dir": State.default_model_dir,
        "model_dir": State.model_dir,
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


class CheckVideosPayload(BaseModel):
    videos: list[str]


@app.post("/api/videos/check")
def check_videos(payload: CheckVideosPayload):
    """Which of these video basenames have an mp4 on disk. Used by the setup
    page to filter the pick list to a predictions file's videos — a direct
    existence check, NOT the (truncated) directory listing above."""
    vdir = _videos_dir()
    exist = [v for v in payload.videos if (vdir / f"{v}.mp4").is_file()]
    return {"exist": exist}


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
    source_path: str | None = None   # None/empty = start from an empty root
    trajectory_id: int | None = None
    force_reinit: bool = False
    tag: str = ""                    # distinguishes parallel sessions per video


@app.post("/api/load")
def load_videos(payload: LoadPayload):
    """Create (or resume) a session per video. First point where frames and
    metadata are touched — nothing is loaded before the user confirms."""
    source_path = (payload.source_path or "").strip() or None
    if source_path:
        kind, err = track_io.detect_source(source_path)
    else:
        kind, err = "none", None
    tag = track_io.sanitize_tag(payload.tag)
    results = []
    for video in payload.videos:
        video = video.strip().removesuffix(".mp4")
        video = Path(video).name if "/" in video else video
        sid = track_io.session_id(video, tag)
        try:
            sp = track_io.session_path(State.session_dir, sid)
            if sp.exists() and not payload.force_reinit:
                State.sessions[sid] = track_io.load_session(sp)
                results.append({"sid": sid, "video": video, "tag": tag,
                                "ok": True, "resumed": True,
                                "n_nodes": len(State.sessions[sid]["nodes"])})
                continue
            if err:
                raise ValueError(f"bad source file: {err}")
            if source_path:
                steps, meta = track_io.load_source_for_video(
                    source_path, kind, video, payload.trajectory_id)
            else:
                # no predictions: single empty root; first tracks come from the
                # "Track all fish" model button or from manual point edits
                steps = [{"prompt": track_io.DEFAULT_ROOT_PROMPT, "kind": "root",
                          "tracks": {}, "model_raw_output": None}]
                meta = {}
            meta = track_io.get_video_meta(State.data_dir, video, meta)
            if not (meta.get("width") and meta.get("height")):
                raise ValueError("could not determine video dimensions "
                                 "(no source meta and no JPEGImages frame 0)")
            if not meta.get("n_frames"):
                frames = _list_frames(video)
                meta["n_frames"] = (max(frames) + 1) if frames else None
            if not meta.get("n_frames"):
                raise ValueError(f"no frames found under JPEGImages/{video}")
            for step in steps:
                step["tracks"] = track_io.clip_tracks(step["tracks"],
                                                      meta["n_frames"])
            session = track_io.build_session(
                video, _videos_dir() / f"{video}.mp4", meta,
                {"kind": kind, "path": source_path,
                 "trajectory_id": payload.trajectory_id},
                steps, tag=tag)
            with State._lock:
                State.sessions[sid] = session
                _save(session)
            results.append({"sid": sid, "video": video, "tag": tag,
                            "ok": True, "resumed": False,
                            "n_nodes": len(session["nodes"])})
        except (ValueError, OSError) as e:
            results.append({"sid": sid, "video": video, "tag": tag,
                            "ok": False, "error": str(e)})
    return {"sessions": results}


@app.get("/api/videos")
def list_videos():
    """Loaded + on-disk sessions (one entry per session, not per video)."""
    seen = {}
    for s in track_io.list_sessions(State.session_dir):
        seen[s["sid"]] = {"sid": s["sid"], "video": s["video"], "tag": s["tag"],
                          "n_nodes": s["n_nodes"],
                          "selected_node_id": s["selected_node_id"],
                          "loaded": s["sid"] in State.sessions}
    for sid, s in State.sessions.items():
        seen[sid] = {"sid": sid, "video": s["video"], "tag": s.get("tag") or "",
                     "n_nodes": len(s["nodes"]),
                     "selected_node_id": s["selected_node_id"], "loaded": True}
    return sorted(seen.values(), key=lambda x: (x["video"], x["tag"]))


@app.get("/api/session/{sid}")
def get_session(sid: str):
    s = _get_session(sid)
    return dict(s, available_frames=_list_frames(s["video"]))


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


@app.post("/api/session/{sid}/select")
def select_node(sid: str, payload: SelectPayload):
    s = _get_session(sid)
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


@app.post("/api/session/{sid}/node/manual")
def add_manual_node(sid: str, payload: ManualPayload):
    s = _get_session(sid)
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


@app.delete("/api/session/{sid}/node/{node_id}")
def delete_node(sid: str, node_id: str):
    s = _get_session(sid)
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
    prompt: str = ""
    mode: str = "correction"    # "correction" | "track" (initial tracking)


@app.post("/api/session/{sid}/node/model")
def add_model_node(sid: str, payload: ModelPayload):
    if State.runner is None:
        raise HTTPException(409, "no model loaded; pick a checkpoint in setup")
    status = State.runner.status()
    if status["state"] != "ready":
        raise HTTPException(409, f"model not ready: {status['state']}"
                            + (f" ({status.get('error')})" if status.get("error") else ""))
    s = _get_session(sid)
    if payload.parent_id not in s["nodes"]:
        raise HTTPException(400, f"unknown node {payload.parent_id!r}")
    if payload.mode not in ("correction", "track"):
        raise HTTPException(400, f"unknown mode {payload.mode!r}")
    prompt = payload.prompt.strip()
    if payload.mode == "track":
        prompt = s.get("root_prompt", track_io.DEFAULT_ROOT_PROMPT)
    elif not prompt:
        raise HTTPException(400, "empty prompt")
    with State._lock:
        if any(j["status"] == "running" for j in State.jobs.values()):
            raise HTTPException(409, "a model job is already running")
        job_id = uuid.uuid4().hex[:12]
        State.jobs[job_id] = {"status": "running", "sid": sid,
                              "parent_id": payload.parent_id,
                              "prompt": prompt, "mode": payload.mode}
    threading.Thread(target=_run_model_job,
                     args=(job_id, sid, payload.parent_id, prompt,
                           payload.mode),
                     daemon=True).start()
    return {"job_id": job_id}


def _run_model_job(job_id: str, sid: str, parent_id: str, prompt: str,
                   mode: str = "correction"):
    job = State.jobs[job_id]
    try:
        session = State.sessions[sid]
        parent = session["nodes"].get(parent_id)
        if parent is None:
            raise ValueError(f"parent node {parent_id} disappeared")
        if mode == "track":
            raw_text, tracks = State.runner.run_tracking(session)
        else:
            raw_text, tracks = State.runner.run_correction(
                session, parent["tracks"], prompt)
        tracks = track_io.clip_tracks(tracks, session["n_frames"])
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


class ModelLoadPayload(BaseModel):
    model_dir: str


@app.post("/api/model/load")
def model_load(payload: ModelLoadPayload):
    """Start loading a checkpoint in a background thread. One checkpoint per
    process: vLLM cannot reliably free GPU memory in-process, so switching
    requires an app restart."""
    model_dir = payload.model_dir.strip()
    p = Path(model_dir).expanduser()
    if not p.is_dir():
        raise HTTPException(400, f"not a directory: {model_dir}")
    if not (p / "config.json").exists():
        raise HTTPException(400, f"no config.json in {model_dir} — "
                            "expected a HF checkpoint directory")
    with State._lock:
        if State.model_dir is not None:
            if Path(State.model_dir).resolve() == p.resolve():
                return {"ok": True, "model_dir": State.model_dir,
                        "already": True}
            raise HTTPException(
                409, f"checkpoint {State.model_dir} already loading/loaded; "
                     "restart the app to switch checkpoints")
        State.model_dir = model_dir

    def _boot():
        # Import (pulls vllm) and load in the background so the UI stays
        # usable; load failures (e.g. CUDA OOM) land in /api/model/status.
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
                    run_correction=None, run_tracking=None)
    threading.Thread(target=_boot, daemon=True).start()
    return {"ok": True, "model_dir": model_dir, "already": False}


@app.get("/api/model/status")
def model_status():
    if State.runner is None:
        # model_dir set but runner not created yet = the background import of
        # inference/vllm is still running
        out = {"state": "loading"} if State.model_dir else {"state": "not_loaded"}
    else:
        out = State.runner.status()
    out["model_dir"] = State.model_dir
    return out


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

class ExportPayload(BaseModel):
    leaf_node_id: str
    output_path: str | None = None


@app.post("/api/session/{sid}/export")
def export_session(sid: str, payload: ExportPayload):
    s = _get_session(sid)
    out = payload.output_path or str(State.session_dir / "exports"
                                     / f"{sid}_export.jsonl")
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
                        help="Default HF checkpoint path prefilled in the UI "
                             "(nothing loads until requested from the UI)")
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
    State.default_model_dir = args.model_dir

    print(f"Model:       pick in the UI (default {State.default_model_dir})")
    print(f"Data dir:    {State.data_dir}")
    print(f"Sessions:    {State.session_dir}")
    print(f"Listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
