"""vLLM wrapper for the annotation app — the ONLY viewer module importing vllm.

Reuses the exact prompt-construction path of the eval pipeline
(scripts/run_vllm.py): DataFormatter with the trained Molmo2 config,
build_multi_turn_chat for the correction chat structure
(turn0 user = root prompt + video, turn0 assistant = the parent step's tracks
serialized as `<tracks coords=...>`, turn1 user = the typed correction prompt),
then processor.apply_chat_template + process_vision_info, matching
build_vllm_input's video branch.

The context only ever contains the SINGLE immediately-previous step — the
model's context window cannot fit more; steps are never chained.
"""
import logging
import sys
import threading
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "viewer") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "viewer"))

# MUST be the first vllm-related import: sets VLLM_ENABLE_V1_MULTIPROCESSING=0
# and VLLM_VIDEO_LOADER_BACKEND=molmo2 before `import vllm` (run_vllm.py:11-12).
import scripts.run_vllm as rv  # noqa: E402

from vllm import LLM  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from molmo_utils import process_vision_info  # noqa: E402

import track_io  # noqa: E402

log = logging.getLogger("annotate.inference")

MAX_TOKENS = 38000   # matches submit_scripts/eval/vllm_eval_all_cfc_eel.sh
MAX_FPS = 2          # eel evals always pass --max_fps 2


def build_correction_chat(session, parent_tracks, correction_prompt,
                          max_frames=None, frame_sample_mode=None,
                          max_fps=MAX_FPS):
    """Build the chat message list for one correction step (no GPU needed —
    also used by the prompt-oracle regression test)."""
    from olmo.data.academic_video_track_datasets import CFCMultiTurn
    video_fps = session["video_fps"]
    # "frame N" mentions -> seconds, like the training-data loader does
    correction_prompt = CFCMultiTurn.replace_frames_with_time(
        correction_prompt, video_fps)
    common = dict(
        width=session["width"],
        height=session["height"],
        label=session.get("expression", "fish"),
        sampling_fps=session["sampling_fps"],
        style="video_point_track_per_frame",
    )
    frame_trajectories = track_io.tracks_to_frame_trajectories(
        parent_tracks, session["n_frames"], video_fps)
    turn0 = dict(common,
                 question=session.get("root_prompt", track_io.DEFAULT_ROOT_PROMPT),
                 points=frame_trajectories)
    # The final turn's points only produce the (discarded) response side of
    # get_user_prompt, BUT they must be non-empty: with empty points the
    # formatter ignores `question` and substitutes a template prompt
    # (data_formatter.py:1446). Reuse the parent tracks.
    turn1 = dict(common, question=correction_prompt, points=frame_trajectories)
    raw = {"multi_turn_messages": [turn0, turn1]}
    return rv.build_multi_turn_chat(
        raw,
        video_path=session["video_path"],
        max_frames=max_frames,
        frame_sample_mode=frame_sample_mode,
        max_fps=max_fps,
        sampling_fps=session["sampling_fps"],
    )


class ModelRunner:
    """Owns the vLLM LLM + processor. load() is safe to call from a background
    thread; failures (e.g. CUDA OOM) are captured into status() instead of
    crashing the server."""

    def __init__(self, model_dir, max_fps=MAX_FPS, max_tokens=MAX_TOKENS):
        self.model_dir = model_dir
        self.max_fps = max_fps
        self.max_tokens = max_tokens
        self.llm = None
        self.processor = None
        self._state = "not_loaded"
        self._error = None
        self._load_lock = threading.Lock()
        self._gen_lock = threading.Lock()   # serialize generate() calls

    def status(self):
        out = {"state": self._state}
        if self._error:
            out["error"] = self._error
        return out

    def load(self):
        with self._load_lock:
            if self._state in ("ready", "loading"):
                return
            self._state = "loading"
            self._error = None
        try:
            import torch
            log.info("loading vLLM model from %s", self.model_dir)
            # exact args of scripts/run_vllm.py:323-339
            self.llm = LLM(
                model=self.model_dir,
                trust_remote_code=True,
                tensor_parallel_size=max(torch.cuda.device_count(), 1),
                gpu_memory_utilization=0.90,
                dtype="bfloat16",
                limit_mm_per_prompt={"image": 6, "video": 1},
                max_num_batched_tokens=36864,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                dtype="auto",
                device_map="auto",
                padding_side="left",
            )
            self._state = "ready"
            log.info("model ready")
        except Exception as e:  # noqa: BLE001 - surfaced via status()
            log.exception("model load failed")
            self._state = "error"
            self._error = f"{type(e).__name__}: {e}"

    def run_correction(self, session, parent_tracks, correction_prompt):
        """One correction step. Returns (raw_output_text, tracks)."""
        if self._state != "ready":
            raise RuntimeError(f"model not ready ({self._state}: {self._error})")
        vp = self.processor.video_processor
        messages = build_correction_chat(
            session, parent_tracks, correction_prompt,
            max_frames=vp.num_frames,
            frame_sample_mode=vp.frame_sample_mode,
            max_fps=self.max_fps,
        )
        images, videos_inputs, video_kwargs = process_vision_info(messages)
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        multi_modal_data = {}
        if images:
            multi_modal_data["image"] = images
        if videos_inputs:
            multi_modal_data["video"] = videos_inputs
        inp = {"prompt": prompt,
               "multi_modal_data": multi_modal_data,
               "mm_processor_kwargs": video_kwargs}

        with self._gen_lock:
            outputs = self.llm.generate(
                [inp], SamplingParams(max_tokens=self.max_tokens, temperature=0))
        text = outputs[0].outputs[0].text
        log.info("model output: %d chars", len(text))

        tracks = track_io.parse_tracks_text(
            text, session["width"], session["height"], session["video_fps"])
        if not tracks:
            raise ValueError(
                "model output contained no parseable <tracks>; raw output "
                f"(first 500 chars): {text[:500]!r}")
        return text, tracks
