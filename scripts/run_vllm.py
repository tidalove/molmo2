import os
import re
import json
import argparse
import logging
from glob import glob
from pathlib import Path

## Turn off multiprocessing to make the scheduling deterministic
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_VIDEO_LOADER_BACKEND"] = "molmo2"

import numpy as np
import torch

from olmo.util import prepare_cli_environment, log_metrics_to_console
from olmo.preprocessing.data_formatter import GENERAL_PROMPTS_V1, apply_keyword_prompt, build_prompt_for_inference
from olmo.data.get_dataset import get_dataset_by_name

from vllm import LLM
from vllm.sampling_params import SamplingParams
from transformers import AutoProcessor

try:
    from molmo_utils import process_vision_info
except ImportError:
    pass

log = logging.getLogger(__name__)


def get_message(
    images=None,
    video_path=None,
    max_frames=384,
    frame_sample_mode="fps",
    max_fps=None,
    sampling_fps=None,
    input_text="",
    style="demo",
):
    content = [
        dict(type="text", text=input_text, style=style)
    ]
    if images:
        image_content = [
            dict(type="image", image=image)
            for image in images
        ]
        content.extend(image_content)
    if video_path:
        video_kwargs = {
            "max_frames": max_frames,
            "frame_sample_mode": frame_sample_mode,
        }
        if max_fps is not None:
            video_kwargs["max_fps"] = max_fps
        if sampling_fps is not None:
            video_kwargs["sampling_fps"] = sampling_fps
        video_content = dict(type="video", video=video_path, **video_kwargs)
        content.append(video_content)

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def get_prompt_text(style, prompt_override, example=None):
    """Get prompt text: explicit override > style template > dataset example."""
    if prompt_override:
        return prompt_override
    if style:
        templates = GENERAL_PROMPTS_V1[style]
        # Check if templates have keyword placeholders
        keywords = sorted(re.findall("{([^{}]+)}", templates[0]))
        if keywords and example:
            return apply_keyword_prompt(templates, example, None, dbg=True)
        return templates[0]
    # Fallback for dataset-driven mode
    if example and 'message_list' in example:
        msg = example['message_list'][0]
        return build_prompt_for_inference(msg)
    if example and 'question' in example:
        return example['question']
    raise ValueError("No prompt source: provide --style or --prompt")


def build_examples(args, dataset=None):
    """Build list of examples from video_dir or dataset."""
    if args.video_dir:
        log.info(f"Building examples from {args.video_dir}")
        videos = sorted(glob(os.path.join(args.video_dir, "*.mp4")))
        if not videos:
            raise ValueError(f"No .mp4 files found in {args.video_dir}")
        examples = [{"video": v, "example_id": Path(v).stem} for v in videos]
    else:
        examples = []
        for i in range(len(dataset)):
            ex = dataset.get(i, None)
            ex_id = ex.get('metadata', {}).get('example_id', str(i))
            examples.append({"raw": ex, "video": ex["video"], "example_id": ex_id})

    if args.max_examples:
        examples = examples[:args.max_examples]

    # Shard: interleaved slicing for balanced load
    if args.num_shards is not None:
        examples = examples[args.shard_index::args.num_shards]
        log.info(f"Shard {args.shard_index}/{args.num_shards}: {len(examples)} examples")

    return examples


def build_vllm_input(example, style, prompt_override, processor):
    """Build a single vLLM input dict from an example."""
    raw = example.get("raw", None)

    # Determine prompt text
    prompt_text = get_prompt_text(style, prompt_override, raw)

    # Determine style for chat template
    effective_style = style
    if not effective_style and raw and 'message_list' in raw:
        effective_style = raw['message_list'][0].get('style', 'demo')
    if not effective_style:
        effective_style = 'demo'

    # Determine sampling_fps from dataset example if available
    sampling_fps = None
    if raw and 'sampling_fps' in raw:
        sampling_fps = raw['sampling_fps']

    frame_sample_mode = processor.video_processor.frame_sample_mode
    max_frames = processor.video_processor.num_frames
    max_fps = processor.video_processor.max_fps
    default_sampling_fps = processor.video_processor.sampling_fps

    messages = get_message(
        images=None,
        video_path=example["video"],
        max_frames=max_frames,
        frame_sample_mode=frame_sample_mode,
        max_fps=max_fps,
        sampling_fps=sampling_fps if sampling_fps is not None else default_sampling_fps,
        input_text=prompt_text,
        style=effective_style,
    )

    images, videos_inputs, video_kwargs = process_vision_info(messages)
    multi_modal_data = {}
    if images:
        multi_modal_data["image"] = images
    if videos_inputs:
        multi_modal_data["video"] = videos_inputs

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return {
        "prompt": prompt,
        "multi_modal_data": multi_modal_data,
        "mm_processor_kwargs": video_kwargs,
    }


def collect_metadata(example):
    """Extract serializable metadata fields from a dataset example."""
    result = {"example_id": example["example_id"], "video": example["video"]}
    raw = example.get("raw")
    if raw:
        meta = raw.get("metadata", {})
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                result[k] = v
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch vLLM inference over videos")
    parser.add_argument("model_dir", help="Path to Molmo2 checkpoint")
    # Input modes (mutually exclusive)
    parser.add_argument("--video_dir", help="Directory of .mp4 files (mode 1)")
    parser.add_argument("--task", help="Dataset name from get_dataset_by_name (mode 2)")
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    # Prompt control
    parser.add_argument("--style", help="Prompt style from GENERAL_PROMPTS_V1 (e.g. video_motion_caption)")
    parser.add_argument("--prompt", help="Explicit prompt text (overrides --style)")
    # Output
    parser.add_argument("--save_dir", required=True, help="Output directory for predictions.json")
    # Processing
    parser.add_argument("--chunk_size", type=int, default=4, help="Examples per vLLM generate call")
    parser.add_argument("--max_examples", type=int, default=None, help="Cap number of examples (for debugging)")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max generation tokens")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--use_float32", action="store_true")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after inference")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed examples")
    parser.add_argument("--shard_index", type=int, default=None, help="Shard index for parallel eval")
    parser.add_argument("--num_shards", type=int, default=None, help="Total number of shards for parallel eval")
    args = parser.parse_args()

    # Validate input modes
    if bool(args.video_dir) == bool(args.task):
        parser.error("Provide exactly one of --video_dir or --task")
    if args.video_dir and not (args.style or args.prompt):
        parser.error("--video_dir mode requires --style or --prompt")
    if (args.shard_index is None) != (args.num_shards is None):
        parser.error("--shard_index and --num_shards must be used together")

    prepare_cli_environment()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Init vLLM
    model_dir = args.model_dir
    log.info(f"Loading model from {model_dir}")
    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="float32" if args.use_float32 else "bfloat16",
        limit_mm_per_prompt={"image": 6, "video": 1},
        max_num_batched_tokens=36864,
    )

    processor = AutoProcessor.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype="auto",
        device_map="auto",
        padding_side="left",
    )

    log.info("llm and processor instantiated")

    # Build examples
    dataset = None
    if args.task:
        log.info(f"Loading dataset: {args.task} split={args.split}")
        dataset = get_dataset_by_name(args.task, args.split)
        log.info(f"Dataset size: {len(dataset)}")

    examples = build_examples(args, dataset)
    log.info(f"Total examples: {len(examples)}")

    # Resume: load existing predictions and filter
    if args.num_shards is not None:
        output_path = os.path.join(args.save_dir, f"predictions_shard{args.shard_index}.json")
    else:
        output_path = os.path.join(args.save_dir, "predictions.json")
    existing_predictions = []
    done_ids = set()
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            existing_predictions = json.load(f)
        done_ids = {p["example_id"] for p in existing_predictions}
        before = len(examples)
        examples = [ex for ex in examples if ex["example_id"] not in done_ids]
        log.info(f"Resume: {before - len(examples)} already done, {len(examples)} remaining")

    if not examples:
        log.info("No examples to process. Done.")
        exit(0)

    # Process in chunks
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    all_predictions = list(existing_predictions)
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0)

    for chunk_start in range(0, len(examples), args.chunk_size):
        chunk = examples[chunk_start:chunk_start + args.chunk_size]
        chunk_idx = chunk_start // args.chunk_size + 1
        total_chunks = (len(examples) + args.chunk_size - 1) // args.chunk_size
        log.info(f"Processing chunk {chunk_idx}/{total_chunks} ({len(chunk)} examples)")

        # Build vLLM inputs
        vllm_inputs = []
        valid_examples = []
        for ex in chunk:
            try:
                inp = build_vllm_input(ex, args.style, args.prompt, processor)
                vllm_inputs.append(inp)
                valid_examples.append(ex)
            except Exception as e:
                log.error(f"Failed to build input for {ex['example_id']}: {e}")

        if not vllm_inputs:
            continue

        # Generate
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)

        # Collect results
        for inp, ex, output in zip(vllm_inputs, valid_examples, outputs):
            prediction = output.outputs[0].text
            result = collect_metadata(ex)
            result["prediction"] = prediction
            result["input"] = inp["prompt"]
            all_predictions.append(result)

        # Save incrementally
        with open(output_path, "w") as f:
            json.dump(all_predictions, f, indent=2)
        log.info(f"Saved {len(all_predictions)} predictions to {output_path}")

    log.info(f"Done. Total predictions: {len(all_predictions)}")

    # Run evaluation if requested
    if args.eval:
        if not args.task:
            log.warning("--eval requires --task, skipping evaluation")
        else:
            # Destroy vLLM's process group so torchmetrics doesn't try NCCL sync on CPU tensors
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
            from scripts.run_standalone_eval import run_eval
            metrics = run_eval(output_path, args.task, args.split)