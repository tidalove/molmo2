"""Round-trip verification: HF-backed CFC classes vs. the local-json originals.

For sampled examples per config, asserts that the new cfc_hf_* classes produce
the same runtime payloads as the existing classes, modulo the intended 2 fps
subsampling (old annotations restricted to frame % 3 == 0 must equal the new
ones exactly), plus optional hub checks against the pushed repo.

Requires the local annotation cache (run build_cfc_hf_dataset.py --local-cache
or a prior download()) and the original source jsons on disk.

Usage:
    python scripts/verify_cfc_hf_dataset.py [--configs cfc_track ...]
        [--split validation-v2] [--n-samples 5] [--hub] [--repo-id ID]
"""
import argparse
import logging
import os
import sys
from os.path import abspath, dirname

os.environ.setdefault("MOLMO_DATA_DIR", "data")
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import numpy as np

from olmo.data import get_dataset as get_dataset_module
from olmo.data.get_dataset import get_dataset_by_name

log = logging.getLogger("verify_cfc_hf_dataset")

KEEP_EVERY = 3

# new dataset name -> old dataset name (get_dataset registrations)
PAIRS = {
    "cfc_hf_track_eval_2fps": "cfc_track_eval_2fps",
    "cfc_hf_target_track_eval_2fps": "cfc_target_track_eval_2fps",
    "cfc_hf_synthetic_correction_full_eval_2fps": "cfc_synthetic_correction_full_eval_2fps",
    "cfc_hf_synthetic_correction_vague_eval_2fps": "cfc_synthetic_correction_vague_eval_2fps",
    "cfc_hf_synthetic_correction_wrong_only_eval_2fps": "cfc_synthetic_correction_wrong_only_eval_2fps",
    "cfc_hf_synthetic_correction_no_info_eval_2fps": "cfc_synthetic_correction_no_info_eval_2fps",
    "cfc_hf_synthetic_correction_incomplete_eval_2fps": "cfc_synthetic_correction_incomplete_eval_2fps",
    "cfc_hf_correction_real_full_eval_2fps": "cfc_correction_real_full_eval_2fps",
    "cfc_hf_correction_real_wrong_only_eval_2fps": "cfc_correction_real_wrong_only_eval_2fps",
    "cfc_hf_correction_real_vague_eval_2fps": "cfc_correction_real_vague_eval_2fps",
    "cfc_hf_correction_real_no_info_eval_2fps": "cfc_correction_real_no_info_eval_2fps",
    "cfc_hf_text_eval_2fps": "cfc_text_eval_2fps",
}

# validation-only exceptions: real wrong-only has no val jsonl
TRAIN_ONLY = {"cfc_hf_correction_real_wrong_only_eval_2fps"}


def subsampled(frames):
    """Restrict a list of {frame,...} dicts to the 2 fps kept frames."""
    return [f for f in frames if f["frame"] % KEEP_EVERY == 0]


def points_equal(old_points, new_points, where):
    """Compare per-frame point structures (dict- or list-keyed points)."""
    assert len(old_points) == len(new_points), \
        f"{where}: {len(old_points)} vs {len(new_points)} frames"
    for of, nf in zip(old_points, new_points):
        assert of["frame"] == nf["frame"], f"{where}: frame {of['frame']} vs {nf['frame']}"
        assert abs(of["time"] - nf["time"]) < 1e-9, f"{where}: time @{of['frame']}"
        op, np_ = of["points"], nf["points"]
        okeys = sorted(int(k) for k in op)
        nkeys = sorted(int(k) for k in np_)
        assert okeys == nkeys, f"{where}: slots @{of['frame']}: {okeys} vs {nkeys}"
        for k in okeys:
            o = op[k] if k in op else op[str(k)]
            n = np_[k] if k in np_ else np_[str(k)]
            assert o["point"] == n["point"], \
                f"{where}: point @{of['frame']} slot {k}: {o['point']} vs {n['point']}"
            assert bool(o["occluded"]) == bool(n["occluded"]), \
                f"{where}: occluded @{of['frame']} slot {k}"


def masks_equal(old_masks, new_masks, keep_every, where):
    """old: dense per-native-frame local file; new: rehydrated (None-padded).
    Compare decoded masks at kept frames; old non-kept frames are dropped."""
    from pycocotools import mask as mask_utils
    assert sorted(old_masks) == sorted(new_masks), \
        f"{where}: mask slots {sorted(old_masks)} vs {sorted(new_masks)}"
    for slot in old_masks:
        old_frames, new_frames = old_masks[slot], new_masks[slot]
        for fidx in range(0, len(old_frames), keep_every):
            o, n = old_frames[fidx], new_frames[fidx]
            assert (o is None) == (n is None), f"{where}: slot {slot} @{fidx} null mismatch"
            if o is not None:
                od = mask_utils.decode({"size": o["size"], "counts": o["counts"]})
                nd = mask_utils.decode({"size": n["size"], "counts": n["counts"]})
                assert np.array_equal(od, nd), f"{where}: slot {slot} @{fidx} mask differs"


def verify_pair(new_name, old_name, split, n_samples, rng):
    new_ds = get_dataset_by_name(new_name, split=split)
    old_ds = get_dataset_by_name(old_name, split=split)

    n_checked = 0
    idxs = rng.permutation(len(new_ds))[:n_samples]
    for idx in idxs:
        item_new = new_ds.get(int(idx), rng)
        ex_id = item_new["metadata"]["example_id"]
        old_idx = old_ds.data_lookup.get(ex_id)
        assert old_idx is not None, f"{ex_id} missing from {old_name}"
        item_old = old_ds.get(old_idx, rng)
        where = f"{new_name}:{ex_id}"

        # scalar metadata
        for key in ("w", "h", "video_fps", "expression"):
            assert item_old["metadata"][key] == item_new["metadata"][key], \
                f"{where}: metadata[{key}]"
        assert item_old["metadata"].get("mask_id") == item_new["metadata"].get("mask_id"), \
            f"{where}: mask_id"

        if "multi_turn_messages" in item_new:  # correction family
            ml_old, ml_new = item_old["multi_turn_messages"], item_new["multi_turn_messages"]
            assert len(ml_old) == len(ml_new), f"{where}: n turns"
            for t, (mo, mn) in enumerate(zip(ml_old, ml_new)):
                assert mo["question"] == mn["question"], \
                    f"{where}: turn {t} prompt:\n  old: {mo['question']}\n  new: {mn['question']}"
                points_equal(subsampled(mo["points"]), mn["points"], f"{where} turn {t}")
        else:  # track family
            mo, mn = item_old["message_list"][0], item_new["message_list"][0]
            assert mo["label"] == mn["label"], f"{where}: label"
            points_equal(subsampled(mo["points"]), mn["points"], where)

        old_masks = item_old["metadata"].get("masks")
        new_masks = item_new["metadata"].get("masks")
        if old_masks or new_masks:
            keep = item_new["metadata"]["video_fps"] // 2
            masks_equal(old_masks, new_masks, KEEP_EVERY, where)
            assert keep == KEEP_EVERY
        n_checked += 1

    assert len(new_ds) == len(old_ds), \
        f"{new_name}: {len(new_ds)} rows vs {old_name}: {len(old_ds)}"
    log.info(f"OK {new_name} vs {old_name} [{split}]: "
             f"{n_checked} examples checked, {len(new_ds)} rows total")


def verify_hub(repo_id, configs):
    """Smoke test: the pushed repo loads and row counts match the local cache."""
    import datasets as hf_datasets
    from olmo.data.cfc_hf_datasets import CFC_HF_CLASSES

    by_config = {c.HF_CONFIG: c for c in CFC_HF_CLASSES}
    for config in configs:
        cls = by_config[config]
        for data_split in sorted(set(cls.SPLIT_MAP.values())):
            ds = hf_datasets.load_dataset(repo_id, config, split=data_split)
            local = cls._load_hf_rows(data_split)
            assert len(ds) == len(local), \
                f"{config}/{data_split}: hub {len(ds)} vs local {len(local)}"
            assert ds.features == local.features, f"{config}/{data_split}: features differ"
            log.info(f"OK hub {config}/{data_split}: {len(ds)} rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Subset of NEW dataset names (default: all pairs)")
    parser.add_argument("--split", default="validation-v2")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hub", action="store_true", help="Also check the pushed repo")
    parser.add_argument("--repo-id", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    rng = np.random.RandomState(args.seed)

    names = args.configs or list(PAIRS)
    for new_name in names:
        split = "train-v2" if new_name in TRAIN_ONLY and args.split != "train-v2" else args.split
        verify_pair(new_name, PAIRS[new_name], split, args.n_samples, rng)

    if args.hub:
        from olmo.data.cfc_hf_datasets import DEFAULT_CFC_HF_REPO, CFC_HF_CLASSES
        repo_id = args.repo_id or DEFAULT_CFC_HF_REPO
        configs = [c.HF_CONFIG for c in CFC_HF_CLASSES
                   if args.configs is None or any(c.HF_CONFIG in n for n in args.configs)]
        verify_hub(repo_id, configs)

    log.info("All verifications passed.")


if __name__ == "__main__":
    main()
