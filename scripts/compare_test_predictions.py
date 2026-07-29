"""Compare a short test predictions.json against a reference predictions.json.

Matches entries by example_id and reports, per entry: whether the stored
`input` prompt is byte-identical, whether the `prediction` string is
byte-identical, and the hota_before/hota_after deltas where present. Used to
sanity-check that the HF-backed datasets feed vllm the exact same prompts as
the local-json datasets did.

Usage:
    python scripts/compare_test_predictions.py --new PATH --ref PATH
"""
import argparse
import json


def load_by_eid(path):
    with open(path) as f:
        preds = json.load(f)
    return {p["example_id"]: p for p in preds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", required=True)
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()

    new = load_by_eid(args.new)
    ref = load_by_eid(args.ref)

    n_matched = n_input_eq = n_pred_eq = 0
    missing = []
    for eid, np_ in new.items():
        rp = ref.get(eid)
        if rp is None:
            missing.append(eid)
            continue
        n_matched += 1
        input_eq = np_.get("input") == rp.get("input")
        pred_eq = np_.get("prediction") == rp.get("prediction")
        n_input_eq += input_eq
        n_pred_eq += pred_eq
        line = f"{eid}: input {'==' if input_eq else 'DIFFERS'}, prediction {'==' if pred_eq else 'DIFFERS'}"
        for k in ("hota_before", "hota_after"):
            a, b = np_.get(k), rp.get(k)
            if a is not None and b is not None:
                line += f", {k} {a:.4f} vs {b:.4f} (d={a - b:+.4f})"
        print(line)
        if not input_eq:
            ni, ri = np_.get("input", ""), rp.get("input", "")
            for i, (ca, cb) in enumerate(zip(ni, ri)):
                if ca != cb:
                    print(f"    first input diff at char {i}: new ...{ni[max(0, i - 60):i + 60]!r}...")
                    print(f"                                  ref ...{ri[max(0, i - 60):i + 60]!r}...")
                    break
            else:
                print(f"    input lengths differ: new {len(ni)} vs ref {len(ri)}")

    print(f"\nmatched {n_matched}/{len(new)} example_ids "
          f"({len(missing)} missing from ref: {missing[:5]}{'...' if len(missing) > 5 else ''})")
    print(f"inputs identical:      {n_input_eq}/{n_matched}")
    print(f"predictions identical: {n_pred_eq}/{n_matched}")


if __name__ == "__main__":
    main()
