"""Compare a golden capture against a later session.

Exit 1 if any shared arm disagrees on token ids, or if decode tok/s drops
below 95% of the baseline arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _key(arm: dict) -> tuple:
    return (arm["backend"], bool(arm["cuda_graph"]), int(arm["num_speculative_tokens"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--min-ratio", type=float, default=0.95)
    args = parser.parse_args()
    base = json.loads(args.baseline.read_text())
    cur = json.loads(args.current.read_text())
    by_cur = {_key(a): a for a in cur["arms"]}
    failed = 0
    for arm in base["arms"]:
        k = _key(arm)
        other = by_cur.get(k)
        label = f"{k[0]} graph={k[1]} spec={k[2]}"
        if other is None:
            print(f"MISSING {label}")
            failed += 1
            continue
        if arm["token_ids"] != other["token_ids"]:
            print(f"TOKENS {label}: baseline {arm['token_ids'][:8]}... vs {other['token_ids'][:8]}...")
            failed += 1
            continue
        ratio = other["decode_tok_s"] / arm["decode_tok_s"] if arm["decode_tok_s"] else 0.0
        if ratio < args.min_ratio:
            print(
                f"SLOW {label}: {other['decode_tok_s']} tok/s "
                f"({ratio:.3f} of {arm['decode_tok_s']})"
            )
            failed += 1
        else:
            print(f"OK {label}: tokens match, {other['decode_tok_s']} tok/s ({ratio:.3f})")
    if cur.get("errors"):
        print(f"current capture recorded {len(cur['errors'])} arm errors")
        failed += len(cur["errors"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
