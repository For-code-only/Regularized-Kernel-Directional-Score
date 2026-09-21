#!/usr/bin/env python3
"""Run the unmodified causal-learn ANM with a declared decision wrapper.

Input: a CSV with exactly two numeric columns and a header (x, y).
Output: one CSV record. No causal labels are accepted or read.
The package returns two residual-independence p values. This wrapper returns
1 (x -> y) or 2 (y -> x) only when exactly that direction is not rejected
at alpha = 0.05; it returns 0 when both directions agree on rejection.
"""

import argparse
import csv
from importlib import metadata
from pathlib import Path
import random
import sys
import time
import warnings


VERSION = "0.1.4.8"
ALPHA = 0.05
DEFAULT_SEED = 1700909001
FIELDS = [
    "method", "n", "seed", "alpha", "pforward", "preverse", "prediction",
    "status", "packageversion", "seconds", "warnings", "error",
]


def load_pair(path, no_header, np):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        if not no_header:
            header = next(reader, None)
            if header is None or len(header) != 2:
                raise ValueError("Input must have exactly two columns.")
        rows = list(reader)
    if len(rows) < 3 or any(len(row) != 2 for row in rows):
        raise ValueError("Input must contain at least three two-column rows.")
    values = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Input values must be finite; rows are not dropped.")
    scales = values.std(axis=0, ddof=1)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Both variables must have positive finite variance.")
    return (values - values.mean(axis=0)) / scales


def decision(pforward, preverse):
    forward_compatible = pforward > ALPHA
    reverse_compatible = preverse > ALPHA
    if forward_compatible and not reverse_compatible:
        return 1, "directed"
    if reverse_compatible and not forward_compatible:
        return 2, "directed"
    if forward_compatible:
        return 0, "both_compatible"
    return 0, "both_rejected"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-header", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 2**32 - 1.")
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files.")

    result = dict.fromkeys(FIELDS, "")
    result.update(method="ANM-GP", seed=args.seed, alpha=ALPHA,
                  prediction=0, status="implementation_failure")
    started = time.perf_counter()
    success = False
    caught = []
    try:
        import numpy as np
        from causallearn.search.FCMBased.ANM.ANM import ANM

        result["packageversion"] = metadata.version("causal-learn")
        if result["packageversion"] != VERSION:
            raise RuntimeError(f"Expected causal-learn {VERSION}, found "
                               f"{result['packageversion']}.")
        random.seed(args.seed)
        np.random.seed(args.seed)
        pair = load_pair(args.input, args.no_header, np)
        result["n"] = len(pair)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            forward, reverse = ANM().cause_or_effect(pair[:, [0]], pair[:, [1]])
        pforward, preverse = float(forward), float(reverse)
        if not (np.isfinite([pforward, preverse]).all()
                and 0 <= pforward <= 1 and 0 <= preverse <= 1):
            raise RuntimeError("ANM returned invalid p values.")
        prediction, status = decision(pforward, preverse)
        result.update(pforward=pforward, preverse=preverse,
                      prediction=prediction, status=status)
        success = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(result["error"], file=sys.stderr)
    finally:
        result["seconds"] = time.perf_counter() - started
        result["warnings"] = " | ".join(dict.fromkeys(
            f"{item.category.__name__}: {item.message}" for item in caught))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerow(result)
        temporary.replace(args.output)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
