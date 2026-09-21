#!/usr/bin/env python3
"""ANM with sklearn RBF kernel ridge regression and the frozen ANM-GP KCI test.

Input is a CSV containing exactly two numeric columns and a header.  The
regressor is the only scientific substitution relative to the actually-run
causal-learn ANM-GP wrapper.  Residual testing and the directional decision
rule remain the causal-learn 0.1.4.8 KCI_UInd defaults and the declared
alpha=0.05 compatibility rule, respectively.
"""
from __future__ import annotations

import argparse
import csv
from importlib import metadata
import json
from pathlib import Path
import random
import sys
import time
import warnings


METHOD = "ANM_KRR"
CAUSAL_LEARN_VERSION = "0.1.4.8"
FIELDS = [
    "method", "n", "seed", "alpha", "pforward", "preverse",
    "prediction", "status", "undirected_reason",
    "regression_implementation", "regression_kernel", "regression_alpha",
    "regression_bandwidth_rule", "regression_gamma_rule",
    "bandwidth_forward", "bandwidth_reverse",
    "gamma_forward", "gamma_reverse", "hyperparameter_selection",
    "residual_test", "test_kernel_x", "test_kernel_residual",
    "test_bandwidth", "pvalue_calibration",
    "test_stat_forward", "test_stat_reverse",
    "causal_learn_version", "numpy_version", "scipy_version",
    "scikit_learn_version", "seconds", "warnings", "error",
]


def load_settings(path: Path) -> dict:
    settings = json.loads(path.read_text(encoding="utf-8"))
    regression = settings.get("regression", {})
    test = settings.get("residual_independence_test", {})
    decision_settings = settings.get("decision", {})
    forbidden = {"cross_validation", "cv_design_policy", "normalized_lambda_grid",
                 "matrix_penalty", "tie_rule", "regularization_objective"}
    if (settings.get("schema") != 2 or settings.get("method") != METHOD or
            regression.get("implementation") != "sklearn.kernel_ridge.KernelRidge" or
            regression.get("kernel") != "rbf" or
            regression.get("bandwidth_rule") !=
            "sigma is the median of strictly positive pairwise Euclidean distances in the candidate-cause sample" or
            regression.get("alpha") != 1.0 or
            regression.get("gamma_rule") != "1 / (2 * sigma^2)" or
            regression.get("hyperparameter_selection") != "none" or
            forbidden.intersection(regression) or
            test.get("implementation") != "causal-learn 0.1.4.8 KCI_UInd" or
            test.get("kernelX") != "Gaussian" or test.get("kernelY") != "Gaussian" or
            test.get("est_width") != "empirical" or test.get("approx") is not True or
            decision_settings.get("alpha") != 0.05 or
            decision_settings.get("not_rejected") != "p > alpha"):
        raise ValueError("Unexpected or modified ANM-KRR configuration.")
    return settings


def load_pair(path: Path, no_header: bool, np):
    # This is intentionally the same validation and sample-standardization
    # logic as anm_gp.py in the actually-run ANM package.
    with path.open(newline="", encoding="utf-8-sig") as stream:
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


def decision(pforward: float, preverse: float, alpha: float):
    forward_compatible = pforward > alpha
    reverse_compatible = preverse > alpha
    if forward_compatible and not reverse_compatible:
        return 1, "directed", ""
    if reverse_compatible and not forward_compatible:
        return 2, "directed", ""
    if forward_compatible:
        return 0, "both_compatible", "both_directions_not_rejected"
    return 0, "both_rejected", "both_directions_rejected"


def median_bandwidth(x, np, pdist) -> float:
    distances = pdist(np.asarray(x, dtype=np.float64).reshape(-1, 1),
                      metric="euclidean")
    positive = distances[np.isfinite(distances) & (distances > 0)]
    if not len(positive):
        raise ValueError("Median heuristic is undefined without a positive pairwise distance.")
    bandwidth = float(np.median(positive))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise RuntimeError("Median heuristic produced an invalid bandwidth.")
    return bandwidth


def fit_krr(x, y, alpha, np, pdist, KernelRidge):
    """Fit sklearn's official RBF KernelRidge with a median-heuristic gamma."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    bandwidth = median_bandwidth(x, np, pdist)
    gamma = 1.0 / (2.0 * bandwidth ** 2)
    if not np.isfinite(gamma) or gamma <= 0:
        raise RuntimeError("Median heuristic produced an invalid RBF gamma.")
    model = KernelRidge(alpha=float(alpha), kernel="rbf", gamma=gamma)
    prediction = np.asarray(model.fit(x, y).predict(x), dtype=np.float64).reshape(-1)
    residual = y - prediction
    if not np.isfinite(residual).all():
        raise RuntimeError("KRR residuals are nonfinite.")
    return {
        "residual": residual.reshape(-1, 1),
        "bandwidth": bandwidth,
        "gamma": gamma,
        "alpha": float(alpha),
    }


def run_pair(pair, seed: int, settings: dict, np, pdist, KernelRidge, KCI_UInd):
    regression = settings["regression"]
    alpha = float(settings["decision"]["alpha"])
    regression_alpha = float(regression["alpha"])
    random.seed(seed)
    np.random.seed(seed)
    forward = fit_krr(pair[:, 0], pair[:, 1], regression_alpha, np, pdist, KernelRidge)
    reverse = fit_krr(pair[:, 1], pair[:, 0], regression_alpha, np, pdist, KernelRidge)
    # Explicit arguments freeze the actual ANM-GP defaults.  In particular,
    # the residual test keeps empirical widths and gamma p-value calibration;
    # the regression median heuristic does not leak into this test.
    independence = KCI_UInd(kernelX="Gaussian", kernelY="Gaussian",
                            est_width="empirical", approx=True)
    pforward, stat_forward = independence.compute_pvalue(
        pair[:, [0]], forward["residual"])
    preverse, stat_reverse = independence.compute_pvalue(
        pair[:, [1]], reverse["residual"])
    pforward, preverse = float(pforward), float(preverse)
    stat_forward, stat_reverse = float(stat_forward), float(stat_reverse)
    values = (pforward, preverse, stat_forward, stat_reverse)
    if (not np.isfinite(values).all() or not 0 <= pforward <= 1 or
            not 0 <= preverse <= 1):
        raise RuntimeError("Residual independence test returned invalid values.")
    prediction, status, undirected_reason = decision(pforward, preverse, alpha)
    return {
        "alpha": alpha,
        "pforward": pforward,
        "preverse": preverse,
        "prediction": prediction,
        "status": status,
        "undirected_reason": undirected_reason,
        "regression_alpha": regression_alpha,
        "bandwidth_forward": forward["bandwidth"],
        "bandwidth_reverse": reverse["bandwidth"],
        "gamma_forward": forward["gamma"],
        "gamma_reverse": reverse["gamma"],
        "test_stat_forward": stat_forward,
        "test_stat_reverse": stat_reverse,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "config" / "krr_settings.json")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--no-header", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 2**32 - 1.")
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files.")

    result = dict.fromkeys(FIELDS, "")
    result.update(method=METHOD, seed=args.seed, prediction=0,
                  status="implementation_failure", undirected_reason="algorithm_error")
    started = time.perf_counter()
    caught = []
    success = False
    try:
        import numpy as np
        from scipy.spatial.distance import pdist
        from sklearn.kernel_ridge import KernelRidge
        from causallearn.utils.KCI.KCI import KCI_UInd

        versions = {
            "causal_learn_version": metadata.version("causal-learn"),
            "numpy_version": metadata.version("numpy"),
            "scipy_version": metadata.version("scipy"),
            "scikit_learn_version": metadata.version("scikit-learn"),
        }
        result.update(versions)
        if versions["causal_learn_version"] != CAUSAL_LEARN_VERSION:
            raise RuntimeError(f"Expected causal-learn {CAUSAL_LEARN_VERSION}, found "
                               f"{versions['causal_learn_version']}.")
        settings = load_settings(args.config.resolve())
        regression = settings["regression"]
        test = settings["residual_independence_test"]
        result.update(
            alpha=settings["decision"]["alpha"],
            regression_implementation=regression["implementation"],
            regression_kernel=regression["kernel"],
            regression_alpha=regression["alpha"],
            regression_bandwidth_rule=regression["bandwidth_rule"],
            regression_gamma_rule=regression["gamma_rule"],
            hyperparameter_selection=regression["hyperparameter_selection"],
            residual_test=test["implementation"],
            test_kernel_x=test["kernelX"],
            test_kernel_residual=test["kernelY"],
            test_bandwidth=test["est_width"],
            pvalue_calibration=test["pvalue_calibration"],
        )
        pair = load_pair(args.input, args.no_header, np)
        result["n"] = len(pair)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result.update(run_pair(pair, args.seed, settings, np, pdist,
                                   KernelRidge, KCI_UInd))
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
