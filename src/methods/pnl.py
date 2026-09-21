"""Unmodified causal-learn PNL with an external, larger-p decision rule."""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import time

METHOD = "PNL-causal-learn"
SOURCE_COMMIT = "0dacacf36390e3084704636bc3c36e82e35ee0cc"
SOURCE_SHA256 = "c69c5eb49814b27e43279ae7e5645e3c55897c214a653cace4b9ff9f14671752"
SOURCE_PATH = Path(__file__).resolve().parents[1] / "vendor/causal_learn/PNL.py"
SOURCE_URL = ("https://github.com/py-why/causal-learn/blob/" + SOURCE_COMMIT
              + "/causallearn/search/FCMBased/PNL/PNL.py")
EXPECTED_VERSIONS = {"torch": "2.8.0", "numpy": "2.3.5", "scipy": "1.17.0"}
_OFFICIAL_MODULE = None
_ENVIRONMENT = None


def verify_source():
    digest = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError("Official PNL source SHA256 mismatch")
    return digest


def _official():
    global _OFFICIAL_MODULE
    verify_source()
    if _OFFICIAL_MODULE is None:
        spec = importlib.util.spec_from_file_location("paper_official_pnl", SOURCE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _OFFICIAL_MODULE = module
    return _OFFICIAL_MODULE


def check_environment():
    global _ENVIRONMENT
    verify_source()
    if _ENVIRONMENT is None:
        import numpy as np
        import scipy
        import torch
        actual = dict(torch=torch.__version__.split("+")[0], numpy=np.__version__,
                      scipy=scipy.__version__)
        if actual != EXPECTED_VERSIONS:
            raise RuntimeError(f"Frozen PNL environment mismatch: {actual}")
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        _official()
        _ENVIRONMENT = dict(method=METHOD, **actual, source_sha256=SOURCE_SHA256,
                            threads=torch.get_num_threads(),
                            interop_threads=torch.get_num_interop_threads())
    return dict(_ENVIRONMENT)


def _pvalue(value):
    import numpy as np
    values = np.asarray(value)
    if values.size != 1:
        raise ValueError("Official PNL returned a non-scalar p-value")
    scalar = float(values.reshape(-1)[0])
    if not math.isfinite(scalar) or not 0 <= scalar <= 1:
        raise ValueError("Official PNL returned an invalid p-value")
    return scalar


def decision(pforward, preverse):
    """1: X→Y, 2: Y→X, 0: exact tie. This rule is outside upstream's API."""
    pforward, preverse = _pvalue(pforward), _pvalue(preverse)
    return 1 if pforward > preverse else 2 if preverse > pforward else 0


def run_pair(pair, seed):
    """Pass canonical observations unchanged to PNL().cause_or_effect().

    The supplied experiment seed is provenance only: upstream resets Torch to
    zero once per pair and continues its RNG across the two directional fits.
    No training overrides, extra preprocessing, tests or residual files.
    """
    started = time.perf_counter()
    result = dict(method=METHOD, source_commit=SOURCE_COMMIT, source_sha256=SOURCE_SHA256,
                  pforward=None, preverse=None, score_forward=None, score_reverse=None,
                  prediction=0, status="algorithm_error", error="", seed=seed,
                  supplied_pnl_seed_used_by_algorithm=False,
                  effective_torch_seed_at_pair_start=0,
                  official_returns_direction=False,
                  residual_test="upstream_scipy_stats_ttest_ind",
                  decision_rule="larger_official_pvalue_exact_ties_abstain")
    try:
        check_environment()
        import numpy as np
        data = np.asarray(pair)
        if data.ndim != 2 or data.shape[1] != 2 or len(data) < 1:
            raise ValueError("PNL requires a nonempty n-by-2 pair")
        if not np.isfinite(data).all():
            raise ValueError("PNL inputs must be finite; rows are never dropped")
        result["n"] = len(data)
        forward, reverse = _official().PNL().cause_or_effect(data[:, [0]], data[:, [1]])
        forward, reverse = _pvalue(forward), _pvalue(reverse)
        prediction = decision(forward, reverse)
        result.update(pforward=forward, preverse=reverse,
                      score_forward=forward, score_reverse=reverse,
                      prediction=prediction, status="directed" if prediction else "equal_pvalues",
                      undirected_reason="" if prediction else "equal_pvalues")
    except Exception as exc:
        status = "timeout" if isinstance(exc, TimeoutError) else "algorithm_error"
        result.update(status=status, undirected_reason=status,
                      error=f"{type(exc).__name__}: {exc}")
    result["seconds"] = time.perf_counter() - started
    return result
