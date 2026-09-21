# Third-party sources

## causal-learn PNL

The byte-identical PNL source from causal-learn commit
`0dacacf36390e3084704636bc3c36e82e35ee0cc` and its MIT license are retained in
`src/vendor/causal_learn/`. The adapter adds only result handling and an
external larger-p decision rule (exact ties abstain).

Source: https://github.com/py-why/causal-learn/blob/0dacacf36390e3084704636bc3c36e82e35ee0cc/causallearn/search/FCMBased/PNL/PNL.py

## Installed dependencies

NumPy, SciPy, scikit-learn, causal-learn, PyTorch, R, Rcpp, jsonlite and
digest are installed from their distributions and retain their respective licenses.
They are not repackaged as binary environments in this repository.

## Tübingen Cause–Effect Pairs

The two-column numerical inputs, labels and official weights used in these
experiments are attributed to the Tübingen cause–effect benchmark and the
original sources linked by its descriptions. The repository does not grant a
new license to the source datasets. Per-pair source and description URLs are
recorded in `data/tcep/pairs.csv`.

The 67 continuous–continuous pairs are our selection based on the official
data descriptions, not an official predefined subset.

Benchmark: https://webdav.tuebingen.mpg.de/cause-effect/

Reference: J. M. Mooij, J. Peters, D. Janzing, J. Zscheischler and B. Schölkopf.
Distinguishing cause from effect using observational data: methods and benchmarks.
Journal of Machine Learning Research 17(32):1–102, 2016.
