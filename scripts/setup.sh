#!/usr/bin/env bash
set -Eeuo pipefail
RKDS_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
[[ "$(uname -s)" == Linux ]] || { echo 'Install on Linux (Ubuntu 24.04 recommended).'; exit 2; }
RKDS_SUDO=()
if [[ "$(id -u)" != 0 ]]; then RKDS_SUDO=(sudo); fi
if ! command -v Rscript >/dev/null || ! command -v g++ >/dev/null ||
   ! Rscript -e 'stopifnot(all(vapply(c("Rcpp","jsonlite","digest"),requireNamespace,logical(1),quietly=TRUE)))' >/dev/null 2>&1; then
  "${RKDS_SUDO[@]}" apt-get update
  "${RKDS_SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    r-base r-cran-rcpp r-cran-jsonlite r-cran-digest build-essential \
    libopenblas0-pthread python3-venv python3-pip
fi
python3 -c 'import sys; assert (3,11)<=sys.version_info[:2]<(3,14), "Use Python 3.11–3.13; Python 3.12 recommended"'
if [[ ! -x "$RKDS_ROOT/.venv/bin/python" ]]; then
  if ! python3 -m venv "$RKDS_ROOT/.venv"; then
    "${RKDS_SUDO[@]}" apt-get update
    "${RKDS_SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv
    python3 -m venv "$RKDS_ROOT/.venv"
  fi
fi
"$RKDS_ROOT/.venv/bin/python" -m pip install -r "$RKDS_ROOT/requirements.txt"
"$RKDS_ROOT/.venv/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0'
echo 'Dependencies installed. No experiment has been started.'
