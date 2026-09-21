#!/usr/bin/env bash
# Linux detached launcher. Status/export never install dependencies.
set -Eeuo pipefail
RKDS_ACTION=status
if [[ $# -gt 0 ]]; then RKDS_ACTION="$1"; shift; fi
RKDS_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
RKDS_SUITE=all
RKDS_METHODS=all
RKDS_OPTIONS=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite) RKDS_SUITE="$2"; shift 2;;
    --methods) RKDS_METHODS="$2"; shift 2;;
    --workers|--timeout|--rscript|--reserve-gib|--memory-budget-gib|--decision-rule) shift 2;;
    --partial) shift;;
    --inputs) shift; while [[ $# -gt 0 && "$1" != --* ]]; do shift; done;;
    *) echo "Unknown option: $1"; exit 2;;
  esac
done
case "$RKDS_SUITE" in anm|tcep|sensitivity|nonanm|all|smoke) ;; *) echo 'Unknown suite'; exit 2;; esac
case "$RKDS_METHODS" in all|core|pnl) ;; *) echo 'Unknown method selection'; exit 2;; esac
case "$RKDS_ACTION" in setup|start|resume|retry|run|smoke|status|stop|export|restore|preflight|analyze|verify|logs) ;; *)
  echo 'Usage: bash scripts/start.sh start|resume|retry|status|stop|export [--suite anm|tcep|sensitivity|nonanm|all] [--methods all|core|pnl] [--workers N]'; exit 2;; esac
RKDS_PYTHON=python3
if [[ -x "$RKDS_ROOT/.venv/bin/python" ]]; then RKDS_PYTHON="$RKDS_ROOT/.venv/bin/python"; fi
RKDS_SERVICE="rkds-paper-$RKDS_SUITE-$RKDS_METHODS"
RKDS_LOG="$RKDS_ROOT/logs/launcher_${RKDS_SUITE}_${RKDS_METHODS}.log"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

rkds_command() {
  local rkds_subcommand="$1"; shift
  "$RKDS_PYTHON" -u "$RKDS_ROOT/run.py" "$rkds_subcommand" ${RKDS_OPTIONS[@]+"${RKDS_OPTIONS[@]}"} "$@"
}
rkds_start() {
  [[ "$(uname -s)" == Linux ]] || { echo 'Detached launcher requires Linux; use run.py directly for local validation.'; exit 2; }
  # Probe exactly the selected suite locks before setup, including overlapping all runs.
  if ! "$RKDS_PYTHON" - "$RKDS_ROOT" "$RKDS_SUITE" "$RKDS_METHODS" <<'PY'
from pathlib import Path
from contextlib import ExitStack
import fcntl,sys
root=Path(sys.argv[1]);suite,methods=sys.argv[2:]
suites=['anm','tcep','sensitivity','nonanm'] if suite in ['all','smoke'] else [suite]
with ExitStack() as stack:
    for name in sorted(suites):
        if name=='sensitivity' and methods=='pnl':continue
        path=root/('verification/smoke_run' if suite=='smoke' else 'results')/name/'run.lock'
        path.parent.mkdir(parents=True,exist_ok=True)
        handle=stack.enter_context(path.open('a+'))
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit(1)
PY
  then echo 'A selected suite is already running.'; rkds_command status; return; fi
  if [[ ! -x "$RKDS_ROOT/.venv/bin/python" ]]; then bash "$RKDS_ROOT/scripts/setup.sh"; fi
  RKDS_PYTHON="$RKDS_ROOT/.venv/bin/python"
  rkds_command verify >/dev/null
  rkds_command setup
  rkds_command preflight
  mkdir -p "$RKDS_ROOT/logs"
  local rkds_mode=run
  if [[ "$RKDS_ACTION" == retry ]]; then rkds_mode=retry; fi
  if [[ "$RKDS_SUITE" == smoke ]]; then rkds_mode=smoke; fi
  if [[ -d /run/systemd/system ]] && command -v systemd-run >/dev/null; then
    local rkds_sudo=''
    if [[ "$(id -u)" != 0 ]]; then rkds_sudo=sudo; fi
    if systemctl is-active --quiet "$RKDS_SERVICE"; then echo 'This launcher service is already active.'; return; fi
    $rkds_sudo systemctl reset-failed "$RKDS_SERVICE" >/dev/null 2>&1 || true
    $rkds_sudo systemd-run --unit="$RKDS_SERVICE" --collect \
      --property="User=$(id -un)" --property="Group=$(id -gn)" \
      --property="WorkingDirectory=$RKDS_ROOT" --property=KillMode=mixed \
      --property=TimeoutStopSec=infinity --property="StandardOutput=append:$RKDS_LOG" \
      --property=StandardError=inherit \
      "$RKDS_PYTHON" -u "$RKDS_ROOT/run.py" "$rkds_mode" ${RKDS_OPTIONS[@]+"${RKDS_OPTIONS[@]}"}
  else
    "$RKDS_PYTHON" - "$RKDS_ROOT" "$RKDS_LOG" "$RKDS_PYTHON" "$rkds_mode" ${RKDS_OPTIONS[@]+"${RKDS_OPTIONS[@]}"} <<'PY'
from pathlib import Path
import subprocess,sys
root,log,python,mode=sys.argv[1:5]
with open(log,'ab') as stream:
    process=subprocess.Popen([python,'-u',str(Path(root)/'run.py'),mode,*sys.argv[5:]],cwd=root,
        stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=True)
print('Background PID:',process.pid)
PY
  fi
  echo "Started. SSH may be closed. Log: $RKDS_LOG"
}
case "$RKDS_ACTION" in
  setup) bash "$RKDS_ROOT/scripts/setup.sh"; RKDS_PYTHON="$RKDS_ROOT/.venv/bin/python"; rkds_command setup; rkds_command preflight;;
  start|resume|retry) rkds_start;;
  logs) tail -n 60 -F "$RKDS_LOG";;
  *) rkds_command "$RKDS_ACTION";;
esac
