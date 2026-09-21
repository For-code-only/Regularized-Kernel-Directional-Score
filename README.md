# RKDS Experiments

Code for the ANM, TCEP, non-ANM, and sensitivity experiments, including RKDS and the comparison methods.

## Setup

The installer targets Ubuntu/Linux with Python 3.12 and installs the required Python and R dependencies. No GPU is required. Run these commands from this directory:

```bash
bash scripts/setup.sh
.venv/bin/python run.py setup
```

## Run

Start with a small smoke test:

```bash
.venv/bin/python run.py smoke --workers 1
```

Run an experiment suite:

```bash
.venv/bin/python run.py run --suite anm --workers 1
```

Available suites: `anm`, `tcep`, `nonanm`, `sensitivity`, or `all`. Increase `--workers` to use more parallel workers. Full experiments can be computationally expensive.

`src/` contains the implementations, `scripts/` the shell launchers, `config/` the settings and seeds, and `data/` the TCEP data. Compressed case lists and data are read directly; no manual extraction is needed. Results are saved in `results/`. TCEP includes cap1000 and full; sensitivity uses n=1000 only.

PNL uses unchanged [causal-learn source](https://github.com/py-why/causal-learn/blob/0dacacf36390e3084704636bc3c36e82e35ee0cc/causallearn/search/FCMBased/PNL/PNL.py). Our wrapper chooses the larger returned p-value and abstains on exact ties, without a significance threshold or additional HSIC. Upstream returns t-test p-values, not an independence test or a direction.

To summarize saved results, run `python run.py analyze --suite all`; add `--decision-rule score` for the supplementary direct-score comparison. Use `.venv/bin/python` when using the installed environment. Other options: `python run.py --help`.

Third-party sources and licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
