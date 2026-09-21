#!/usr/bin/env python3
"""Streaming evaluation using method decisions or optional direct score comparison.

No model is fitted here. All formal denominators come from the frozen manifest.
Missing records require --allow-partial; incomplete groups never receive formal
A/C/CA values. All rates in machine-readable files are fractions, not percentages.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("anm", "tcep", "sensitivity", "nonanm")
METHODS = (("RKDS", "H13"), ("ANM_GP", "default"), ("ANM_KRR", "default"),
           ("RECI_MON3", "default"), ("KCDC", "D1"), ("PNL-causal-learn", "default"))
VARIANTS = ("H13", "H13_theta_1e-08", "H13_theta_1e-07", "H13_theta_1e-05",
            "H13_theta_1e-04", "H13_theta_1e-03", "H13_sigma1_1e-01",
            "H13_sigma1_4e-01", "H13_sigma2_1e-01", "H13_sigma2_4e-01",
            "H13_sigma3_8e-01", "H13_sigma3_3.2e+00")
MODELS = ("ANM_control", "heteroscedastic", "post_nonlinear", "multiplicative")
HASH = re.compile(r"^[0-9a-f]{64}$")
GROUP_FIELDS = ("suite", "scope", "n", "branch", "level", "cap", "seed", "subset",
                "weighting", "method", "configuration")
RESULT_FIELDS = ("method", "configuration", "direction_encoding", "truth", "prediction_native",
                 "prediction_evaluated", "decision_rule",
                 "presented_prediction_native", "score_forward", "score_reverse", "p_forward",
                 "p_reverse", "presented_score_forward", "presented_score_reverse", "success",
                 "status", "error", "undirected_reason", "data_hash", "swapped", "n_used",
                 "seconds", "timing_scope", "protocol_hash", "diagnostic_coordinate_system",
                 "denominator_forward", "denominator_reverse", "condition_number_forward",
                 "condition_number_reverse", "screen_p", "source_data_hash", "tcep_reference_data_hash",
                 "tcep_reference_hash_match", "tcep_reference_hash_policy", "details", "checkpoint")
TCEP_HASH_POLICY = "audit_only_cross_runtime_csv_bytes"


class ValidationError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise ValidationError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_constant(value):
    raise ValidationError(f"Nonfinite JSON constant: {value}")


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Invalid JSON {path}: {exc}") from exc


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def equal(left, right):
    if str(left) == str(right):
        return True
    try:
        a, b = Decimal(str(left)), Decimal(str(right))
        return a.is_finite() and b.is_finite() and a == b
    except InvalidOperation:
        return False


def expected_methods(suite):
    return tuple(("RKDS", name) for name in VARIANTS) if suite == "sensitivity" else METHODS


def tcep_source_hashes(root, manifest, frozen):
    """The raw source CSV remains byte-strict; only historical standardized bytes may differ."""
    relative = "data/tcep/pairs.csv"
    pair_path = root / relative
    need(relative in frozen and sha256(pair_path) == frozen[relative], "TCEP pair metadata is not frozen")
    with pair_path.open(encoding="utf-8-sig", newline="") as handle:
        pairs = list(csv.DictReader(handle))
    by_id = {int(row["pair_id"]): row for row in pairs}
    need(len(by_id) == len(pairs), "Duplicate TCEP source pair metadata")
    archive_relative = "data/tcep/observations.zip"
    archive_path = root / archive_relative
    need(archive_relative in frozen and sha256(archive_path) == frozen[archive_relative],
         "TCEP observations archive is not frozen")
    hashes = {}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        need(len(names) == len(set(names)), "Duplicate TCEP archive members")
        need(set(names) == {f"pair{pair_id:04d}.csv" for pair_id in by_id}, "TCEP archive inventory mismatch")
        for row in manifest.values():
            relative = row["data_file"]
            if relative in hashes:
                continue
            member = f"pair{int(row['pair_id']):04d}.csv"
            need(relative == "data/tcep/cache/" + member, "TCEP source path/pair mismatch")
            expected = by_id[int(row["pair_id"])]["cache_sha256"]
            need(HASH.fullmatch(expected), f"Invalid TCEP source hash: {relative}")
            need(hashlib.sha256(archive.read(member)).hexdigest() == expected,
                 f"TCEP raw source CSV changed: {relative}")
            hashes[relative] = expected
    return hashes


def validate_tcep_provenance(meta, entry, case, source_hash):
    """Return an auditable record; do not require R/platform CSV serialization equality."""
    where = case["case_id"]
    need(entry.get("case_id") == where and equal(entry.get("row_seed"), case["row_seed"]),
         f"{where}: frozen TCEP index/seed mismatch")
    if not meta.get("data_hash"):
        return None  # A recorded preparation failure may have no canonical observations.
    need(isinstance(meta["data_hash"], str) and HASH.fullmatch(meta["data_hash"]), f"{where}: invalid canonical hash")
    need(meta.get("original_rows") == entry["original_rows"], f"{where}: TCEP indices differ from original experiment")
    need(meta.get("source_data_hash") == source_hash, f"{where}: TCEP raw source provenance mismatch")
    need(meta.get("tcep_reference_data_hash") == entry["data_hash"], f"{where}: TCEP historical reference hash mismatch")
    need(meta.get("tcep_reference_hash_policy") == TCEP_HASH_POLICY, f"{where}: TCEP reference hash policy mismatch")
    match = meta["data_hash"] == entry["data_hash"]
    need(type(meta.get("tcep_reference_hash_match")) is bool and meta["tcep_reference_hash_match"] == match,
         f"{where}: TCEP reference hash match flag is inconsistent")
    for field in ("row_seed", "presentation_seed"):
        need(equal(meta.get(field), case[field]), f"{where}: TCEP prepared {field} differs from manifest")
    return dict(case_id=where, data_hash=meta["data_hash"], source_data_hash=source_hash,
                tcep_reference_data_hash=entry["data_hash"], tcep_reference_hash_match=match,
                tcep_reference_hash_policy=TCEP_HASH_POLICY,
                row_seed=case["row_seed"], presentation_seed=case["presentation_seed"])


def verify_freeze(root):
    root = Path(root).resolve()
    path = root / "MANIFEST.json"
    document = read_json(path)
    files = document.get("files")
    need(isinstance(files, dict) and files, "MANIFEST.json needs a nonempty files hash map")
    for name, digest in files.items():
        target = (root / name).resolve()
        need(not Path(name).is_absolute() and target.is_relative_to(root), f"Unsafe frozen path: {name}")
        need(isinstance(digest, str) and HASH.fullmatch(digest), f"Invalid frozen hash: {name}")
        need(target.is_file() and sha256(target) == digest, f"Frozen file missing/modified: {name}")
    for suite in SUITES:
        need(f"config/manifests/{suite}.csv.gz" in files, f"Unfrozen {suite} manifest")
    return sha256(path)


def load_manifest(root, suite, strict_scope=True):
    path = Path(root) / "config" / "manifests" / f"{suite}.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    need(rows and all(row.get("case_id") for row in rows), f"Empty/invalid manifest: {path}")
    manifest = {row["case_id"]: row for row in rows}
    need(len(rows) == len(manifest), "Duplicate manifest case_id")
    for row in rows:
        need(row.get("truth") in ("1", "2"), f"Invalid manifest truth: {row['case_id']}")
        need(int(row["n"]) > 0, f"Invalid n: {row['case_id']}")
    if not strict_scope:  # Only used by synthetic unit fixtures, never by the CLI.
        return manifest
    need(len(rows) == {"anm": 58320, "tcep": 1122, "sensitivity": 1944, "nonanm": 3600}[suite],
         f"{suite}: frozen paper scope has an unexpected case count")
    if suite == "anm":
        need(Counter(row["n"] for row in rows) == {"200": 19440, "500": 19440, "1000": 19440},
             "Main ANM must retain all three sample sizes")
        cells = Counter((row["n"], row["cell_id"]) for row in rows)
        need(len(cells) == 2916 and set(cells.values()) == {20}, "ANM cells/repeats differ from the paper")
    elif suite == "sensitivity":
        cells = Counter(row["cell_id"] for row in rows)
        need({row["n"] for row in rows} == {"1000"} and len(cells) == 972 and set(cells.values()) == {2},
             "Sensitivity requires n1000, 972 cells, two repeats")
    elif suite == "nonanm":
        need(Counter((row["n"], row["model"]) for row in rows) ==
             {(str(n), model): 300 for n in (200, 500, 1000) for model in MODELS}, "NonANM design count mismatch")
        worlds = defaultdict(list)
        for row in rows:
            worlds[(row["n"], row["world_id"])].append(row)
        need(len(worlds) == 900, "NonANM requires 900 independent paired worlds")
        for group in worlds.values():
            need(len(group) == 4 and {r["model"] for r in group} == set(MODELS), "Incomplete paired world")
            for field in ("function_id", "cause_id", "noise_id", "rho", "delta", "design_seed", "data_seed"):
                need(len({r[field] for r in group}) == 1, f"Paired world differs on {field}")
    else:
        groups = defaultdict(list)
        for row in rows:
            need(row["cap"] in ("1000", "full"), "TCEP includes a cap outside the paper scope")
            groups[(row["cap"], row["subsample_seed"])].append(row)
            weight = float(row["weight"])
            need(math.isfinite(weight) and weight > 0, "Invalid official weight")
        need(sum(cap == "1000" for cap, _ in groups) == 10 and sum(cap == "full" for cap, _ in groups) == 1,
             "TCEP requires ten cap1000 seeds and one full evaluation")
        reference = None
        for group in groups.values():
            pairs = {row["pair_id"]: (row["truth"], row["weight"], row["continuous_continuous"]) for row in group}
            need(len(group) == len(pairs) == 102, "Each TCEP cap/seed needs 102 unique pairs")
            need(reference is None or pairs == reference, "TCEP labels/weights/subset differ across seeds")
            reference = pairs
            need(math.isclose(sum(float(row["weight"]) for row in group), 38.4979, abs_tol=1e-8), "Official weight sum mismatch")
            continuous = [r for r in group if r["continuous_continuous"] == "yes"]
            need(len(continuous) == 67 and math.isclose(sum(float(r["weight"]) for r in continuous), 22.7063, abs_tol=1e-8),
                 "Confirmed continuous subset mismatch")
    return manifest


def validate_record(raw, expected, fingerprint, method_keys):
    need(isinstance(raw, dict), "Record must be an object")
    row = dict(raw)
    where = f"{expected['case_id']}/{row.get('method')}/{row.get('configuration')}"
    for name, value in expected.items():
        need(name in row and equal(row[name], value), f"{where}: manifest mismatch {name}")
    need((row.get("method"), row.get("configuration")) in method_keys, f"{where}: unexpected method/configuration")
    need(row.get("protocol_hash") == fingerprint, f"{where}: record protocol_hash mismatch")
    need(row.get("direction_encoding") == "12", f"{where}: direction encoding must be 12")
    need(type(row.get("truth")) is int and row["truth"] in (1, 2), f"{where}: invalid truth")
    need(type(row.get("prediction_native")) is int and row["prediction_native"] in (0, 1, 2), f"{where}: invalid prediction")
    need(type(row.get("success")) is bool, f"{where}: success must be boolean")
    need(isinstance(row.get("error"), str), f"{where}: missing error string")
    need(isinstance(row.get("status"), str), f"{where}: missing status")
    need(isinstance(row.get("details", {}), dict), f"{where}: details must be an object")
    need(finite(row.get("seconds")) and row["seconds"] >= 0, f"{where}: invalid elapsed seconds")
    if row["success"]:
        need(not row["error"], f"{where}: success with an error")
        need(all(finite(row.get(k)) for k in ("score_forward", "score_reverse")), f"{where}: success without two finite scores")
        need(isinstance(row.get("data_hash"), str) and HASH.fullmatch(row["data_hash"]), f"{where}: invalid data_hash")
        need(type(row.get("swapped")) is bool, f"{where}: missing presentation metadata")
        need(equal(row.get("n_used"), expected["n"]), f"{where}: unexpected sample removal")
    else:
        need(row["prediction_native"] == 0 and row["error"].strip(), f"{where}: failed result needs zero prediction and error")
        if row.get("data_hash"):
            need(isinstance(row["data_hash"], str) and HASH.fullmatch(row["data_hash"]), f"{where}: invalid failure hash")
    if row["success"] and "presented_prediction_native" in row:
        presented = row["presented_prediction_native"]
        need(type(presented) is int and presented in (0, 1, 2), f"{where}: invalid presented prediction")
        need(row["prediction_native"] == (3 - presented if row["swapped"] and presented else presented),
             f"{where}: direction coordinate mismatch")
    if row["success"] and "presented_score_forward" in row and "presented_score_reverse" in row:
        pair = (row["presented_score_forward"], row["presented_score_reverse"])
        need((row["score_forward"], row["score_reverse"]) == (pair[::-1] if row["swapped"] else pair), f"{where}: score coordinate mismatch")
    if row["success"] and row["method"] in ("ANM_GP", "ANM_KRR", "PNL-causal-learn"):
        p, q = row.get("p_forward"), row.get("p_reverse")
        need(all(finite(v) and 0 <= v <= 1 for v in (p, q)), f"{where}: invalid residual p-values")
        if row["method"] == "PNL-causal-learn":
            predicted = 1 if p > q else 2 if q > p else 0
        else:
            predicted = (1 if p > 0.05 else 2) if (p > 0.05) != (q > 0.05) else 0
        need(row["prediction_native"] == predicted, f"{where}: method decision rule mismatch")
    return row


def score_decision(row):
    """Compare original-coordinate outputs, ignoring screens and thresholds."""
    if not row["success"]:
        return 0
    larger = row["method"] in ("ANM_GP", "ANM_KRR", "PNL-causal-learn")
    pair = (row["p_forward"], row["p_reverse"]) if larger else (row["score_forward"], row["score_reverse"])
    forward, reverse = pair
    need(all(finite(v) for v in pair), "Score comparison requires two finite values")
    if forward == reverse:
        return 0
    return 1 if (forward > reverse if larger else forward < reverse) else 2


def grouping_keys(suite, case, method, configuration):
    """Every TCEP key contains one cap AND one seed: weights are never pooled across seeds."""
    base = dict(suite=suite, method=method, configuration=configuration)
    def key(**fields):
        fields = dict(base, **fields)
        return tuple(str(fields.get(name, "")) for name in GROUP_FIELDS)
    if suite == "tcep":
        subsets = ["all"] + (["continuous_confirmed"] if case["continuous_continuous"] == "yes" else [])
        for subset in subsets:
            for weighting in ("unweighted", "official_weighted"):
                yield key(scope="per_seed", cap=case["cap"], seed=case["subsample_seed"], subset=subset, weighting=weighting)
        return
    n = str(case["n"])
    if suite == "nonanm":
        scopes = [case["model"]]
        if case["model"] != "ANM_control":
            scopes.append("nonanm_combined")
    else:
        scopes = ["overall"]
    for scope in scopes:
        for size in ("all", n):
            yield key(scope=scope, n=size)
            for branch in ("function_name", "cause_name", "noise_name", "rho"):
                if case.get(branch) not in (None, ""):
                    yield key(scope=scope, n=size, branch=branch, level=case[branch])
            if suite == "nonanm" and scope == "post_nonlinear":
                yield key(scope=scope, n=size, branch="outer_id", level=case["outer_id"])


def ratio(a, b):
    return a / b if b else None


class Counts:
    def __init__(self):
        self.expected = self.observed = self.success = self.directed = self.correct = self.abstain = self.failed = 0
        self.weight = self.observed_weight = self.directed_weight = self.correct_weight = 0.0

    def plan(self, weight):
        self.expected += 1
        self.weight += weight

    def add(self, row, weight):
        self.observed += 1
        self.observed_weight += weight
        success = row["success"]
        prediction = row.get("prediction_evaluated", row["prediction_native"])
        directed = success and prediction != 0
        correct = directed and prediction == row["truth"]
        self.success += success
        self.failed += not success
        self.directed += directed
        self.correct += correct
        self.abstain += success and not directed
        self.directed_weight += weight * directed
        self.correct_weight += weight * correct

    def finish(self):
        complete = self.expected == self.observed
        return dict(N=self.expected, observed=self.observed, missing=self.expected-self.observed,
                    success=self.success, failures=self.failed, directed=self.directed,
                    correct=self.correct, abstentions=self.abstain, total_weight=self.weight,
                    observed_weight=self.observed_weight, directed_weight=self.directed_weight,
                    correct_weight=self.correct_weight, complete=complete,
                    A=ratio(self.correct_weight, self.weight) if complete else None,
                    C=ratio(self.directed_weight, self.weight) if complete else None,
                    CA=ratio(self.correct_weight, self.directed_weight) if complete else None,
                    partial_observed_A=ratio(self.correct_weight, self.observed_weight) if not complete else None,
                    partial_observed_C=ratio(self.directed_weight, self.observed_weight) if not complete else None,
                    partial_observed_CA=ratio(self.correct_weight, self.directed_weight) if not complete else None)


def quantile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def detail(row, name):
    return row.get(name, (row.get("details") or {}).get(name))


def csv_value(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return value


def write_csv(path, rows, fields=None):
    fields = list(fields or dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(v) for k, v in row.items() if k in fields})


def seed_summaries(metrics):
    groups = defaultdict(list)
    for row in metrics:
        if row["scope"] == "per_seed":
            key = tuple(row[k] for k in ("cap", "subset", "weighting", "method", "configuration"))
            groups[key].append(row)
    results = []
    for key, rows in sorted(groups.items()):
        result = dict(zip(("cap", "subset", "weighting", "method", "configuration"), key))
        result.update(scope="seed_mean", suite="tcep", seed_count=len(rows), complete=all(r["complete"] for r in rows),
                      N_per_seed=rows[0]["N"], weight_per_seed=rows[0]["total_weight"])
        for metric in ("A", "C", "CA"):
            values = [row[metric] for row in rows if row[metric] is not None]
            result[f"{metric}_defined_seeds"] = len(values)
            # Undefined CA (no decisions) stays undefined, never zero or silently discarded.
            result[metric] = statistics.mean(values) if result["complete"] and len(values) == len(rows) else None
            result[f"{metric}_sd"] = statistics.stdev(values) if result[metric] is not None and len(values) > 1 else None
        results.append(result)
    return results


def add_ranks(rows):
    groups = defaultdict(list)
    for row in rows:
        key = tuple((k, row.get(k, "")) for k in GROUP_FIELDS if k not in ("method", "configuration"))
        groups[key].append(row)
    for group in groups.values():
        for row in group:
            row["A_rank"] = 1 + sum(other["A"] > row["A"] for other in group if other.get("A") is not None) if row.get("A") is not None else None


def render_md(result):
    lines = [f"# {result['suite']} — {result['decision_rule']} decisions", "",
             "**COMPLETE**" if result["complete"] else "**PARTIAL — not a completed experiment; incomplete formal metrics are blank.**", "",
             "A = correct / all planned cases; C = directed / all planned cases; CA = correct / directed.",
             "Successful abstentions and execution failures are separate. Failures stay in A/C denominators.",
             "Machine-readable rates are fractions; tables below show percentages. Rankings are descriptive, not significance tests.", "",
             f"Planned records: {result['accounting']['expected_records']}; observed: {result['accounting']['observed_records']}; "
             f"missing: {result['accounting']['missing_records']}; failures: {result['accounting']['failures']}; "
             f"successful abstentions: {result['accounting']['abstentions']}.", ""]
    lines += ["PNL uses unmodified causal-learn outputs with an external larger-p rule; exact ties abstain. Its upstream t-test is not an independence test.", ""]
    if result["decision_rule"] == "score":
        lines += ["Direct comparison: larger p for ANM-GP, ANM-KRR and PNL; smaller score for RKDS, RECI-MON3 and KCDC. Screens and significance thresholds are ignored; exact ties abstain.", ""]
    if result["suite"] == "tcep":
        lines += ["Each seed is scored separately using the official pair weights. Means and sample SDs give every seed equal weight; full is evaluated once. Undefined per-seed CA is not silently removed.", "",
                  "The 67 continuous–continuous pairs were selected from official data descriptions; this is not an official predefined subset. TCEP screen p-values are benchmark heuristics.", ""]
        audit = result["tcep_provenance_audit"]
        lines += ["Frozen raw source CSVs, exact row indices and seeds are checked strictly. The historical standardized CSV byte hash is an audit reference, not a cross-runtime byte-equality requirement; all methods must share the actual current canonical hash.", "",
                  f"Prepared cases audited: {audit['prepared_cases']}; historical canonical byte-hash differences: {audit['reference_hash_mismatches']}. Per-case provenance is retained in records.csv and summary.json.", ""]
        rows = result["seed_summary"]
        group_fields = ("cap", "subset", "weighting")
    else:
        if result["suite"] == "nonanm":
            lines += ["ANM_control is separate from nonanm_combined. Four structures share each world; sample sizes use independent worlds. No independent/no-edge negative control is present.", ""]
        rows = [r for r in result["metrics"] if not r["branch"] or r["n"] == "all"]
        group_fields = ("scope", "n", "branch", "level")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in group_fields)].append(row)
    def pct(value):
        return "—" if value is None else f"{100 * value:.2f}"
    for key, group in sorted(groups.items()):
        lines += ["## " + " / ".join(v for v in key if v), "", "| Method/config | N | A (%) | C (%) | CA (%) | Directed | Abstained | Failed | A rank |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in group:
            values = []
            for metric in ("A", "C", "CA"):
                value = pct(row.get(metric))
                if row.get(f"{metric}_sd") is not None:
                    value += " ± " + pct(row[f"{metric}_sd"])
                values.append(value)
            lines.append(f"| {row['method']} / {row['configuration']} | {row.get('N', row.get('N_per_seed', ''))} | " + " | ".join(values) + f" | {row.get('directed', '—')} | {row.get('abstentions', '—')} | {row.get('failures', '—')} | {row.get('A_rank') or '—'} |")
        lines.append("")
    if result["sensitivity"]:
        lines += ["## Native changes relative to H13", "", "Same-case comparisons use original coordinates and require both fits to succeed. A direction flip excludes direction↔abstention transitions; native changes include all unequal native codes.", "",
                  "| Configuration | Paired successes | Native changes | Change rate (%) | Opposite directions | Direction→abstain | Abstain→direction |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for row in result["sensitivity"]:
            lines.append(f"| {row['configuration']} | {row['paired_success']} | {row['native_changes']} | {pct(row['native_change_rate'])} | {row['direction_flips']} | {row['direction_to_abstain']} | {row['abstain_to_direction']} |")
        lines.append("")
    lines += ["## Numerical diagnostics", "", "Two directional values per expected RKDS record. Missing/nonfinite diagnostics are counted, not omitted from the audit.", "",
              "| n/cap | Configuration | Min denominator | Nonpositive | Denominator missing | κ median | κ max | κ missing |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in result["diagnostics"]:
        fmt = lambda v: "—" if v is None else f"{v:.6g}"
        lines.append(f"| {row['n_or_cap']} | {row['configuration']} | {fmt(row['denominator_min'])} | {row['denominator_nonpositive']} | {row['denominator_missing']} | {fmt(row['condition_median'])} | {fmt(row['condition_max'])} | {row['condition_missing']} |")
    lines += ["", "## Runtime (recorded seconds, not wall-clock)", "", "Do not sum shared-grid timing across sensitivity configurations. Hardware/concurrency affect these descriptive times.", "", "| n/cap | Method/config | Mean | Median | P95 | Max | Timing scopes |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in result["runtime"]:
        lines.append(f"| {row['n_or_cap']} | {row['method']}/{row['configuration']} | {row['mean']:.6g} | {row['median']:.6g} | {row['p95']:.6g} | {row['max']:.6g} | {json.dumps(row['timing_scopes'], sort_keys=True)} |")
    lines += ["", "## Abstentions", "", "```json", json.dumps(result["abstention_reasons"], indent=2, sort_keys=True), "```", ""]
    return "\n".join(lines)


def evaluate(root, suite, allow_partial=False, strict_scope=True, fingerprint=None, decision_rule="native"):
    """Validate and atomically publish this suite. Tests may use strict_scope=False."""
    root = Path(root).resolve()
    need(suite in SUITES, f"Unknown suite: {suite}")
    need(decision_rule in ("native", "score"), "Unknown decision rule")
    need(decision_rule == "native" or suite != "sensitivity", "Sensitivity uses method decisions only")
    fingerprint = fingerprint or verify_freeze(root)
    manifest = load_manifest(root, suite, strict_scope)
    tcep_indices = {}
    source_hashes = {}
    if suite == "tcep":
        for relative in {row.get("index_file") for row in manifest.values()} - {None, ""}:
            path = (root / relative).resolve()
            need(not Path(relative).is_absolute() and path.is_relative_to(root), "Unsafe TCEP index path")
            frozen = read_json(root / "MANIFEST.json")["files"]
            need(relative in frozen and sha256(path) == frozen[relative], "TCEP indices are not frozen")
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                document = json.load(handle, parse_constant=reject_constant)
            need(document.get("schema_version") == 1 and document.get("index_base") == 1,
                 "Unsupported TCEP row-index schema")
            tcep_indices.update(document["cases"])
        need(not strict_scope or set(manifest).issubset(tcep_indices), "Missing frozen TCEP index entries")
        if tcep_indices:
            source_hashes = tcep_source_hashes(root, manifest, read_json(root / "MANIFEST.json")["files"])
    method_keys = set(expected_methods(suite))
    counts = defaultdict(Counts)
    diagnostics = defaultdict(lambda: {"expected": 0, "denominator": [], "condition": []})
    for case in manifest.values():
        diagnostic_key = (case["cap"] if suite == "tcep" else case["n"],)
        for method, config in expected_methods(suite):
            for key in grouping_keys(suite, case, method, config):
                weight = float(case["weight"]) if key[8] == "official_weighted" else 1.0
                counts[key].plan(weight)
            if method == "RKDS":
                diagnostics[diagnostic_key + (config,)]["expected"] += 2
    output = root / "analysis" / suite if decision_rule == "native" else root / "analysis" / "score" / suite
    output.mkdir(parents=True, exist_ok=True)
    runtime = defaultdict(list)
    timing_scopes = defaultdict(Counter)
    abstention_reasons = defaultdict(Counter)
    changes = {config: Counter() for config in VARIANTS} if suite == "sensitivity" else {}
    seen = set()
    checkpoint_hashes = {}
    tcep_provenance = []
    failures = abstentions = successes = 0
    record_fields = list(dict.fromkeys(list(next(iter(manifest.values()))) + list(RESULT_FIELDS)))
    with tempfile.TemporaryDirectory(prefix=".evaluate-", dir=output) as temporary:
        temporary = Path(temporary)
        with (temporary / "records.csv").open("w", encoding="utf-8", newline="") as record_file, (temporary / "errors.csv").open("w", encoding="utf-8", newline="") as error_file:
            writers = [csv.DictWriter(handle, fieldnames=record_fields, extrasaction="ignore") for handle in (record_file, error_file)]
            for writer in writers:
                writer.writeheader()
            for path in sorted((root / "results" / suite / "checkpoints").glob("*.json")):
                raw = path.read_bytes()
                bundle = json.loads(raw, parse_constant=reject_constant)
                need(isinstance(bundle, dict) and bundle.get("fingerprint") == fingerprint, f"{path.name}: checkpoint fingerprint mismatch")
                need(bundle.get("suite") == suite, f"{path.name}: checkpoint suite mismatch")
                records, inputs, completed = bundle.get("records"), bundle.get("inputs"), bundle.get("completed")
                need(isinstance(records, list) and isinstance(inputs, dict) and isinstance(completed, list), f"{path.name}: invalid checkpoint containers")
                need(len(inputs) == 1, f"{path.name}: checkpoint must contain one case input")
                case_id = next(iter(inputs))
                need(case_id in manifest, f"{path.name}: case outside manifest")
                need(case_id not in seen, f"Duplicate checkpoint case: {case_id}")
                seen.add(case_id)
                job = hashlib.sha256(case_id.encode()).hexdigest()[:20]
                need(path.stem == bundle.get("job_id") == job, f"{path.name}: job ID mismatch")
                meta = inputs[case_id]
                need(isinstance(meta, dict), f"{case_id}: invalid input metadata")
                if case_id in tcep_indices:
                    entry = tcep_indices[case_id]
                    provenance = validate_tcep_provenance(meta, entry, manifest[case_id],
                                                         source_hashes[manifest[case_id]["data_file"]])
                    if provenance:
                        tcep_provenance.append(provenance)
                row_keys = set()
                local = {}
                for original in records:
                    need(original.get("case_id") == case_id, f"{path.name}: cross-case record")
                    row = validate_record(original, manifest[case_id], fingerprint, method_keys)
                    row["prediction_evaluated"] = row["prediction_native"] if decision_rule == "native" else score_decision(row)
                    row["decision_rule"] = decision_rule
                    key = (row["method"], row["configuration"])
                    need(key not in row_keys, f"Duplicate method/configuration: {case_id}/{key}")
                    row_keys.add(key)
                    local[key] = row
                    if row.get("data_hash"):
                        need(row["data_hash"] == meta.get("data_hash"), f"{case_id}: data_hash differs from canonical input")
                        if case_id in tcep_indices:
                            for field in ("source_data_hash", "tcep_reference_data_hash", "tcep_reference_hash_match", "tcep_reference_hash_policy"):
                                need(field not in row or row[field] == meta[field], f"{case_id}: record TCEP provenance differs from input")
                                row[field] = meta[field]
                    if row["success"]:
                        need(meta.get("swapped") is row["swapped"] and equal(meta.get("truth"), row["truth"]), f"{case_id}: input coordinates disagree")
                        need(equal(meta.get("n_used"), row["n_used"]), f"{case_id}: canonical n differs")
                        need(equal(meta.get("generation_attempts"), 1), f"{case_id}: regenerated sample")
                        need(len(meta.get("original_rows", [])) == int(row["n_used"]) and len(set(meta["original_rows"])) == int(row["n_used"]), f"{case_id}: invalid row selection metadata")
                    for group_key in grouping_keys(suite, manifest[case_id], *key):
                        weight = float(manifest[case_id]["weight"]) if group_key[8] == "official_weighted" else 1.0
                        counts[group_key].add(row, weight)
                    failures += not row["success"]
                    successes += row["success"]
                    is_abstention = row["success"] and row["prediction_evaluated"] == 0
                    abstentions += is_abstention
                    if is_abstention:
                        reason = "equal_scores" if decision_rule == "score" else row.get("undirected_reason") or row["status"]
                        abstention_reasons[row["method"]][reason] += 1
                    size = str(manifest[case_id]["cap"] if suite == "tcep" else manifest[case_id]["n"])
                    runtime[(size,) + key].append(row["seconds"])
                    timing_scopes[(size,) + key][row.get("timing_scope", "unspecified")] += 1
                    if row["method"] == "RKDS":
                        diagnostic = diagnostics[(size, row["configuration"])]
                        for prefix, field in (("denominator", "denominator"), ("condition_number", "condition")):
                            for direction in ("forward", "reverse"):
                                value = detail(row, f"{prefix}_{direction}")
                                if finite(value):
                                    diagnostic[field].append(value)
                    row["checkpoint"] = path.name
                    export = {k: csv_value(v) for k, v in row.items() if k in record_fields}
                    writers[0].writerow(export)
                    if not row["success"]:
                        writers[1].writerow(export)
                need(completed == ([case_id] if row_keys == method_keys else []), f"{case_id}: completed marker disagrees with method keys")
                if suite == "sensitivity":
                    baseline = local.get(("RKDS", "H13"))
                    for config, counter in changes.items():
                        variant = local.get(("RKDS", config))
                        if baseline is None or variant is None:
                            counter["missing_pair"] += 1
                        elif not baseline["success"] or not variant["success"]:
                            counter["failed_pair"] += 1
                        else:
                            p, q = baseline["prediction_native"], variant["prediction_native"]
                            counter["paired_success"] += 1
                            counter["native_changes"] += p != q
                            counter["direction_flips"] += bool(p and q and p != q)
                            counter["direction_to_abstain"] += bool(p and not q)
                            counter["abstain_to_direction"] += bool(not p and q)
                checkpoint_hashes[path.name] = hashlib.sha256(raw).hexdigest()
        observed = successes + failures
        expected = len(manifest) * len(method_keys)
        missing = expected - observed
        need(missing == 0 or allow_partial, f"Incomplete {suite}: missing {missing}/{expected} records; use --allow-partial for diagnostic output")
        metrics = [dict(zip(GROUP_FIELDS, key), **value.finish()) for key, value in sorted(counts.items())]
        add_ranks(metrics)
        seeds = seed_summaries(metrics)
        add_ranks(seeds)
        numerical = []
        for (size, config), values in sorted(diagnostics.items()):
            denominators, conditions = values["denominator"], values["condition"]
            numerical.append(dict(n_or_cap=size, configuration=config, expected_directional_values=values["expected"],
                                  denominator_min=min(denominators) if denominators else None,
                                  denominator_nonpositive=sum(v <= 0 for v in denominators),
                                  denominator_missing=values["expected"]-len(denominators),
                                  condition_median=quantile(conditions, .5), condition_max=max(conditions) if conditions else None,
                                  condition_missing=values["expected"]-len(conditions)))
        variants = []
        for config, counter in changes.items():
            counter["missing_pair"] += len(manifest)-len(seen)
            item = {name: counter[name] for name in ("paired_success", "missing_pair", "failed_pair", "native_changes", "direction_flips", "direction_to_abstain", "abstain_to_direction")}
            item.update(configuration=config, N=len(manifest), native_change_rate=ratio(counter["native_changes"], counter["paired_success"]))
            variants.append(item)
        if suite in ("anm", "nonanm"):
            for key, values in list(runtime.items()):
                combined = ("all",) + key[1:]
                runtime[combined].extend(values)
                timing_scopes[combined].update(timing_scopes[key])
        runtimes = [dict(n_or_cap=key[0], method=key[1], configuration=key[2], count=len(values),
                         mean=statistics.mean(values), median=quantile(values, .5), p95=quantile(values, .95),
                         max=max(values), timing_scopes=dict(timing_scopes[key])) for key, values in sorted(runtime.items())]
        result = dict(schema_version=1, suite=suite, decision_rule=decision_rule, complete=missing == 0,
                      generated_utc=datetime.now(timezone.utc).isoformat(), protocol_hash=fingerprint,
                      protocol_fingerprint=fingerprint,
                      accounting=dict(expected_cases=len(manifest), checkpoint_cases=len(seen), expected_records=expected,
                                      observed_records=observed, missing_records=missing, success_records=successes,
                                      failures=failures, abstentions=abstentions),
                      rate_units="fraction", direction_encoding="12", metrics=metrics, seed_summary=seeds,
                      sensitivity=variants, diagnostics=numerical, runtime=runtimes,
                      abstention_reasons={k: dict(v) for k, v in abstention_reasons.items()},
                      tcep_provenance_audit=dict(policy=TCEP_HASH_POLICY, prepared_cases=len(tcep_provenance),
                                                 reference_hash_mismatches=sum(not row["tcep_reference_hash_match"] for row in tcep_provenance),
                                                 cases=tcep_provenance),
                      checkpoint_sha256=checkpoint_hashes,
                      checkpoint_snapshot=dict(files=[dict(path=f"results/{suite}/checkpoints/{name}", sha256=digest)
                                                       for name, digest in sorted(checkpoint_hashes.items())]),
                      caveat="Descriptive only. Incomplete groups have blank formal rates; partial_observed rates are not complete-experiment estimates.")
        write_csv(temporary / "metrics.csv", metrics + seeds)
        (temporary / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
        (temporary / "summary.md").write_text(render_md(result), encoding="utf-8")
        for name in ("records.csv", "errors.csv", "metrics.csv", "summary.json", "summary.md"):
            os.replace(temporary / name, output / name)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES + ("all",), required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--allow-partial", action="store_true", help="Write explicitly partial diagnostics; never label them complete")
    parser.add_argument("--decision-rule", choices=("native", "score"), default="native")
    args = parser.parse_args(argv)
    try:
        fingerprint = verify_freeze(args.root)
        for suite in SUITES if args.suite == "all" else (args.suite,):
            if args.suite == "all" and suite == "sensitivity" and args.decision_rule == "score":
                continue
            result = evaluate(args.root, suite, args.allow_partial, fingerprint=fingerprint, decision_rule=args.decision_rule)
            print(json.dumps(dict(suite=suite, complete=result["complete"], **result["accounting"]), sort_keys=True))
        return 0
    except (ValidationError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"Evaluation refused: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
