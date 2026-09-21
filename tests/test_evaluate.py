"""Artificial result fixtures only: never run an experiment algorithm."""
import copy
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

SPEC = importlib.util.spec_from_file_location("paper_evaluate", Path(__file__).resolve().parents[1] / "src/evaluate.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def case(case_id="case1", **values):
    row = dict(case_id=case_id, domain="anm", split="main", n="8", truth="1",
               function_name="f", cause_name="g", noise_name="e", rho="1", cap="",
               data_seed="123", presentation_seed="456")
    row.update({k: str(v) for k, v in values.items()})
    return row


def record(row, method="RKDS", configuration="H13", prediction=1, success=True):
    value = dict(row, method=method, configuration=configuration, truth=int(row["truth"]),
                 prediction_native=prediction, direction_encoding="12", success=success,
                 error="" if success else "artificial failure", status="directed" if prediction else "abstain",
                 score_forward=.1 if success else None, score_reverse=.2 if success else None,
                 data_hash="a" * 64, seconds=.5, swapped=False, n_used=int(row["n"]),
                 denominator_forward=.001, denominator_reverse=.002,
                 condition_number_forward=10, condition_number_reverse=30,
                 details={})
    if method in ("ANM_GP", "ANM_KRR", "PNL-causal-learn"):
        value["p_forward"], value["p_reverse"] = {1: (.8, .001), 2: (.001, .8), 0: (.001, .001)}[prediction]
        value["score_forward"], value["score_reverse"] = value["p_forward"], value["p_reverse"]
    return value


class Fixture:
    def __init__(self, root, suite, cases, records=None, extra_frozen=()):
        self.root, self.suite, self.cases = Path(root), suite, cases
        self.manifests = self.root / "config" / "manifests"
        self.manifests.mkdir(parents=True)
        for name in evaluation.SUITES:
            with gzip.open(self.manifests / f"{name}.csv.gz", "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
                writer.writeheader()
                writer.writerows(cases if name == suite else [cases[0]])
        files = {f"config/manifests/{name}.csv.gz": evaluation.sha256(self.manifests / f"{name}.csv.gz") for name in evaluation.SUITES}
        files.update({relative: evaluation.sha256(self.root / relative) for relative in extra_frozen})
        (self.root / "MANIFEST.json").write_text(json.dumps(dict(experiment_id="artificial-fixture", files=files)))
        self.fingerprint = evaluation.sha256(self.root / "MANIFEST.json")
        self.paths = {}
        for row in cases:
            case_id = row["case_id"]
            current = records.get(case_id, []) if records is not None else [record(row, *key) for key in evaluation.expected_methods(suite)]
            self.save(row, current)

    def save(self, row, records, meta=None):
        case_id = row["case_id"]
        job = hashlib.sha256(case_id.encode()).hexdigest()[:20]
        path = self.root / "results" / self.suite / "checkpoints" / f"{job}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        records = copy.deepcopy(records)
        for value in records:
            value["protocol_hash"] = self.fingerprint
        if meta is None:
            meta = dict(data_hash="a" * 64, swapped=False, truth=int(row["truth"]),
                        n_used=int(row["n"]), generation_attempts=1, original_rows=list(range(1, int(row["n"])+1)))
        keys = {(value["method"], value["configuration"]) for value in records}
        bundle = dict(job_id=job, fingerprint=self.fingerprint, suite=self.suite, role="all",
                      records=records, inputs={case_id: meta}, completed=[case_id] if keys == set(evaluation.expected_methods(self.suite)) else [])
        path.write_text(json.dumps(bundle))
        self.paths[case_id] = path

    def modify(self, case_id, callback):
        path = self.paths[case_id]
        value = json.loads(path.read_text())
        callback(value)
        path.write_text(json.dumps(value))

    def run(self, **options):
        return evaluation.evaluate(self.root, self.suite, strict_scope=False, **options)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_counts_failures_and_abstentions_separate(self):
        counts = evaluation.Counts()
        values = [(1, True), (0, True), (0, False), (2, True)]
        for prediction, success in values:
            counts.plan(1)
            counts.add(record(case(), prediction=prediction, success=success), 1)
        result = counts.finish()
        self.assertEqual((result["N"], result["correct"], result["directed"], result["abstentions"], result["failures"]), (4, 1, 2, 1, 1))
        self.assertEqual((result["A"], result["C"], result["CA"]), (.25, .5, .5))

    def test_complete_native_outputs_and_snapshot(self):
        fixture = Fixture(self.root, "anm", [case()])
        result = fixture.run()
        self.assertTrue(result["complete"])
        self.assertEqual(result["accounting"]["observed_records"], 6)
        self.assertEqual(result["protocol_fingerprint"], fixture.fingerprint)
        self.assertTrue(result["checkpoint_snapshot"]["files"][0]["path"].startswith("results/anm/checkpoints/"))
        self.assertEqual(result["diagnostics"][0]["condition_median"], 20)
        self.assertEqual(result["diagnostics"][0]["n_or_cap"], "8")
        self.assertEqual({p.name for p in (self.root / "analysis" / "anm").iterdir()}, {"summary.md", "summary.json", "metrics.csv", "records.csv", "errors.csv"})
        for metric in result["metrics"]:
            self.assertEqual(metric["A"], 1)

    def test_pnl_compares_pvalues_without_alpha(self):
        row = case()
        value = record(row, "PNL-causal-learn", "default", prediction=1)
        value.update(p_forward=.02, p_reverse=.01, score_forward=.02, score_reverse=.01, protocol_hash="test")
        evaluation.validate_record(value, row, "test", set(evaluation.METHODS))
        value["prediction_native"] = 0
        with self.assertRaisesRegex(evaluation.ValidationError, "decision rule"):
            evaluation.validate_record(value, row, "test", set(evaluation.METHODS))

    def test_direct_score_direction_and_exact_ties(self):
        for method, config in evaluation.METHODS:
            value = record(case(), method, config)
            larger = method in ("ANM_GP", "ANM_KRR", "PNL-causal-learn")
            value.update(score_forward=0., score_reverse=1e-15, p_forward=0., p_reverse=1e-15)
            self.assertEqual(evaluation.score_decision(value), 2 if larger else 1)
            value.update(score_reverse=0., p_reverse=0.)
            self.assertEqual(evaluation.score_decision(value), 0)
            value.update(success=False)
            self.assertEqual(evaluation.score_decision(value), 0)

    def test_score_analysis_preserves_native_records(self):
        row = case()
        values = [record(row, *key) for key in evaluation.METHODS]
        values[0]["prediction_native"] = 0
        gp = next(v for v in values if v["method"] == "ANM_GP")
        gp.update(prediction_native=0, p_forward=.02, p_reverse=.01, score_forward=.02, score_reverse=.01)
        fixture = Fixture(self.root, "anm", [row], {row["case_id"]: values})
        before = fixture.paths["case1"].read_bytes()
        native = fixture.run()
        result = fixture.run(decision_rule="score")
        self.assertEqual(result["accounting"]["abstentions"], 0)
        self.assertEqual(native["accounting"]["abstentions"], 2)
        self.assertEqual(before, fixture.paths["case1"].read_bytes())
        for metric in result["metrics"]:
            self.assertEqual(metric["A"], 1)
        with (self.root / "analysis/score/anm/records.csv").open() as handle:
            saved = list(csv.DictReader(handle))
        self.assertEqual(saved[0]["prediction_native"], "0")
        self.assertEqual(saved[0]["prediction_evaluated"], "1")
        self.assertTrue((self.root / "analysis/anm/summary.json").is_file())

    def test_partial_requires_opt_in_and_formal_rates_blank(self):
        row = case()
        fixture = Fixture(self.root, "anm", [row], {row["case_id"]: [record(row)]})
        with self.assertRaisesRegex(evaluation.ValidationError, "Incomplete"):
            fixture.run()
        result = fixture.run(allow_partial=True)
        self.assertFalse(result["complete"])
        missing = [r for r in result["metrics"] if r["method"] == "KCDC"][0]
        self.assertEqual(missing["missing"], 1)
        self.assertIsNone(missing["A"])
        self.assertIsNone(missing["CA"])
        self.assertIn("PARTIAL", (self.root / "analysis" / "anm" / "summary.md").read_text())

    def test_preparation_failure_without_hash_is_not_abstention(self):
        row = case()
        values = [record(row, *key, prediction=0, success=False) for key in evaluation.METHODS]
        for value in values:
            value["data_hash"] = None
        fixture = Fixture(self.root, "anm", [row], {row["case_id"]: values})
        fixture.save(row, values, meta=dict(prepared=False, preparation_error="artificial failure"))
        result = fixture.run()
        self.assertEqual(result["accounting"]["failures"], 6)
        self.assertEqual(result["accounting"]["abstentions"], 0)
        self.assertTrue(result["complete"])

    def test_duplicate_record_is_rejected(self):
        fixture = Fixture(self.root, "anm", [case()])
        fixture.modify("case1", lambda b: b["records"].append(copy.deepcopy(b["records"][0])))
        with self.assertRaisesRegex(evaluation.ValidationError, "Duplicate method"):
            fixture.run()

    def test_hash_disagreement_is_rejected(self):
        fixture = Fixture(self.root, "anm", [case()])
        fixture.modify("case1", lambda b: b["records"][1].update(data_hash="b"*64))
        with self.assertRaisesRegex(evaluation.ValidationError, "data_hash"):
            fixture.run()

    def test_manifest_seed_mismatch_is_rejected(self):
        fixture = Fixture(self.root, "anm", [case()])
        fixture.modify("case1", lambda b: b["records"][0].update(data_seed=124))
        with self.assertRaisesRegex(evaluation.ValidationError, "manifest mismatch data_seed"):
            fixture.run()

    def test_modified_frozen_manifest_is_rejected(self):
        fixture = Fixture(self.root, "anm", [case()])
        with gzip.open(fixture.manifests / "anm.csv.gz", "at", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(evaluation.ValidationError, "Frozen file"):
            fixture.run()

    def test_p_values_and_boundary_native_rule(self):
        row = case()
        value = record(row, "ANM_KRR", "default", prediction=1)
        value.update(protocol_hash="f"*64, p_forward=.05, p_reverse=.001)
        with self.assertRaisesRegex(evaluation.ValidationError, "decision rule"):
            evaluation.validate_record(value, row, "f"*64, set(evaluation.METHODS))
        value["prediction_native"] = 0
        evaluation.validate_record(value, row, "f"*64, set(evaluation.METHODS))
        value["p_forward"] = 1.1
        with self.assertRaisesRegex(evaluation.ValidationError, "p-values"):
            evaluation.validate_record(value, row, "f"*64, set(evaluation.METHODS))

    def test_coordinate_mapping(self):
        row = case()
        value = record(row, prediction=2)
        value.update(protocol_hash="f"*64, swapped=True, presented_prediction_native=1,
                     presented_score_forward=.2, presented_score_reverse=.1)
        evaluation.validate_record(value, row, "f"*64, set(evaluation.METHODS))
        value["presented_prediction_native"] = 2
        with self.assertRaisesRegex(evaluation.ValidationError, "coordinate"):
            evaluation.validate_record(value, row, "f"*64, set(evaluation.METHODS))

    def test_sensitivity_shared_cases_native_transitions(self):
        row = case(n=1000)
        values = [record(row, *key) for key in evaluation.expected_methods("sensitivity")]
        values[1]["prediction_native"] = 2
        values[2]["prediction_native"] = 0
        fixture = Fixture(self.root, "sensitivity", [row], {row["case_id"]: values})
        result = fixture.run()
        variants = {r["configuration"]: r for r in result["sensitivity"]}
        self.assertEqual(variants[evaluation.VARIANTS[1]]["direction_flips"], 1)
        self.assertEqual(variants[evaluation.VARIANTS[2]]["direction_to_abstain"], 1)
        self.assertEqual(variants[evaluation.VARIANTS[2]]["native_change_rate"], 1)
        self.assertEqual(variants["H13"]["native_changes"], 0)
        self.assertEqual(result["accounting"]["observed_records"], 12)

    def test_nonanm_control_not_in_combined_and_outer_only_post(self):
        cases = [case(model, domain="nonanm", split="nonanm", model=model, world_id="world", outer_id="cubic") for model in evaluation.MODELS]
        fixture = Fixture(self.root, "nonanm", cases)
        result = fixture.run()
        combined = [r for r in result["metrics"] if r["scope"] == "nonanm_combined" and r["n"] == "all" and not r["branch"]]
        self.assertEqual([r["N"] for r in combined], [3]*6)
        self.assertEqual({r["scope"] for r in result["metrics"] if r["branch"] == "outer_id"}, {"post_nonlinear"})

    def test_tcep_weighted_seed_mean_not_pooled_ca(self):
        cases = [case(f"p{pair}s{seed}", domain="tcep", split="tcep", cap=1000,
                      pair_id=pair, subsample_seed=seed, weight=weight,
                      continuous_continuous="yes" if pair == 1 else "ambiguous")
                 for seed in (1, 2) for pair, weight in ((1, 1), (2, 3))]
        records = {}
        # Seed1: both directed, weight1 correct/4 => CA .25.
        # Seed2: only weight1 directed and correct => CA 1. Mean .625, not pooled .4.
        for row in cases:
            predicted = 1 if row["pair_id"] == "1" else 2 if row["subsample_seed"] == "1" else 0
            records[row["case_id"]] = [record(row, *key, prediction=predicted) for key in evaluation.METHODS]
        result = Fixture(self.root, "tcep", cases, records).run()
        metric = next(r for r in result["seed_summary"] if r["method"] == "RKDS" and r["subset"] == "all" and r["weighting"] == "official_weighted")
        self.assertAlmostEqual(metric["A"], .25)
        self.assertAlmostEqual(metric["C"], .625)
        self.assertAlmostEqual(metric["CA"], .625)
        self.assertAlmostEqual(metric["CA_sd"], .75 / 2**.5)
        subset = next(r for r in result["seed_summary"] if r["method"] == "RKDS" and r["subset"] == "continuous_confirmed" and r["weighting"] == "official_weighted")
        self.assertEqual(subset["N_per_seed"], 1)
        self.assertEqual(subset["weight_per_seed"], 1)

    def test_undefined_seed_ca_not_silently_averaged(self):
        rows = [dict(scope="per_seed", cap="1000", subset="all", weighting="unweighted", method="RKDS",
                     configuration="H13", complete=True, N=2, total_weight=2, A=0, C=0, CA=ca) for ca in (None, 1)]
        result = evaluation.seed_summaries(rows)[0]
        self.assertIsNone(result["CA"])
        self.assertEqual(result["CA_defined_seeds"], 1)

    def test_competition_ranks_exact_not_rounded(self):
        rows = [dict(scope="x", method=str(i), configuration="x", A=value) for i, value in enumerate((.8, .8, .79999))]
        evaluation.add_ranks(rows)
        self.assertEqual([row["A_rank"] for row in rows], [1, 1, 3])

    def test_empty_prepared_checkpoint_is_valid_partial(self):
        row = case()
        fixture = Fixture(self.root, "anm", [row], {row["case_id"]: []})
        result = fixture.run(allow_partial=True)
        self.assertEqual(result["accounting"]["missing_records"], 6)

    def test_scope_contract_rejects_tiny_production_manifest(self):
        fixture = Fixture(self.root, "anm", [case()])
        with self.assertRaisesRegex(evaluation.ValidationError, "case count"):
            evaluation.load_manifest(fixture.root, "anm")

    def tcep_provenance_fixture(self):
        source = self.root / "data/tcep/observations.zip"
        source.parent.mkdir(parents=True)
        raw = ("X,Y\n" + "\n".join(f"{i},{i*i}" for i in range(1, 9)) + "\n").encode()
        with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("pair0001.csv", raw)
        source_hash = hashlib.sha256(raw).hexdigest()
        pair_file = self.root / "data/tcep/pairs.csv"
        pair_file.write_text(f"pair_id,cache_sha256\n1,{source_hash}\n")
        row = case("paircase", domain="tcep", split="tcep", pair_id=1, cap=1000,
                   subsample_seed=1, row_seed=987, weight=1, continuous_continuous="yes",
                   data_file="data/tcep/cache/pair0001.csv", index_file="data/tcep/indices.json.gz")
        entry = dict(case_id=row["case_id"], row_seed=987, original_rows=list(range(1, 9)), data_hash="b"*64)
        with gzip.open(self.root / row["index_file"], "wt") as handle:
            json.dump(dict(schema_version=1, index_base=1, cases={row["case_id"]: entry}), handle)
        fixture = Fixture(self.root, "tcep", [row], extra_frozen=("data/tcep/observations.zip", row["index_file"], "data/tcep/pairs.csv"))
        fixture.modify(row["case_id"], lambda b: b["inputs"][row["case_id"]].update(
            source_data_hash=source_hash, tcep_reference_data_hash="b"*64,
            tcep_reference_hash_match=False, tcep_reference_hash_policy=evaluation.TCEP_HASH_POLICY,
            row_seed=987, presentation_seed=456))
        return fixture, row

    def test_cross_runtime_canonical_hash_difference_is_audited(self):
        fixture, row = self.tcep_provenance_fixture()
        result = fixture.run()
        self.assertTrue(result["complete"])
        self.assertEqual(result["tcep_provenance_audit"]["prepared_cases"], 1)
        self.assertEqual(result["tcep_provenance_audit"]["reference_hash_mismatches"], 1)
        with (self.root / "analysis/tcep/records.csv").open() as handle:
            records = list(csv.DictReader(handle))
        self.assertEqual({r["data_hash"] for r in records}, {"a"*64})
        self.assertEqual({r["tcep_reference_data_hash"] for r in records}, {"b"*64})
        self.assertEqual({r["tcep_reference_hash_match"] for r in records}, {"False"})

    def test_tcep_provenance_tampering_rejected(self):
        fixture, row = self.tcep_provenance_fixture()
        path = fixture.paths[row["case_id"]]
        baseline = json.loads(path.read_text())
        mutations = {"source_data_hash": "c"*64, "tcep_reference_data_hash": "c"*64,
                     "tcep_reference_hash_match": True, "tcep_reference_hash_policy": "ignore",
                     "original_rows": list(range(8, 0, -1)), "row_seed": 988, "presentation_seed": 457}
        for field, value in mutations.items():
            with self.subTest(field=field):
                changed = copy.deepcopy(baseline)
                changed["inputs"][row["case_id"]][field] = value
                path.write_text(json.dumps(changed))
                with self.assertRaises(evaluation.ValidationError):
                    fixture.run()
        path.write_text(json.dumps(baseline))
        fixture.modify(row["case_id"], lambda b: b["records"][0].update(data_hash="d"*64))
        with self.assertRaisesRegex(evaluation.ValidationError, "data_hash differs"):
            fixture.run()

    def test_tcep_raw_source_bytes_remain_strict(self):
        fixture, row = self.tcep_provenance_fixture()
        source = self.root / "data/tcep/observations.zip"
        with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("pair0001.csv", "X,Y\n9,81\n")
        with self.assertRaisesRegex(evaluation.ValidationError, "Frozen file"):
            fixture.run()

    def test_tcep_archived_source_still_matches_pair_hash(self):
        fixture, row = self.tcep_provenance_fixture()
        source = self.root / "data/tcep/observations.zip"
        with zipfile.ZipFile(source, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("pair0001.csv", "X,Y\n9,81\n")
        frozen = evaluation.read_json(self.root / "MANIFEST.json")["files"]
        frozen["data/tcep/observations.zip"] = evaluation.sha256(source)
        with self.assertRaisesRegex(evaluation.ValidationError, "source CSV changed"):
            evaluation.tcep_source_hashes(self.root, {row["case_id"]: row}, frozen)


if __name__ == "__main__":
    unittest.main()
