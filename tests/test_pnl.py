"""Fast PNL contract tests; the official training loop is not run here."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("test_pnl_adapter", ROOT / "src/methods/pnl.py")
PNL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PNL)


class PNLTests(unittest.TestCase):
    def test_official_source_hash(self):
        self.assertEqual(PNL.verify_source(), PNL.SOURCE_SHA256)
        self.assertTrue((PNL.SOURCE_PATH.parent / "LICENSE").is_file())

    def test_direction_and_exact_ties(self):
        for p, q, expected in [(0.02, 0.01, 1), (0.1, 0.9, 2),
                                (0, 0, 0), (1, 1, 0), (1e-15, 0, 1)]:
            self.assertEqual(PNL.decision(p, q), expected)

    def test_invalid_pvalues_rejected(self):
        for bad in (float("nan"), float("inf"), -0.1, 1.1, [0.2, 0.3]):
            with self.assertRaises(ValueError):
                PNL.decision(bad, .5)

    def test_official_call_no_overrides_or_input_changes(self):
        pair = np.array([[100., 5.], [100., 5.], [300., 4.]], dtype=np.float64)
        original = pair.copy()
        model = mock.Mock()
        model.cause_or_effect.return_value = (np.array([.02], dtype=np.float32), np.array([.01], dtype=np.float32))
        official = SimpleNamespace(PNL=mock.Mock(return_value=model))
        with mock.patch.object(PNL, "check_environment"), mock.patch.object(PNL, "_official", return_value=official):
            result = PNL.run_pair(pair, 123456)
        official.PNL.assert_called_once_with()
        model.cause_or_effect.assert_called_once()
        x, y = model.cause_or_effect.call_args.args
        np.testing.assert_array_equal(x, original[:, [0]])
        np.testing.assert_array_equal(y, original[:, [1]])
        np.testing.assert_array_equal(pair, original)
        self.assertEqual(x.dtype, original.dtype)
        self.assertEqual(result["prediction"], 1)
        self.assertFalse(result["supplied_pnl_seed_used_by_algorithm"])
        self.assertFalse(result["official_returns_direction"])
        self.assertFalse(result["error"])
        self.assertNotIn("alpha", result)

    def test_failures_are_not_successful_abstentions(self):
        for error, status in [(ValueError("test failure"), "algorithm_error"),
                              (TimeoutError("test timeout"), "timeout")]:
            with mock.patch.object(PNL, "check_environment", side_effect=error):
                result = PNL.run_pair(np.ones((3, 2)), 1)
            self.assertEqual(result["status"], status)
            self.assertEqual(result["prediction"], 0)
            self.assertIsNone(result["pforward"])
            self.assertTrue(result["error"])


if __name__ == "__main__":
    unittest.main()
