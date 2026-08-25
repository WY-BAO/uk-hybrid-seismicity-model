"""Tests for the Hybrid B probability definition."""

from __future__ import annotations

import sys
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREPARED_INPUT = ROOT / "input" / "p_source.csv"
sys.path.insert(0, str(ROOT / "scripts"))

from _common import (  # noqa: E402
    source_informed_softmax,
    validate_probability_array,
)


class HybridBProbabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if PREPARED_INPUT.is_file():
            table = pd.read_csv(PREPARED_INPUT)
            cls.p_source = table["p_source"].to_numpy(float)
            cls.count = table["count"].to_numpy(int)
            cls.grid_id = table["grid_id"].to_numpy(int)
        else:
            cls.p_source = np.array([0.0, 0.2, 0.3, 0.5])
            cls.count = np.array([1, 2, 3, 4])
            cls.grid_id = np.array([1, 2, 3, 4])

    def test_zero_correction_recovers_source_baseline(self) -> None:
        observed = source_informed_softmax(
            self.p_source, np.zeros_like(self.p_source)
        )
        # p_source is reloaded from decimal CSV, so allow a conservative
        # floating-point round-trip tolerance while retaining exact zeros.
        np.testing.assert_allclose(observed, self.p_source, rtol=0.0, atol=1e-14)

    def test_zero_source_probabilities_are_not_replaced(self) -> None:
        # Raising on divide-by-zero proves the implementation never evaluates
        # log(0); it indexes the positive subset before taking logarithms.
        with np.errstate(divide="raise", invalid="raise"):
            observed = source_informed_softmax(
                self.p_source, np.linspace(-1.0, 1.0, self.p_source.size)
            )
        self.assertTrue(np.array_equal(observed[self.p_source == 0.0], np.zeros_like(observed[self.p_source == 0.0])))

    def test_probability_constraints_for_deterministic_corrections(self) -> None:
        corrections = np.vstack(
            [
                np.zeros_like(self.p_source),
                np.linspace(-3.0, 3.0, self.p_source.size),
                np.linspace(3.0, -3.0, self.p_source.size),
            ]
        )
        probability = source_informed_softmax(self.p_source, corrections)
        error = validate_probability_array(probability, 1e-12)
        self.assertLessEqual(error, 1e-12)

    @unittest.skipUnless(PREPARED_INPUT.is_file(), "requires prepared model input")
    def test_zero_source_cells_define_the_excluded_domain(self) -> None:
        excluded = self.p_source == 0.0
        modelled = ~excluded
        self.assertEqual(
            self.grid_id[excluded].tolist(), [1, 11, 99, 111, 122, 123]
        )
        self.assertEqual(int(excluded.sum()), 6)
        self.assertEqual(int(modelled.sum()), 126)
        self.assertEqual(int(self.count[excluded].sum()), 8)
        self.assertEqual(int(self.count[modelled].sum()), 1005)

    @unittest.skipUnless(PREPARED_INPUT.is_file(), "requires prepared model input")
    def test_stan_data_contains_only_the_model_domain(self) -> None:
        stan_data = json.loads(
            (ROOT / "input" / "stan_data.json").read_text(encoding="utf-8")
        )
        self.assertEqual(int(stan_data["N"]), 126)
        self.assertEqual(len(stan_data["x"]), 126)
        self.assertEqual(len(stan_data["count"]), 126)
        self.assertEqual(sum(stan_data["count"]), 1005)
        self.assertEqual(len(stan_data["p_source"]), 126)
        self.assertTrue(all(value > 0.0 for value in stan_data["p_source"]))


if __name__ == "__main__":
    unittest.main()
