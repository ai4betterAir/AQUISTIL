import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from missingness_regimes import (  # noqa: E402
    GAP_REGIME_BOUNDS,
    apply_missingness,
    effective_gap_mask_diagnostics,
    gap_mask_diagnostics,
)


class CleanGapMaskTests(unittest.TestCase):
    def test_gap_masks_are_regime_pure(self):
        for regime in sorted(GAP_REGIME_BOUNDS):
            with self.subTest(regime=regime):
                frame = pd.DataFrame({"target": np.arange(5000, dtype=float)})
                _, mask = apply_missingness(frame, "target", regime, frac=0.2, seed=42)
                diagnostics = gap_mask_diagnostics(
                    mask, regime, requested_fraction=0.2, observed_count=len(frame)
                )
                effective = effective_gap_mask_diagnostics(
                    mask,
                    np.zeros(len(frame), dtype=bool),
                    regime,
                    requested_fraction=0.2,
                    observed_count=len(frame),
                )

                self.assertTrue(mask.any())
                self.assertEqual(diagnostics["Gap_Purity"], 1.0)
                self.assertEqual(diagnostics["Out_of_Regime_Points"], 0)
                self.assertEqual(effective["Effective_Gap_Purity"], 1.0)
                lower, upper = GAP_REGIME_BOUNDS[regime]
                self.assertTrue(
                    lower <= diagnostics["Min_Gap"] <= diagnostics["Max_Gap"] <= upper
                )

    def test_gap_masks_keep_observed_anchors_around_natural_missingness(self):
        values = np.full(80, np.nan)
        values[10:40] = np.arange(30, dtype=float)
        frame = pd.DataFrame({"target": values})

        _, mask = apply_missingness(frame, "target", "short_gap", frac=0.5, seed=7)
        masked_positions = np.flatnonzero(mask.to_numpy(dtype=bool))

        self.assertTrue(masked_positions.size)
        self.assertFalse(mask.iloc[10])
        self.assertFalse(mask.iloc[39])
        effective = effective_gap_mask_diagnostics(
            mask, pd.isna(values), "short_gap", requested_fraction=0.5, observed_count=30
        )
        self.assertEqual(effective["Effective_Gap_Purity"], 1.0)

    def test_long_gap_does_not_fall_back_to_random_points(self):
        values = np.full(300, np.nan)
        values[0:50] = 1.0
        values[100:150] = 1.0
        values[200:250] = 1.0
        frame = pd.DataFrame({"target": values})

        _, mask = apply_missingness(frame, "target", "long_gap", frac=0.3, seed=42)

        self.assertFalse(mask.any())


if __name__ == "__main__":
    unittest.main()
