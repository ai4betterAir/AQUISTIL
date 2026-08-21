import sys
import unittest
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from regional_imputation import _mask_sha256  # noqa: E402
from tools import analyze_frozen_validation as analysis  # noqa: E402
import config_spatial as config  # noqa: E402
import main  # noqa: E402


class FrozenValidationTests(unittest.TestCase):
    def _metrics(self, lightgbm_mask="abc"):
        rows = []
        for model, rmse, mae, correlation, nse, mask in (
            ("AQUISTIL", 2.0, 1.0, 0.9, 0.8, "abc"),
            ("LightGBM", 3.0, 1.5, 0.8, 0.6, lightgbm_mask),
        ):
            rows.append(
                {
                    "Region": "Lower Hunter",
                    "Site": "ALL",
                    "Target": "PM10",
                    "Regime": "random",
                    "Missingness_Level": 0.05,
                    "Missingness_Percent": 5.0,
                    "Seed": 13,
                    "Scope": "Region_Micro",
                    "Model": model,
                    "Mask_SHA256": mask,
                    "N_Masked": 100,
                    "N_Valid": 100,
                    "RMSE": rmse,
                    "MAE": mae,
                    "R": correlation,
                    "NSE": nse,
                }
            )
        return pd.DataFrame(rows)

    def test_mask_fingerprint_is_stable_and_sensitive(self):
        data = pd.DataFrame(
            {
                "Site": ["A", "A", "B"],
                "DateTime": pd.date_range("2026-01-01", periods=3, freq="h"),
            }
        )
        first = _mask_sha256(data, pd.Series([True, False, True]))
        second = _mask_sha256(data.copy(), pd.Series([True, False, True]))
        changed = _mask_sha256(data, pd.Series([False, True, True]))
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_paired_delta_uses_aquistil_minus_lightgbm(self):
        pairs = analysis.build_pairs(self._metrics())
        self.assertEqual(pairs.loc[0, "Delta_RMSE"], -1.0)
        self.assertEqual(pairs.loc[0, "Delta_MAE"], -0.5)
        self.assertAlmostEqual(pairs.loc[0, "Delta_R"], 0.1)
        summary = analysis.summarize_pairs(pairs, iterations=1000, seed=7)
        self.assertEqual(summary.loc[0, "Delta_RMSE_Mean"], -1.0)
        self.assertEqual(summary.loc[0, "AQUISTIL_RMSE_Win_Percent"], 100.0)

    def test_pairing_rejects_different_masks(self):
        with self.assertRaisesRegex(ValueError, "identical masks"):
            analysis.build_pairs(self._metrics(lightgbm_mask="different"))

    def test_frozen_stage3_features_cover_every_validation_task(self):
        choices = main._load_progressive_best_features_for_targets(["PM10", "PM2.5"])
        expected = {
            (main._canon_token(region), main._canon_token(target))
            for region in config.HELD_OUT_VALIDATION_REGIONS
            for target in ("PM10", "PM2.5")
        }
        self.assertEqual(set(choices), expected)
        lower_hunter_pm10 = choices[
            (main._canon_token("Lower Hunter"), main._canon_token("PM10"))
        ]
        self.assertEqual(lower_hunter_pm10["feature_source"], "stage3")
        self.assertIn("IDW_Spatial_PM25", lower_hunter_pm10["features"])


if __name__ == "__main__":
    unittest.main()
