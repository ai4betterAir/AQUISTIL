import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

from main import _reindex_to_complete_hourly_grid  # noqa: E402
from missingness_regimes import GAP_REGIME_BOUNDS  # noqa: E402
from regional_imputation import _balanced_region_mask  # noqa: E402


def _site_frame(site, n_hours, target="PM10", natural_missing=None):
    frame = pd.DataFrame(
        {
            "DateTime": pd.date_range("2026-01-01", periods=n_hours, freq="h"),
            "Region": "Synthetic",
            "Site": site,
            target: np.arange(n_hours, dtype=float) + 1.0,
            "PM2.5": np.arange(n_hours, dtype=float) / 2.0,
            "NEPH": np.arange(n_hours, dtype=float) / 3.0,
        }
    )
    if natural_missing:
        for start, stop in natural_missing:
            frame.loc[start:stop - 1, target] = np.nan
    return frame


class SyntheticRegionalQualityTests(unittest.TestCase):
    def test_hourly_reindex_inserts_missing_timestamps(self):
        frame = pd.DataFrame(
            {
                "DateTime": pd.to_datetime(
                    ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 04:00"]
                ),
                "PM10": [1.0, 2.0, 5.0],
            }
        )
        hourly = _reindex_to_complete_hourly_grid(frame, "unit-test")

        self.assertEqual(len(hourly), 5)
        self.assertTrue(hourly["DateTime"].diff().dropna().eq(pd.Timedelta(hours=1)).all())
        self.assertEqual(int(hourly["PM10"].isna().sum()), 2)

    def test_per_site_missingness_uses_percentage_not_smallest_absolute_n(self):
        data = pd.concat(
            [_site_frame("Short", 500), _site_frame("Long", 1500)],
            ignore_index=True,
        )

        mask, diagnostics, exclusions = _balanced_region_mask(
            data, "PM10", "random", 0.30, seed=13, region="Synthetic"
        )

        self.assertFalse(exclusions)
        by_site = mask.groupby(data["Site"]).sum().to_dict()
        self.assertEqual(by_site["Short"], 150)
        self.assertEqual(by_site["Long"], 450)
        self.assertAlmostEqual(diagnostics["Short"]["Achieved_Missingness"], 0.30)
        self.assertAlmostEqual(diagnostics["Long"]["Achieved_Missingness"], 0.30)

    def test_gap_regimes_are_effectively_pure_at_requested_levels(self):
        data = pd.concat(
            [
                _site_frame("A", 6000, natural_missing=[(100, 105), (4000, 4010)]),
                _site_frame("B", 7500, natural_missing=[(200, 202), (6200, 6210)]),
            ],
            ignore_index=True,
        )
        for regime, bounds in GAP_REGIME_BOUNDS.items():
            for fraction in (0.05, 0.10, 0.20, 0.30):
                with self.subTest(regime=regime, fraction=fraction):
                    _, diagnostics, exclusions = _balanced_region_mask(
                        data, "PM10", regime, fraction, seed=29, region="Synthetic"
                    )
                    self.assertFalse(exclusions)
                    for site_diag in diagnostics.values():
                        self.assertEqual(site_diag["Effective_Gap_Purity"], 1.0)
                        self.assertEqual(site_diag["Out_of_Regime_Effective_Points"], 0)
                        self.assertGreater(site_diag["Achieved_Missingness"], 0)
                        self.assertLess(abs(site_diag["Achieved_Missingness"] - fraction), 0.02)
                        self.assertGreaterEqual(site_diag["Min_Effective_Gap"], bounds[0])
                        self.assertLessEqual(site_diag["Max_Effective_Gap"], bounds[1])

    def test_all_nan_target_station_is_target_specific_exclusion(self):
        good = _site_frame("Good", 2000)
        bad = _site_frame("Bad", 2000)
        bad["PM10"] = np.nan
        data = pd.concat([good, bad], ignore_index=True)

        mask, diagnostics, exclusions = _balanced_region_mask(
            data, "PM10", "medium_gap", 0.10, seed=42, region="Synthetic"
        )

        self.assertTrue(mask.loc[data["Site"].eq("Good")].any())
        self.assertIn("Good", diagnostics)
        self.assertEqual(len(exclusions), 1)
        self.assertEqual(exclusions[0]["Site"], "Bad")
        self.assertIn("entirely NaN", exclusions[0]["Reason"])


if __name__ == "__main__":
    unittest.main()
