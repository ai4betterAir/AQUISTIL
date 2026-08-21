import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from Model import AQUISTIL  # noqa: E402


class AquistilAblationSwitchTests(unittest.TestCase):
    def test_no_forward_fill_preserves_feature_nans(self):
        frame = pd.DataFrame(
            {
                "lag_1": [10.0, np.nan, np.nan, 20.0],
                "copollutant": [1.0, np.nan, 3.0, np.nan],
            }
        )
        cleaned = AQUISTIL._clean_features(
            frame,
            train_mask=pd.Series([True, True, True, True]),
            groups=np.array(["A", "A", "A", "A"]),
            forward_fill=False,
            median_fill=False,
        )

        self.assertTrue(cleaned.isna().equals(frame.isna()))

    def test_forward_fill_switch_changes_stale_lag_behavior(self):
        frame = pd.DataFrame({"lag_1": [10.0, np.nan, np.nan, 20.0]})
        filled = AQUISTIL._clean_features(
            frame,
            train_mask=pd.Series([True, True, True, True]),
            groups=np.array(["A", "A", "A", "A"]),
            forward_fill=True,
            median_fill=False,
        )
        not_filled = AQUISTIL._clean_features(
            frame,
            train_mask=pd.Series([True, True, True, True]),
            groups=np.array(["A", "A", "A", "A"]),
            forward_fill=False,
            median_fill=False,
        )

        self.assertEqual(filled.loc[1, "lag_1"], 10.0)
        self.assertTrue(pd.isna(not_filled.loc[1, "lag_1"]))

    def test_build_features_can_disable_history_and_event_features(self):
        data = pd.DataFrame(
            {
                "DateTime": pd.date_range("2026-01-01", periods=8, freq="h"),
                "Site": ["A"] * 8,
                "Region": ["R"] * 8,
                "PM10": np.arange(8, dtype=float),
                "NEPH": np.arange(8, dtype=float) / 10,
            }
        )
        original_missing = np.array([False, False, True, True, False, False, False, False])
        features = AQUISTIL._build_features(
            df=data,
            target="PM10",
            feature_columns=["NEPH"],
            original_target=data["PM10"],
            original_missing=original_missing,
            groups=data["Site"].to_numpy(),
            use_history_features=False,
            use_event_features=False,
        )

        self.assertIn("NEPH", features.columns)
        self.assertIn("gap_length", features.columns)
        self.assertFalse(any(str(column).startswith("lag_") for column in features.columns))
        self.assertNotIn("event_score", features.columns)


if __name__ == "__main__":
    unittest.main()
