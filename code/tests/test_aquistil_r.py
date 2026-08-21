import unittest
from unittest import mock

import numpy as np
import pandas as pd

from Model import AQUISTIL_R as model


class AQUISTILRTests(unittest.TestCase):
    def test_project_simplex_and_robust_stack_favour_accurate_expert(self):
        projected = model._project_simplex([1.2, -0.1, 0.4])
        self.assertTrue(np.all(projected >= 0.0))
        self.assertTrue(np.isclose(projected.sum(), 1.0))

        rng = np.random.default_rng(17)
        truth = np.linspace(2.0, 30.0, 180)
        accurate = truth + rng.normal(0.0, 0.15, len(truth))
        biased = truth + 5.0
        unstable = truth + rng.normal(0.0, 4.0, len(truth))
        unstable[0] += 1000.0
        weights = model._robust_convex_weights(
            np.column_stack([accurate, biased, unstable]),
            truth,
            np.full(3, 1.0 / 3.0),
            regularization=0.001,
        )

        self.assertTrue(np.all(weights >= 0.0))
        self.assertTrue(np.isclose(weights.sum(), 1.0))
        self.assertGreater(weights[0], 0.80)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[0], weights[2])

    def test_regime_validation_masks_never_hide_existing_missing_values(self):
        size = 420
        frame = pd.DataFrame(
            {
                "DateTime": pd.date_range("2024-01-01", periods=size, freq="h"),
                "Site": ["A"] * 210 + ["B"] * 210,
                "PM2.5": np.linspace(1.0, 80.0, size),
            }
        )
        frame.loc[[4, 5, 215], "PM2.5"] = np.nan
        originally_missing = frame["PM2.5"].isna().to_numpy()

        medium = model._build_validation_mask(
            frame,
            "PM2.5",
            "medium_gap",
            seed=11,
            validation_fraction=0.15,
            min_points=48,
            max_points=70,
            min_training_points=50,
            event_quantile=0.90,
        )
        self.assertTrue(medium.any())
        self.assertFalse(np.any(medium & originally_missing))
        self.assertTrue(
            all(
                24 <= len(segment) <= 71
                for segment in model._segments(medium, model._group_codes(frame))
            )
        )

        event = model._build_validation_mask(
            frame,
            "PM2.5",
            "event",
            seed=12,
            validation_fraction=0.10,
            min_points=30,
            max_points=50,
            min_training_points=50,
            event_quantile=0.90,
        )
        self.assertTrue(event.any())
        self.assertFalse(np.any(event & originally_missing))
        self.assertGreater(frame.loc[event, "PM2.5"].mean(), frame["PM2.5"].mean())

        small = frame.iloc[:120].copy()
        long_gap = model._build_validation_mask(
            small,
            "PM2.5",
            "long_gap",
            seed=13,
            validation_fraction=0.20,
            min_points=48,
            max_points=50,
            min_training_points=80,
            event_quantile=0.90,
        )
        self.assertLessEqual(int(long_gap.sum()), 40)
        self.assertGreaterEqual(int((small["PM2.5"].notna().to_numpy() & ~long_gap).sum()), 80)

    def test_impute_preserves_observations_and_survives_failed_expert(self):
        size = 240
        proxy = 12.0 + 3.0 * np.sin(np.arange(size) / 12.0)
        original_truth = proxy.copy()
        frame = pd.DataFrame(
            {
                "DateTime": pd.date_range("2024-02-01", periods=size, freq="h"),
                "Site": "TEST",
                "proxy": proxy,
                "PM2.5": original_truth,
            },
            index=np.arange(1000, 1000 + size),
        )
        real_missing = np.zeros(size, dtype=bool)
        real_missing[75:93] = True
        real_missing[[130, 181]] = True
        frame.iloc[np.flatnonzero(real_missing), frame.columns.get_loc("PM2.5")] = np.nan
        observed_before = frame.loc[~real_missing, "PM2.5"].copy()
        calls = []

        def imputer_for(name):
            if name == "Fail":
                def failed(data, target, input_columns, custom_strategies=None, **kwargs):
                    calls.append((name, data[target].isna().to_numpy().copy()))
                    raise RuntimeError("intentional test failure")

                return failed

            offset = {"Good": 0.0, "Biased": 6.0}[name]

            def imputer(data, target, input_columns, custom_strategies=None, **kwargs):
                calls.append((name, data[target].isna().to_numpy().copy()))
                result = data.copy()
                missing = pd.to_numeric(result[target], errors="coerce").isna()
                prediction = pd.to_numeric(result["proxy"], errors="coerce") + offset
                result.loc[missing, target] = prediction.loc[missing]
                return result

            return imputer

        with mock.patch.object(model, "_load_imputer", side_effect=imputer_for):
            result = model.impute_mice(
                frame,
                "PM2.5",
                ["proxy"],
                missingness_regime="short_gap",
                random_state=31,
                aquistil_r_base_models=("Good", "Biased", "Fail"),
                aquistil_r_validation_folds=2,
                aquistil_r_validation_fraction=0.12,
                aquistil_r_min_validation_points=24,
                aquistil_r_max_validation_points=40,
                aquistil_r_min_training_points=40,
                aquistil_r_min_route_rows=10,
                aquistil_r_min_expert_coverage=0.80,
                aquistil_r_prior_strength=1.0,
                aquistil_r_regularization=0.001,
                aquistil_r_huber_delta=1.5,
                aquistil_r_event_quantile=0.90,
                aquistil_r_disagreement_blend=0.20,
                aquistil_r_event_disagreement_blend=0.10,
                aquistil_r_disagreement_start=0.35,
                aquistil_r_disagreement_full=1.25,
                aquistil_r_lower_quantile=0.001,
                aquistil_r_upper_quantile=0.999,
                aquistil_r_bound_iqr_factor=3.0,
                aquistil_r_lower_bound=None,
                aquistil_r_upper_bound=None,
                aquistil_r_add_diagnostics=True,
                aquistil_r_save_components=True,
            )

        pd.testing.assert_series_equal(result.loc[~real_missing, "PM2.5"], observed_before)
        self.assertTrue(result.loc[real_missing, "PM2.5"].notna().all())
        self.assertLess(
            np.mean(np.abs(result.loc[real_missing, "PM2.5"] - original_truth[real_missing])),
            0.5,
        )
        self.assertTrue(result.index.equals(frame.index))
        self.assertTrue(
            result["PM2.5_AQUISTIL_R_Uncertainty90"].loc[real_missing].notna().all()
        )
        self.assertTrue(
            result["PM2.5_AQUISTIL_R_Confidence"].loc[real_missing].between(0.0, 1.0).all()
        )
        self.assertTrue(
            (result.loc[real_missing, "PM2.5_AQUISTIL_R_Weight_Fail"] == 0.0).all()
        )

        # Every call, including synthetic OOF fits and the final fit, must
        # preserve the true missing mask. Validation can only add missing rows.
        self.assertTrue(calls)
        self.assertTrue(all(np.all(call_mask[real_missing]) for _, call_mask in calls))

    def test_all_failed_experts_use_groupwise_temporal_fallback(self):
        size = 100
        truth = pd.Series(np.linspace(4.0, 20.0, size))
        frame = pd.DataFrame(
            {
                "DateTime": pd.date_range("2024-03-01", periods=size, freq="h"),
                "Site": ["A"] * 50 + ["B"] * 50,
                "proxy": truth * 2.0,
                "PM2.5": truth,
            }
        )
        missing = np.zeros(size, dtype=bool)
        missing[[0, 1, 2, 25, 50, 51, 75]] = True
        frame.loc[missing, "PM2.5"] = np.nan

        with mock.patch.object(model, "_load_imputer", side_effect=ImportError("missing")):
            result = model.impute_mice(
                frame,
                "PM2.5",
                ["proxy"],
                missingness_regime="random",
                aquistil_r_base_models=("Fail",),
                aquistil_r_validation_folds=0,
                aquistil_r_min_training_points=20,
                aquistil_r_disagreement_blend=0.20,
                aquistil_r_event_disagreement_blend=0.10,
                aquistil_r_disagreement_start=0.35,
                aquistil_r_disagreement_full=1.25,
                aquistil_r_lower_quantile=0.001,
                aquistil_r_upper_quantile=0.999,
                aquistil_r_bound_iqr_factor=3.0,
                aquistil_r_lower_bound=None,
                aquistil_r_upper_bound=None,
                aquistil_r_add_diagnostics=True,
            )

        self.assertTrue(result.loc[missing, "PM2.5"].notna().all())
        pd.testing.assert_series_equal(result.loc[~missing, "PM2.5"], frame.loc[~missing, "PM2.5"])
        self.assertTrue((result.loc[missing, "PM2.5_AQUISTIL_R_Confidence"] == 0.0).all())


if __name__ == "__main__":
    unittest.main()
