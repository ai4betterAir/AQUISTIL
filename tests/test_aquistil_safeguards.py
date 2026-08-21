import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "Model"))

import AQUISTIL  # noqa: E402
import AQUISTIL_ExogenousOnly  # noqa: E402
import AQUISTIL_NoAdaptive  # noqa: E402


class AquistilSafeguardTests(unittest.TestCase):
    def test_no_adaptive_disables_only_the_adaptive_guardrail(self):
        with patch.object(
            AQUISTIL_NoAdaptive._base, "impute_mice", return_value="ok"
        ) as imputer:
            AQUISTIL_NoAdaptive.impute_mice(None, "PM10", [])
        forwarded = imputer.call_args.kwargs
        self.assertFalse(forwarded["adaptive_gap_guardrails"])
        self.assertNotIn("regime_aware", forwarded)

    def test_exogenous_only_excludes_derived_target_components(self):
        with patch.object(
            AQUISTIL_ExogenousOnly._base, "impute_mice", return_value="ok"
        ) as imputer:
            AQUISTIL_ExogenousOnly.impute_mice(None, "PM10", [])
        forwarded = imputer.call_args.kwargs
        for switch in (
            "history_features",
            "event_features",
            "event_refinement",
            "adaptive_gap_guardrails",
            "regime_aware",
            "gap_features",
            "uncertainty_models",
        ):
            self.assertFalse(forwarded[switch])

    def test_backbone_matches_lightgbm_baseline_properties(self):
        params = AQUISTIL._lgbm_regressor(42, n_jobs=2).get_params()
        expected = {
        "n_estimators": 60,
        "max_depth": 3,
        "learning_rate": 0.08,
        "num_leaves": 15,
        "min_child_samples": 40,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_lambda": 2.0,
        "random_state": 42,
        "n_jobs": 2,
        }
        self.assertEqual({name: params[name] for name in expected}, expected)


    def test_event_refinement_is_off_for_gap_experiments(self):
        allowed = ("event", "")
        self.assertTrue(AQUISTIL._event_refinement_enabled(True, "event", allowed))
        self.assertTrue(AQUISTIL._event_refinement_enabled(True, "", allowed))
        self.assertFalse(AQUISTIL._event_refinement_enabled(True, "short_gap", allowed))
        self.assertFalse(AQUISTIL._event_refinement_enabled(True, "random", allowed))


    def test_adaptive_blend_requires_supported_spatial_reference(self):
        index = pd.Index([2, 3])
        model_features = pd.DataFrame({"gap_length": [100.0, 100.0]}, index=index)
        raw_reference = pd.DataFrame(
        {"adaptive_spatial": [12.0, 13.0], "adaptive_spatial_support": [0, 0]},
        index=index,
        )
        original = pd.DataFrame(
        {"DateTime": pd.date_range("2020-01-01", periods=4, freq="h")}
        )
        original_y = pd.Series([10.0, 11.0, np.nan, np.nan])
        config = SimpleNamespace(
        AQUISTIL_ADAPTIVE_GAP_GUARDRAILS_ENABLED=True,
        AQUISTIL_ADAPTIVE_BLEND_RULES={"PM2.5": {"long_gap": 0.5}},
        AQUISTIL_ADAPTIVE_UNCERTAINTY_EXTRA_BLEND={},
        AQUISTIL_ADAPTIVE_MIN_SPATIAL_CONTRIBUTORS=1,
        )

        adjusted, weights = AQUISTIL._apply_adaptive_gap_guardrails(
        final_prediction=np.array([20.0, 21.0]),
        X_missing=model_features,
        reference_features=raw_reference,
        original=original,
        target="PM2.5",
        original_y=original_y,
        observed=np.array([True, True, False, False]),
        missing_index=index,
        uncertainty=np.array([1.0, 1.0]),
        regime="long_gap",
        config=config,
        )

        np.testing.assert_array_equal(weights, [0.0, 0.0])
        np.testing.assert_array_equal(adjusted, [20.0, 21.0])

    def test_adaptive_spatial_falls_back_to_other_pooled_sites(self):
        times = pd.to_datetime(["2020-01-01", "2020-01-02"])
        frame = pd.DataFrame(
            {
                "DateTime": np.tile(times, 3),
                "Site": np.repeat(["A", "B", "C"], 2),
                "PM10": [10.0, np.nan, 12.0, 20.0, 14.0, 30.0],
            }
        )
        adaptive, support, inventory = AQUISTIL._adaptive_spatial_aggregate(
            df=frame,
            target="PM10",
            feature_columns=[],
            observed_mask=frame["PM10"].notna().to_numpy(),
            site_name="Test Region",
            config=SimpleNamespace(SITE_COORDINATES={}),
        )

        self.assertAlmostEqual(adaptive.iloc[1], 25.0)
        self.assertEqual(support.iloc[1], 2)
        self.assertTrue(inventory)


    def test_pm10_final_bounds_apply_without_adaptive_blending(self):
        index = pd.Index([3])
        original = pd.DataFrame(
        {"DateTime": pd.date_range("2020-01-01", periods=4, freq="h")}
        )
        original_y = pd.Series([10.0, 12.0, 14.0, np.nan])
        config = SimpleNamespace(
        AQUISTIL_FINAL_CLIP_TARGETS={"PM10": True},
        AQUISTIL_ADAPTIVE_LOWER_QUANTILE=0.01,
        AQUISTIL_ADAPTIVE_UPPER_QUANTILE=0.995,
        AQUISTIL_ADAPTIVE_IQR_FACTOR=1.5,
        AQUISTIL_ADAPTIVE_NONNEGATIVE=True,
        )

        finalized = AQUISTIL._finalize_predictions(
        np.array([10000.0]),
        original,
        "PM10",
        original_y,
        np.array([True, True, True, False]),
        index,
        config,
        )

        self.assertTrue(np.isfinite(finalized[0]))
        self.assertLess(finalized[0], 100.0)

    def test_missing_topology_is_computed_independently_by_site(self):
        target = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan, 10.0, np.nan, 12.0])
        missing = target.isna().to_numpy()
        groups = np.array(["A", "A", "A", "A", "A", "B", "B", "B"])

        topology = AQUISTIL._missing_topology_features(
            target,
            missing,
            groups=groups,
            include_boundary_values=True,
            allow_future_context=True,
        )

        np.testing.assert_array_equal(
            topology["gap_total_length"].to_numpy(), [0, 2, 2, 0, 1, 0, 1, 0]
        )
        np.testing.assert_array_equal(
            topology["gap_position"].to_numpy(), [0, 1, 2, 0, 1, 0, 1, 0]
        )
        np.testing.assert_array_equal(
            topology.loc[[1, 2], "hours_since_previous_target_observation"], [1, 2]
        )
        np.testing.assert_array_equal(
            topology.loc[[1, 2], "hours_until_next_target_observation"], [2, 1]
        )
        np.testing.assert_array_equal(
            topology.loc[[1, 2], "previous_observed_target"], [1.0, 1.0]
        )
        np.testing.assert_array_equal(
            topology.loc[[1, 2], "next_observed_target"], [4.0, 4.0]
        )

    def test_causal_topology_does_not_expose_future_boundary(self):
        target = pd.Series([1.0, np.nan, np.nan, 4.0])
        topology = AQUISTIL._missing_topology_features(
            target,
            target.isna().to_numpy(),
            include_boundary_values=True,
            allow_future_context=False,
        )

        self.assertNotIn("next_observed_target", topology.columns)
        self.assertTrue(
            topology.loc[[1, 2], "hours_until_next_target_observation"].isna().all()
        )
        np.testing.assert_array_equal(
            topology.loc[[1, 2], "distance_to_nearest_observed_boundary"], [1, 2]
        )

    def test_gap_threshold_supports_global_and_pollutant_values(self):
        global_config = SimpleNamespace(AQUISTIL_GAP_EXPERT_MIN_RUN_LENGTH=6)
        target_config = SimpleNamespace(
            AQUISTIL_GAP_EXPERT_MIN_RUN_LENGTH={"PM10": 3, "PM2.5": 12}
        )

        self.assertEqual(AQUISTIL._gap_expert_threshold(global_config, "PM10"), 6)
        self.assertEqual(AQUISTIL._gap_expert_threshold(target_config, "PM10"), 3)
        self.assertEqual(AQUISTIL._gap_expert_threshold(target_config, "PM2.5"), 12)


if __name__ == "__main__":
    unittest.main()
