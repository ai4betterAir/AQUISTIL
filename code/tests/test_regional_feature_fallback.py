import unittest
from unittest import mock

import config_spatial as config
import main


class RegionalFeatureFallbackTests(unittest.TestCase):
    def test_generic_fallback_excludes_target_and_records_provenance(self):
        with (
            mock.patch.object(config, "FALLBACK_TO_INPUT_COLUMNS_WHEN_MISSING", True),
            mock.patch.object(
                config,
                "REGIONAL_GENERIC_FEATURES",
                ["PM2.5", "TEMP", "OZONE", "TEMP", ""],
            ),
        ):
            choice = main._generic_regional_feature_choice("Central_Coast", "PM2.5")

        self.assertEqual(choice["features"], ["TEMP", "OZONE"])
        self.assertEqual(choice["feature_source"], "generic_fallback")
        self.assertEqual(choice["configuration"], "Generic_Local_Feature_Fallback")
        self.assertFalse(choice["strict_progressive"])

    def test_generic_fallback_can_be_disabled(self):
        with mock.patch.object(config, "FALLBACK_TO_INPUT_COLUMNS_WHEN_MISSING", False):
            self.assertEqual(
                main._generic_regional_feature_choice("Central_Coast", "PM2.5"),
                {},
            )
            self.assertEqual(
                main._borrow_stage3_regional_feature_choice(
                    {
                        (main._canon_token("Lower Hunter"), main._canon_token("PM2.5")): {
                            "region": "Lower Hunter",
                            "features": ["TEMP"],
                            "feature_source": "stage3",
                        }
                    },
                    "Central_Coast",
                    "PM2.5",
                ),
                {},
            )

    def test_missing_region_borrows_available_stage3_features(self):
        choices = main._load_progressive_best_features(
            config.PROGRESSIVE_BEST_FEATURES_CSV
        )
        borrowed = main._borrow_stage3_regional_feature_choice(
            choices,
            "Central_Coast",
            "PM2.5",
        )

        lower_hunter = choices[(main._canon_token("Lower Hunter"), main._canon_token("PM2.5"))]
        self.assertEqual(borrowed["feature_source"], "stage3_borrowed")
        self.assertEqual(borrowed["source_region"], "Lower Hunter")
        self.assertEqual(borrowed["features"], lower_hunter["features"])
        self.assertTrue(borrowed["strict_progressive"])

    def test_stage3_choice_remains_strict_and_selected(self):
        choices = main._load_progressive_best_features(
            config.PROGRESSIVE_BEST_FEATURES_CSV
        )
        lower_hunter = choices[(main._canon_token("Lower Hunter"), main._canon_token("PM2.5"))]

        self.assertEqual(lower_hunter["feature_source"], "stage3")
        self.assertTrue(lower_hunter["strict_progressive"])
        self.assertIn("IDW_Spatial_PM25", lower_hunter["features"])


if __name__ == "__main__":
    unittest.main()
