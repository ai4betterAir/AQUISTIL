import logging
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_TEST_OUTPUT_DIR = tempfile.TemporaryDirectory()
os.environ["AQUISTIL_IMPUTATION_OUTPUT_DIR"] = _TEST_OUTPUT_DIR.name

import main
from regional_imputation import run_balanced_regional_task


def tearDownModule():
    _TEST_OUTPUT_DIR.cleanup()


class RegionalParallelismTests(unittest.TestCase):
    def test_worker_logs_are_routed_to_separate_target_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            handlers = main._add_target_log_handlers(["PM10", "NO2"], temp_dir)
            try:
                with main._target_logging_context("PM10"):
                    logging.info("pm10-only-marker")
                with main._target_logging_context("NO2"):
                    logging.info("no2-only-marker")
            finally:
                root_logger = logging.getLogger()
                for handler in handlers:
                    root_logger.removeHandler(handler)
                    handler.close()

            pm10_log = (Path(temp_dir) / "PM10.log").read_text()
            no2_log = (Path(temp_dir) / "NO2.log").read_text()

        self.assertIn("pm10-only-marker", pm10_log)
        self.assertNotIn("no2-only-marker", pm10_log)
        self.assertIn("no2-only-marker", no2_log)
        self.assertNotIn("pm10-only-marker", no2_log)

    def test_cpu_budget_is_divided_between_target_workers(self):
        workers, model_n_jobs = main._resolve_target_parallelism(
            target_count=5,
            task_count=55,
            requested_workers=5,
            requested_model_n_jobs=0,
            available_cpus=40,
        )

        self.assertEqual(workers, 5)
        self.assertEqual(model_n_jobs, 8)

    def test_worker_count_is_bounded_by_targets_tasks_and_cpus(self):
        workers, model_n_jobs = main._resolve_target_parallelism(
            target_count=3,
            task_count=2,
            requested_workers=10,
            requested_model_n_jobs=20,
            available_cpus=4,
        )

        self.assertEqual(workers, 2)
        self.assertEqual(model_n_jobs, 4)

    def test_regions_for_same_target_can_run_concurrently(self):
        workers, model_n_jobs = main._resolve_target_parallelism(
            target_count=2,
            task_count=10,
            requested_workers=10,
            requested_model_n_jobs=0,
            available_cpus=80,
        )

        self.assertEqual(workers, 10)
        self.assertEqual(model_n_jobs, 8)

    def test_regional_task_forwards_parallel_model_controls(self):
        calls = []

        def imputer(data, target, features, custom_strategies=None, **kwargs):
            calls.append(kwargs)
            result = data.copy()
            result[target] = pd.to_numeric(result[target], errors="coerce").fillna(
                pd.to_numeric(result[target], errors="coerce").mean()
            )
            return result

        rows = []
        for site_index, site in enumerate(("SITE_A", "SITE_B")):
            for hour in range(10):
                rows.append(
                    {
                        "DateTime": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=hour),
                        "Region": "Test Region",
                        "Site": site,
                        "PM10": float(hour + site_index),
                        "TEMP": float(20 + hour),
                    }
                )
        data = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_balanced_regional_task(
                regional_data=data,
                region="Test Region",
                target="PM10",
                features=["TEMP"],
                model_name="LightGBM",
                impute_callable=imputer,
                regimes=["random"],
                missingness_levels=[0.2],
                seeds=[42],
                output_root=temp_dir,
                strict_feature_list=True,
                model_n_jobs=3,
            )

            task_root = Path(temp_dir) / "LightGBM" / "Test_Region" / "PM10"
            self.assertTrue((task_root / "metrics_by_site_and_region.csv").is_file())
            self.assertTrue((task_root / "masked_predictions_by_site.csv").is_file())

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["strict_feature_list"])
        self.assertEqual(calls[0]["n_jobs"], 3)
        self.assertFalse(result["metrics"].empty)
        self.assertFalse(result["predictions"].empty)
        self.assertTrue(np.isfinite(result["predictions"]["Imputed"]).all())

    def test_regional_task_resets_index_after_filtering_empty_sites(self):
        received_indexes = []

        def imputer(data, target, features, custom_strategies=None, **kwargs):
            received_indexes.append(data.index.copy())
            result = data.copy()
            result[target] = pd.to_numeric(result[target], errors="coerce").fillna(
                pd.to_numeric(result[target], errors="coerce").mean()
            )
            return result

        rows = []
        for site, has_observations in (("EMPTY_SITE", False), ("VALID_SITE", True)):
            for hour in range(10):
                rows.append(
                    {
                        "DateTime": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=hour),
                        "Region": "Test Region",
                        "Site": site,
                        "PM10": float(hour) if has_observations else np.nan,
                        "TEMP": float(20 + hour),
                    }
                )
        data = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_balanced_regional_task(
                regional_data=data,
                region="Test Region",
                target="PM10",
                features=["TEMP"],
                model_name="LightGBM",
                impute_callable=imputer,
                regimes=["random"],
                missingness_levels=[0.2],
                seeds=[42],
                output_root=temp_dir,
                strict_feature_list=True,
                model_n_jobs=1,
            )

        self.assertEqual(len(received_indexes), 1)
        self.assertTrue(
            received_indexes[0].equals(pd.RangeIndex(start=0, stop=10))
        )


if __name__ == "__main__":
    unittest.main()
