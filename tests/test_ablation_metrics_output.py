import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

import config_spatial as config  # noqa: E402
import main  # noqa: E402
from tools import summarize_aquistil_ablation  # noqa: E402


class AblationMetricsOutputTests(unittest.TestCase):
    def _metrics(self):
        models = ["AQUISTIL"] + list(config.AQUISTIL_ABLATION_MODELS) + ["LightGBM"]
        rows = []
        for position, model in enumerate(models):
            rows.append(
                {
                    "Region": "Test Region",
                    "Site": "ALL",
                    "Target": "PM10",
                    "Model": model,
                    "Regime": "long_gap",
                    "Missingness_Level": 0.1,
                    "Missingness_Percent": 10.0,
                    "Seed": 13,
                    "Scope": "Region_Micro",
                    "RMSE": 5.0 + position,
                    "RMAE": 0.5 + position,
                    "R": 0.8 - position * 0.01,
                    "R2": 0.6 - position * 0.01,
                }
            )
        return pd.DataFrame(rows)

    def test_dedicated_ablation_outputs_are_filtered_and_upserted(self):
        metrics = self._metrics()
        with tempfile.TemporaryDirectory() as directory:
            merged = main._upsert_aquistil_ablation_metrics(metrics, directory)

            self.assertEqual(set(merged["Model"]), set(config.AQUISTIL_ABLATION_MODELS))
            self.assertEqual(len(merged), len(config.AQUISTIL_ABLATION_MODELS))
            self.assertTrue((Path(directory) / "aquistil_ablation_metrics.csv").is_file())
            self.assertTrue(
                (Path(directory) / "aquistil_ablation_metrics_PM10.csv").is_file()
            )

            updated = metrics.copy()
            updated.loc[
                updated["Model"].eq("AQUISTIL_NoHistory"), "RMSE"
            ] = 123.0
            merged = main._upsert_aquistil_ablation_metrics(updated, directory)
            self.assertEqual(len(merged), len(config.AQUISTIL_ABLATION_MODELS))
            value = merged.loc[
                merged["Model"].eq("AQUISTIL_NoHistory"), "RMSE"
            ].iloc[0]
            self.assertEqual(value, 123.0)

            standard_metrics = main._without_aquistil_ablations(metrics)
            self.assertEqual(set(standard_metrics["Model"]), {"AQUISTIL", "LightGBM"})
            main._upsert_regional_metrics_by_target(standard_metrics, directory)
            paper_metrics = pd.read_csv(
                Path(directory) / "regional_pooled_metrics_PM10.csv"
            )
            self.assertEqual(set(paper_metrics["Model"]), {"AQUISTIL", "LightGBM"})

            comparison = main._write_aquistil_ablation_comparison(
                standard_metrics, directory
            )
            self.assertEqual(len(comparison), 1)
            for model in ["AQUISTIL"] + list(config.AQUISTIL_ABLATION_MODELS) + [
                "LightGBM"
            ]:
                self.assertIn(f"RMSE_{model}", comparison.columns)

    def test_development_ablation_mode_persists_all_assessment_models(self):
        metrics = self._metrics()
        with tempfile.TemporaryDirectory() as directory:
            original_models = config.MODELS_TO_RUN
            original_directory = config.AQUISTIL_ABLATION_METRICS_DIRECTORY
            try:
                config.MODELS_TO_RUN = ["AQUISTIL_NoAdaptive"]
                config.AQUISTIL_ABLATION_METRICS_DIRECTORY = directory
                merged = main._upsert_aquistil_ablation_metrics(metrics, directory)
            finally:
                config.MODELS_TO_RUN = original_models
                config.AQUISTIL_ABLATION_METRICS_DIRECTORY = original_directory

            self.assertEqual(
                set(merged["Model"]),
                {"AQUISTIL", *config.AQUISTIL_ABLATION_MODELS, "LightGBM"},
            )

    def test_resume_requires_the_exact_configured_grid(self):
        rows = []
        for scope, site in (
            ("Site", "SITE A"),
            ("Region_Macro", "ALL"),
            ("Region_Micro", "ALL"),
        ):
            for regime in ("random", "event"):
                for level in (0.05, 0.10):
                    for seed in (13, 29):
                        rows.append(
                            {
                                "Region": "Test Region",
                                "Site": site,
                                "Target": "PM10",
                                "Model": "AQUISTIL",
                                "Regime": regime,
                                "Missingness_Level": level,
                                "Missingness_Percent": level * 100,
                                "Seed": seed,
                                "Scope": scope,
                                "RMSE": 1.0,
                            }
                        )
        metrics = pd.DataFrame(rows)
        originals = (
            config.MISSINGNESS_REGIMES,
            config.MISSINGNESS_LEVELS,
            config.REGIONAL_EVALUATION_SEEDS,
        )
        try:
            config.MISSINGNESS_REGIMES = ["random", "event"]
            config.MISSINGNESS_LEVELS = [0.05, 0.10]
            config.REGIONAL_EVALUATION_SEEDS = [13, 29]
            self.assertTrue(
                main._ablation_task_is_complete(
                    metrics, "Test_Region", "PM10", "AQUISTIL"
                )
            )
            self.assertFalse(
                main._ablation_task_is_complete(
                    metrics.iloc[:-1], "Test Region", "PM10", "AQUISTIL"
                )
            )
        finally:
            (
                config.MISSINGNESS_REGIMES,
                config.MISSINGNESS_LEVELS,
                config.REGIONAL_EVALUATION_SEEDS,
            ) = originals

    def test_all_scope_summary_keeps_each_scope_separate(self):
        metrics = pd.concat(
            [
                self._metrics().assign(Scope=scope)
                for scope in ("Site", "Region_Macro", "Region_Micro")
            ],
            ignore_index=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_dir = root / "Metrics"
            main._upsert_aquistil_ablation_metrics(metrics, str(metrics_dir))
            main._upsert_regional_metrics_by_target(
                main._without_aquistil_ablations(metrics), str(metrics_dir)
            )

            summary = summarize_aquistil_ablation.build_summary(root, scope="All")

            self.assertEqual(
                set(summary["Scope"]), {"Site", "Region_Macro", "Region_Micro"}
            )
            self.assertEqual(
                len(summary),
                3 * (len(config.AQUISTIL_ABLATION_MODELS) + 1),
            )
            written = summarize_aquistil_ablation.write_summary_files(
                root, scope="All"
            )
            self.assertEqual(
                set(written), {"All", "Site", "Region_Macro", "Region_Micro"}
            )
            self.assertTrue(all(path.is_file() for path in written.values()))

    def test_incomplete_ablation_source_is_rejected(self):
        metrics = self._metrics().loc[
            lambda frame: ~frame["Model"].eq("AQUISTIL_NoFFill")
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_dir = root / "Metrics"
            main._upsert_aquistil_ablation_metrics(metrics, str(metrics_dir))
            main._upsert_regional_metrics_by_target(
                main._without_aquistil_ablations(metrics), str(metrics_dir)
            )

            with self.assertRaisesRegex(ValueError, "AQUISTIL_NoFFill"):
                summarize_aquistil_ablation.build_summary(
                    root, scope="Region_Micro"
                )

    def test_ablation_only_resume_can_finalize_from_persisted_baselines(self):
        metrics = pd.concat(
            [
                self._metrics().assign(Scope=scope)
                for scope in ("Site", "Region_Macro", "Region_Micro")
            ],
            ignore_index=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_dir = root / "Metrics"
            main._upsert_aquistil_ablation_metrics(metrics, str(metrics_dir))
            standard = main._without_aquistil_ablations(metrics)
            standard.to_csv(metrics_dir / "regional_pooled_metrics.csv", index=False)

            original_models = config.MODELS_TO_RUN
            original_directory = config.AQUISTIL_ABLATION_METRICS_DIRECTORY
            try:
                config.MODELS_TO_RUN = ["AQUISTIL_NoAdaptive"]
                config.AQUISTIL_ABLATION_METRICS_DIRECTORY = str(metrics_dir)
                main._finalize_aquistil_ablation_outputs(
                    str(metrics_dir), str(root)
                )
            finally:
                config.MODELS_TO_RUN = original_models
                config.AQUISTIL_ABLATION_METRICS_DIRECTORY = original_directory

            self.assertTrue(
                (metrics_dir / "aquistil_ablation_comparison.csv").is_file()
            )
            self.assertTrue(
                (metrics_dir / "aquistil_ablation_summary_all_scopes.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
