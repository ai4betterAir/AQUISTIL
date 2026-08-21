import importlib.util
from pathlib import Path

import pandas as pd

from aggregate_metrics import write_models_comparison


MODEL_PATH = Path(__file__).parents[1] / "Model" / "BaseLine.py"
SPEC = importlib.util.spec_from_file_location("baseline_model", MODEL_PATH)
BASELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASELINE)


def test_baseline_handles_pooled_duplicate_timestamps_without_changing_rows():
    data = pd.DataFrame({
        "DateTime": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"] * 2),
        "Site": ["A", "A", "B", "B"],
        "PM2.5": [1.0, None, 10.0, None],
    })
    result = BASELINE.impute_mice(data, "PM2.5", [])
    assert result.index.equals(data.index)
    assert len(result) == len(data)
    assert result["PM2.5"].notna().all()
    assert result.loc[0, "PM2.5"] == 1.0
    assert result.loc[2, "PM2.5"] == 10.0


def test_comparison_matches_model_names_case_insensitively(tmp_path):
    rows = []
    for model in ("BaseLine", "Interpolation"):
        rows.append({
            "Region": "R", "Site": "S", "Target": "PM2.5", "Regime": "random",
            "Missingness_Level": 0.1, "Missingness_Percent": 10.0, "Seed": 42,
            "Scope": "Site", "Model": model, "RMSE": 1.0, "RMAE": 1.0,
            "R": 0.5, "R2": 0.25,
        })
    output = tmp_path / "comparison.csv"
    result = write_models_comparison(
        pd.DataFrame(rows), output, ["BaseLine", "interpolation"], preserve_existing=False
    )
    assert "RMSE_interpolation" in result.columns
    assert output.exists()
