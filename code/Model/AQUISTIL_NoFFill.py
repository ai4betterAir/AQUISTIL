"""AQUISTIL ablation: let LightGBM consume feature NaNs without forward-fill."""

from Model import AQUISTIL as _base

MODEL_NAME = "AQUISTIL_NoFFill"


def impute_mice(data, target_column, input_columns, custom_strategies=None, **kwargs):
    kwargs["forward_fill_features"] = False
    kwargs["median_fill_features"] = False
    kwargs["regime_aware"] = False
    return _base.impute_mice(
        data,
        target_column,
        input_columns,
        custom_strategies=custom_strategies,
        **kwargs,
    )
