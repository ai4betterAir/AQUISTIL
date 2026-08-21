"""AQUISTIL ablation: remove AQUISTIL-specific features and use baseline LightGBM."""

from Model import LightGBM as _lightgbm

MODEL_NAME = "AQUISTIL_NoAQUISTILFeatures"


def impute_mice(data, target_column, input_columns, custom_strategies=None, **kwargs):
    kwargs["log_feature_list"] = True
    return _lightgbm.impute_mice(
        data,
        target_column,
        input_columns,
        custom_strategies=custom_strategies,
        **kwargs,
    )
