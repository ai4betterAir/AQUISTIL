"""AQUISTIL ablation: disable target-history features only."""

from Model import AQUISTIL as _base

MODEL_NAME = "AQUISTIL_NoHistory"


def impute_mice(data, target_column, input_columns, custom_strategies=None, **kwargs):
    kwargs["history_features"] = False
    kwargs["regime_aware"] = False
    kwargs["log_feature_list"] = True
    return _base.impute_mice(
        data,
        target_column,
        input_columns,
        custom_strategies=custom_strategies,
        **kwargs,
    )
