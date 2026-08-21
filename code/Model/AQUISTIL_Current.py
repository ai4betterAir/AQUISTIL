"""Pre-router AQUISTIL used as the development-control implementation."""

from Model import AQUISTIL as _base

MODEL_NAME = "AQUISTIL_Current"


def impute_mice(data, target_column, input_columns, custom_strategies=None, **kwargs):
    kwargs["regime_aware"] = False
    return _base.impute_mice(
        data,
        target_column,
        input_columns,
        custom_strategies=custom_strategies,
        **kwargs,
    )
