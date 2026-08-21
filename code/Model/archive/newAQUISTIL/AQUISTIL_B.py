"""AQUISTIL ablation B: backbone plus leakage-safe observed target history."""

from Model.AQUISTIL import impute_mice as _impute

MODEL_NAME = "AQUISTIL_B"


def impute_mice(data, target_column, input_columns, custom_strategies=None, **kwargs):
    kwargs.update(history_features=True, spatial_features=False, event_refinement=False)
    return _impute(data, target_column, input_columns, custom_strategies=custom_strategies, **kwargs)
