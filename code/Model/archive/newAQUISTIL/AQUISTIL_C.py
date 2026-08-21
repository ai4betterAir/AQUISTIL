"""AQUISTIL ablation C: backbone plus observed history and spatial features."""

from Model.AQUISTIL import impute_mice as _impute

MODEL_NAME = "AQUISTIL_C"


def impute_mice(data, target_column, input_columns, custom_strategies=None, **kwargs):
    kwargs.update(history_features=True, spatial_features=True, event_refinement=False)
    return _impute(data, target_column, input_columns, custom_strategies=custom_strategies, **kwargs)
