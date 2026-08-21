"""AQUISTIL ablation D: backbone, observed history, spatial and event refinement."""

from Model.AQUISTIL import impute_mice as _impute

MODEL_NAME = "AQUISTIL_D"


def impute_mice(data, target_column, input_columns, custom_strategies=None, **kwargs):
    kwargs.update(history_features=True, spatial_features=True, event_refinement=True)
    return _impute(data, target_column, input_columns, custom_strategies=custom_strategies, **kwargs)
