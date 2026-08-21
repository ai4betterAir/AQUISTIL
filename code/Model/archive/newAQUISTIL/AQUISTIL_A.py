"""AQUISTIL ablation A: LightGBM backbone without history, spatial or event refinement."""

from Model.AQUISTIL import impute_mice as _impute

MODEL_NAME = "AQUISTIL_A"


def impute_mice(data, target_column, input_columns, custom_strategies=None, **kwargs):
    kwargs.update(history_features=False, spatial_features=False, event_refinement=False)
    return _impute(data, target_column, input_columns, custom_strategies=custom_strategies, **kwargs)
