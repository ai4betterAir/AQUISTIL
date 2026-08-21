"""AQUISTIL ablation: disable target-history and event-derived features."""

from Model import AQUISTIL as _base

MODEL_NAME = "AQUISTIL_NoHistoryNoEvent"


def impute_mice(data, target_column, input_columns, custom_strategies=None, **kwargs):
    kwargs["history_features"] = False
    kwargs["event_features"] = False
    kwargs["event_refinement"] = False
    kwargs["regime_aware"] = False
    kwargs["log_feature_list"] = True
    return _base.impute_mice(
        data,
        target_column,
        input_columns,
        custom_strategies=custom_strategies,
        **kwargs,
    )
