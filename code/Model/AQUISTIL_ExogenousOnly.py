"""AQUISTIL ablation: exogenous, spatial, calendar and site features only."""

from Model import AQUISTIL as _base

MODEL_NAME = "AQUISTIL_ExogenousOnly"


def impute_mice(data, target_column, input_columns, custom_strategies=None, **kwargs):
    kwargs["history_features"] = False
    kwargs["event_features"] = False
    kwargs["event_refinement"] = False
    kwargs["adaptive_gap_guardrails"] = False
    kwargs["regime_aware"] = False
    kwargs["gap_features"] = False
    kwargs["uncertainty_models"] = False
    kwargs["log_feature_list"] = True
    return _base.impute_mice(
        data,
        target_column,
        input_columns,
        custom_strategies=custom_strategies,
        **kwargs,
    )
