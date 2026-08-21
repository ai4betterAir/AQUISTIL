"""AQUISTIL ablation: disable adaptive gap guardrails only."""

from Model import AQUISTIL as _base

MODEL_NAME = "AQUISTIL_NoAdaptive"


def impute_mice(data, target_column, input_columns, custom_strategies=None, **kwargs):
    kwargs["adaptive_gap_guardrails"] = False
    return _base.impute_mice(
        data,
        target_column,
        input_columns,
        custom_strategies=custom_strategies,
        **kwargs,
    )
