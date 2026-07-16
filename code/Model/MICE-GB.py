import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import GradientBoostingRegressor

MODEL_NAME = "MICE-GB"


def impute_mice(data, target_column, input_columns, custom_strategies=None, **kwargs):
    df = data.copy()
    cols = [c for c in ([target_column] + input_columns) if c in df.columns]
    if not cols:
        return df
    try:
        estimator = GradientBoostingRegressor(n_estimators=50, random_state=42)
        imputer = IterativeImputer(estimator=estimator, max_iter=10, random_state=42)
        arr = imputer.fit_transform(df[cols])
        df.loc[:, cols] = arr
    except Exception:
        return df
    return df
