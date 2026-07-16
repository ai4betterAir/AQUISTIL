import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

MODEL_NAME = "Median"


def impute_mice(data, target_column, input_columns, custom_strategies=None, **kwargs):
    df = data.copy()
    if target_column not in df.columns:
        return df
    try:
        med = pd.to_numeric(df[target_column], errors='coerce').median()
        df[target_column] = df[target_column].fillna(med)
    except Exception:
        cols = [c for c in ([target_column] + input_columns) if c in df.columns]
        if not cols:
            return df
        imputer = SimpleImputer(strategy='median')
        arr = imputer.fit_transform(df[cols])
        df.loc[:, cols] = arr
    return df
