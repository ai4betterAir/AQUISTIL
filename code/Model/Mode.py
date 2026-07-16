import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

MODEL_NAME = "Mode"


def impute_mice(data, target_column, input_columns, custom_strategies=None, **kwargs):
    df = data.copy()
    if target_column not in df.columns:
        return df
    try:
        mode_val = pd.to_numeric(df[target_column], errors='coerce').mode()
        if len(mode_val) > 0:
            df[target_column] = df[target_column].fillna(mode_val.iloc[0])
        else:
            df[target_column] = df[target_column].fillna(0)
    except Exception:
        cols = [c for c in ([target_column] + input_columns) if c in df.columns]
        if not cols:
            return df
        imputer = SimpleImputer(strategy='most_frequent')
        arr = imputer.fit_transform(df[cols])
        df.loc[:, cols] = arr
    return df
