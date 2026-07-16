import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

MODEL_NAME = "Mean"


def impute_mice(data, target_column, input_columns, custom_strategies=None, **kwargs):
    df = data.copy()
    # Impute target using column mean
    if target_column not in df.columns:
        return df
    try:
        mean_val = pd.to_numeric(df[target_column], errors='coerce').mean()
        df[target_column] = df[target_column].fillna(mean_val)
    except Exception:
        # fallback: use SimpleImputer across provided columns
        cols = [c for c in ([target_column] + input_columns) if c in df.columns]
        if not cols:
            return df
        imputer = SimpleImputer(strategy='mean')
        arr = imputer.fit_transform(df[cols])
        df.loc[:, cols] = arr
    return df
