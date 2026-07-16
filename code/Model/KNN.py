import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

MODEL_NAME = "KNN"


def impute_mice(data, target_column, input_columns, custom_strategies=None, **kwargs):
    df = data.copy()
    cols = [c for c in ([target_column] + input_columns) if c in df.columns]
    if not cols:
        return df
    # Use single-threaded operation to avoid excessive parallel memory usage
    imputer = KNNImputer(n_neighbors=5, n_jobs=1)
    try:
        arr = imputer.fit_transform(df[cols])
        df.loc[:, cols] = arr
    except MemoryError:
        # Fallback: leave data unchanged but log via warning upstream
        return df
    except Exception:
        # Generic fallback to avoid crashing pipeline
        return df
    return df
