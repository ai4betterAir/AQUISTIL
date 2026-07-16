"""
Interpolation standalone model
Imputes missing values using time-based interpolation when DateTime is present,
otherwise uses linear interpolation across rows.
"""

import pandas as pd
import numpy as np

MODEL_NAME = "Interpolation"


def impute_mice(data, target_column, input_columns, custom_strategies=None, **kwargs):
    """Perform interpolation-based imputation.

    Args:
        data (pd.DataFrame): input dataframe
        target_column (str): column to impute
        input_columns (list): additional feature columns (unused here)

    Returns:
        pd.DataFrame: dataframe with imputed values for the requested columns
    """
    df = data.copy()
    cols = [c for c in ([target_column] + input_columns) if c in df.columns]
    if not cols:
        return df

    try:
        # Prefer time interpolation when DateTime-like column exists
        if 'DateTime' in df.columns:
            idx = pd.to_datetime(df['DateTime'], errors='coerce')
            if idx.notna().any():
                df_indexed = df.set_index(idx)
                df_indexed[cols] = df_indexed[cols].interpolate(method='time', limit_direction='both')
                # preserve original order
                df.loc[:, cols] = df_indexed[cols].values
                return df
        if 'datetime' in df.columns:
            idx = pd.to_datetime(df['datetime'], errors='coerce')
            if idx.notna().any():
                df_indexed = df.set_index(idx)
                df_indexed[cols] = df_indexed[cols].interpolate(method='time', limit_direction='both')
                df.loc[:, cols] = df_indexed[cols].values
                return df

        # Fallback to linear interpolation along index
        df[cols] = df[cols].interpolate(method='linear', limit_direction='both')
    except Exception:
        # If interpolation fails, return original dataframe unchanged
        return df

    return df
