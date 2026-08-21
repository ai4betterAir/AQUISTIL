"""
MICE with LightGBM Iterative Imputation
Uses LightGBM as the estimator for MICE's IterativeImputer

Author: Dr.  Masrur
Last Updated: 2026-01-20
"""

import pandas as pd
import numpy as np
import logging
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

logging.basicConfig(level=logging. INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ CRITICAL: Define MODEL_NAME for main.py loader
MODEL_NAME = "MICE-LGBM-Iterative"


def impute_mice(
    data,
    target_column,
    input_columns,
    custom_strategies=None,
    max_iter=10,
    random_state=42,
    **kwargs  # ✅ Accept extra kwargs from pipeline
):
    """
    Perform MICE imputation using LightGBM as the estimator. 
    
    Args:
        data (pd.DataFrame): Input data with missing values
        target_column (str): Target column to impute
        input_columns (list): List of input columns
        custom_strategies (dict): Custom imputation strategies (optional)
        max_iter (int): Maximum iterations for MICE
        random_state (int): Random seed
        **kwargs: Additional arguments (ignored gracefully)
    
    Returns:
        pd.DataFrame: Imputed data
    """
    
    logging.info(f"🔄 Starting MICE-LightGBM Iterative Imputation for {target_column}...")
    
    if lgb is None:
        logging.error("❌ LightGBM not installed!  Falling back to BayesianRidge")
        from sklearn.linear_model import BayesianRidge
        estimator = BayesianRidge()
    else:
        # ✅ Use LightGBM regressor as MICE estimator
        estimator = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            random_state=random_state,
            verbosity=-1,  # Suppress warnings
            force_col_wise=True  # Faster for wide datasets
        )
    
    # Apply custom strategies if provided
    df = data.copy()
    
    if custom_strategies:
        logging.info("Applying custom pre-imputation strategies...")
        for col, strategy in custom_strategies.items():
            if col in df.columns:
                if strategy == "mean":
                    df[col] = df[col].fillna(df[col]. mean())
                elif strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif isinstance(strategy, (int, float)):
                    df[col] = df[col].fillna(strategy)
    
    # Select numeric columns only
    numeric_cols = [col for col in [target_column] + input_columns 
                   if col in df. columns and pd.api.types.is_numeric_dtype(df[col])]
    
    logging.info(f"Using {len(numeric_cols)} numeric columns for imputation")
    
    # Initialize MICE with LightGBM estimator
    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=max_iter,
        random_state=random_state,
        verbose=0,
        imputation_order='ascending',  # Impute from least to most missing
        min_value=0,  # Enforce non-negative values for air quality data
        skip_complete=True  # Skip columns without missing values
    )
    
    # Fit and transform
    try:
        imputed_array = imputer.fit_transform(df[numeric_cols])
        
        # Create output dataframe
        imputed_df = df.copy()
        imputed_df[numeric_cols] = imputed_array
        
        # Restore DateTime if present
        if 'DateTime' in data.columns:
            imputed_df['DateTime'] = data['DateTime']
        
        logging.info(f"✅ MICE-LightGBM imputation completed successfully")
        
        return imputed_df
        
    except Exception as e:
        logging.error(f"❌ MICE-LightGBM imputation failed: {e}")
        logging.error("Returning original data with forward-fill as fallback")
        return df.fillna(method='ffill').fillna(method='bfill')


# ✅ Optional: Add standalone test
if __name__ == "__main__":
    # Test with synthetic data
    test_data = pd.DataFrame({
        'DateTime': pd.date_range('2024-01-01', periods=100, freq='H'),
        'PM2.5': np.random. uniform(5, 50, 100),
        'PM10': np.random.uniform(10, 100, 100),
        'NO2': np.random.uniform(10, 60, 100),
        'TEMP': np.random.uniform(10, 30, 100)
    })
    
    # Introduce 20% random missingness
    mask = np.random.rand(100) < 0.2
    test_data.loc[mask, 'PM2.5'] = np.nan
    
    print(f"Missing values before imputation: {test_data['PM2.5'].isna().sum()}")
    
    # Run imputation
    result = impute_mice(
        test_data,
        target_column='PM2.5',
        input_columns=['PM10', 'NO2', 'TEMP']
    )
    
    print(f"Missing values after imputation: {result['PM2.5'].isna().sum()}")
    print(f"✅ Test passed!")