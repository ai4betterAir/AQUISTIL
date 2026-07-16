from sklearn.experimental import enable_iterative_imputer
from sklearn. impute import IterativeImputer
import pandas as pd
import numpy as np
import logging
from spatial import prepare_spatial_temporal_data

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the model name
MODEL_NAME = "MICEV2"

def impute_mice(data, target_column, input_columns, max_iter=5, random_state=42, tol=0.01, custom_strategies=None, spatial_config=None):
    """
    Perform MICE imputation on the given data with support for custom strategies and spatial-temporal features. 

    Args:
        data (pd.DataFrame): Input data with missing values. 
        target_column (str): The target column to impute.
        input_columns (list): List of input columns for the imputation model.
        max_iter (int): Maximum number of iterations for the imputer.
        random_state (int): Random seed for reproducibility. 
        tol (float): Tolerance for convergence.
        custom_strategies (dict): Dictionary of column-specific imputation strategies.
            Example: {"column1": "mean", "column2":  "median", "column3": 0}
        spatial_config (dict): Configuration for spatial-temporal features. 
            Example: {
                'input_directory': str,
                'target_site':  str,
                'use_spatial': bool,
                'use_temporal': bool,
                'use_lagged': bool,
                'use_rolling': bool
            }

    Returns:
        pd.DataFrame: Imputed data. 
    """
    logging.info("Starting MICE imputation...")

    # Input validation
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    
    if not isinstance(input_columns, list) or not all(isinstance(col, str) for col in input_columns):
        raise ValueError("input_columns must be a list of strings.")
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the data.")
    
    if not all(col in data.columns for col in input_columns):
        missing_cols = [col for col in input_columns if col not in data.columns]
        raise ValueError(f"Input columns not found in data: {missing_cols}")

    # Ensure the target column is not in the input columns
    if target_column in input_columns: 
        logging.warning(f"Target column '{target_column}' is in input_columns. Removing it to avoid data leakage.")
        input_columns = [col for col in input_columns if col != target_column]

    # Prepare data with spatial-temporal features if configured
    if spatial_config and spatial_config.get('use_spatial', False):
        logging.info("Preparing data with spatial-temporal features...")
        data_enhanced, enhanced_input_columns = prepare_spatial_temporal_data(
            data, 
            target_column, 
            input_columns, 
            spatial_config
        )
        
        # Use enhanced features
        input_columns_to_use = enhanced_input_columns
        data_to_use = data_enhanced
    else:
        # Use original data
        input_columns_to_use = input_columns
        data_to_use = data. copy()

    # Apply custom strategies if provided
    if custom_strategies:
        logging.info("Applying custom imputation strategies...")
        for col, strategy in custom_strategies.items():
            if col in data_to_use. columns:
                if strategy == "mean":
                    data_to_use[col] = data_to_use[col].fillna(data_to_use[col].mean())
                elif strategy == "median":
                    data_to_use[col] = data_to_use[col].fillna(data_to_use[col].median())
                elif isinstance(strategy, (int, float)):
                    data_to_use[col] = data_to_use[col]. fillna(strategy)
                else:
                    raise ValueError(f"Unsupported strategy '{strategy}' for column '{col}'.")

    # Debug: Print columns and data info
    logging.info(f"Total features for imputation: {len(input_columns_to_use)}")
    if spatial_config and spatial_config.get('use_spatial', False):
        spatial_features = [col for col in input_columns_to_use if col.startswith('spatial_')]
        temporal_features = [col for col in input_columns_to_use if any(t in col for t in ['Hour', 'Day', 'Month', 'Week'])]
        logging.info(f"  - Spatial features: {len(spatial_features)}")
        logging.info(f"  - Temporal features: {len(temporal_features)}")

    # Prepare data for MICE imputation
    columns_for_imputation = input_columns_to_use + [target_column]
    
    # Filter to only include columns that exist in data_to_use
    columns_for_imputation = [col for col in columns_for_imputation if col in data_to_use.columns]
    
    data_for_imputation = data_to_use[columns_for_imputation]

    # Initialize the MICE imputer
    mice_imputer = IterativeImputer(
        max_iter=max_iter,
        random_state=random_state,
        tol=tol
    )
    
    # Perform imputation on the input columns and target column
    logging.info("Performing imputation...")
    imputed_data = mice_imputer.fit_transform(data_for_imputation)
    
    # Create a DataFrame with the imputed data
    imputed_df = pd. DataFrame(imputed_data, columns=columns_for_imputation, index=data_to_use.index)
    
    # Keep only the original columns from the input data
    original_columns = data. columns.tolist()
    final_df = data. copy()
    
    # Update only the columns that were imputed
    for col in original_columns:
        if col in imputed_df.columns:
            final_df[col] = imputed_df[col]
    
    logging.info("MICE imputation completed successfully.")
    return final_df