import pandas as pd
import numpy as np
from impute_plot import (
    save_error_distribution,
    save_residual_plot,
    save_qq_plot,
    save_correlation_heatmap,
    save_statistical_summary,
    save_scatterplot,
    save_cdf_plot,
    save_histogram,
)
from evaluation_metrics import evaluate_metrics
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _safe_write_csv(df, path, study_site=None):
    import os
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Modify filename to include study site if provided
        if study_site:
            base, ext = os.path.splitext(path)
            path = f"{base}_{study_site}{ext}"
        
        # Reduce file size by selecting relevant columns (example: first 100 rows)
        df = df.iloc[:100]  # Adjust this logic as needed
        df.to_csv(path, index=False)
        logging.info("Saved CSV: %s", path)
        return True
    except Exception as e:
        logging.exception("Failed to save imputed data to %s: %s", path, e)
        return False

def impute_and_evaluate(input_file, plot_save_path, imputed_data_path, target_column_data_path, metrics_save_path, target_column, input_columns, missingness_levels, handle_negatives='exclude', impute_function=None, model_name=None, custom_strategies=None):
    """
    Perform imputation and evaluate the model.

    Args:
        input_file (str): Path to the input CSV file.
        plot_save_path (str): Path to save the plots.
        imputed_data_path (str): Path to save the imputed data.
        target_column_data_path (str): Path to save the target column data (original and imputed).
        metrics_save_path (str): Path to save the performance metrics.
        target_column (str): The target column for prediction.
        input_columns (list): List of input columns for the imputation model.
        missingness_levels (list): List of missingness levels for evaluation.
        handle_negatives (str): Whether to include or exclude negative values. Options: 'include' or 'exclude'.
        impute_function (function): The imputation function to use.
        model_name (str): Name of the model (retrieved from the model file).
        custom_strategies (dict): Dictionary of column-specific imputation strategies.
    """
    # Load the data
    try:
        data = pd.read_csv(input_file)
    except FileNotFoundError:
        logging.error(f"Error: {input_file} not found.")
        return

    # Debug: Check for the target column
    logging.info(f"Checking for target column: {target_column}")
    if target_column not in data.columns:
        logging.warning(f"Target column '{target_column}' not found in the data.")
        return

    # Debug: Check if the target column contains only NaN values
    if data[target_column].isna().all():
        logging.warning(f"Target column '{target_column}' contains only NaN values. Skipping imputation.")
        return

    # Filter the available target and input columns (excluding 'DateTime')
    available_columns = [col for col in [target_column] + input_columns if col in data.columns and col != 'DateTime']

    # If the target column is not present, skip the imputation
    if target_column not in available_columns:
        logging.warning(f"Skipping {input_file} - Target column '{target_column}' not available.")
        return

    # Extract the relevant columns for imputation
    impute_data = data[available_columns]

    # List to store metrics for all missingness levels
    all_metrics = []

    for missingness in missingness_levels:
        # Create a copy of the data for simulation
        simulated_missing = impute_data.copy()

        # Track original missing values
        original_missing_mask = simulated_missing[target_column].isna()

        # Introduce additional missing values to reach the desired missingness level
        total_missingness = missingness
        current_missingness = original_missing_mask.mean()
        additional_missingness = total_missingness - current_missingness

        if additional_missingness > 0:
            # Calculate the number of additional missing values to introduce
            num_additional_missing = int(additional_missingness * len(simulated_missing))

            # Randomly select rows to introduce missing values (excluding rows already missing)
            candidate_indices = simulated_missing[~original_missing_mask].index
            additional_missing_indices = np.random.choice(candidate_indices, size=num_additional_missing, replace=False)

            # Introduce additional missing values
            simulated_missing.loc[additional_missing_indices, target_column] = np.nan

        # Track simulated missing values
        simulated_missing_mask = simulated_missing[target_column].isna() & ~original_missing_mask

        # Perform imputation using the provided impute_function
        imputed_df = impute_function(
            simulated_missing, 
            target_column, 
            input_columns, 
            custom_strategies=custom_strategies  # Removed categorical_columns
        )

        # Add the 'DateTime' column back to imputed_df
        imputed_df['DateTime'] = data['DateTime']

        # Save the final imputed data with 'DateTime' column
        final_imputed_data = data.copy()
        final_imputed_data[target_column] = imputed_df[target_column]
        final_imputed_csv_filename = f"{imputed_data_path}/{input_file.split('/')[-1].split('.')[0]}_{target_column}_{model_name}_imputed_{int(missingness * 100)}.csv"
        final_imputed_data.to_csv(final_imputed_csv_filename, index=False)

        # Save the target column data (original and imputed) for time series plotting
        target_column_data = data[['DateTime', target_column]].copy()
        target_column_data[f"{target_column}_imputed"] = np.nan  # Initialize imputed column with NaN
        target_column_data.loc[simulated_missing_mask | original_missing_mask, f"{target_column}_imputed"] = imputed_df[target_column]

        # Add a column to indicate whether the missing value was original or simulated
        target_column_data["Missing_Type"] = "None"
        target_column_data.loc[original_missing_mask, "Missing_Type"] = "Original"
        target_column_data.loc[simulated_missing_mask, "Missing_Type"] = "Simulated"

        target_column_csv_filename = f"{target_column_data_path}/{input_file.split('/')[-1].split('.')[0]}_{target_column}_{model_name}_target_column_{int(missingness * 100)}.csv"
        target_column_data.to_csv(target_column_csv_filename, index=False)

        # Extract true and imputed values for simulated missing values only
        true_values_simulated = impute_data.loc[simulated_missing_mask, target_column].values
        imputed_values_simulated = imputed_df.loc[simulated_missing_mask, target_column].values

        # Filter out negative values if handle_negatives is set to 'exclude'
        if handle_negatives == 'exclude':
            valid_mask = (true_values_simulated >= 0) & (imputed_values_simulated >= 0)
            true_values_simulated_clean = true_values_simulated[valid_mask]
            imputed_values_simulated_clean = imputed_values_simulated[valid_mask]
        else:
            true_values_simulated_clean = true_values_simulated
            imputed_values_simulated_clean = imputed_values_simulated

        # Evaluate metrics for simulated missing values (with optional negative exclusion)
        metrics = evaluate_metrics(true_values_simulated_clean, imputed_values_simulated_clean, handle_negative=handle_negatives)

        # Add missingness level to the metrics dictionary
        metrics["Missingness"] = missingness

        # Append metrics to the list
        all_metrics.append(metrics)

        # Print metrics results
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")

        # Plot results using impute_plot functions (with optional negative exclusion)
        save_error_distribution(true_values_simulated_clean, imputed_values_simulated_clean, plot_save_path, input_file.split('/')[-1], target_column, model_name, missingness)
        save_residual_plot(true_values_simulated_clean, imputed_values_simulated_clean, plot_save_path, input_file.split('/')[-1], target_column, model_name, missingness)
        save_qq_plot(imputed_values_simulated_clean, plot_save_path, input_file.split('/')[-1], target_column, model_name, missingness)
        save_correlation_heatmap(imputed_df, plot_save_path, input_file.split('/')[-1], target_column, model_name, missingness)
        save_statistical_summary(impute_data, imputed_df, plot_save_path, input_file.split('/')[-1], target_column, model_name, missingness)
        save_scatterplot(true_values_simulated_clean, imputed_values_simulated_clean, plot_save_path, input_file.split('/')[-1], target_column, model_name, missingness)
        save_cdf_plot(true_values_simulated_clean, imputed_values_simulated_clean, plot_save_path, input_file.split('/')[-1], target_column, model_name, missingness)
        save_histogram(impute_data, imputed_values_simulated_clean, plot_save_path, input_file.split('/')[-1], target_column, model_name, missingness)

    # Save all metrics for all missingness levels in one CSV file
    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv_filename = f"{metrics_save_path}/{input_file.split('/')[-1].split('.')[0]}_{target_column}_{model_name}_all_metrics.csv"
    metrics_df.to_csv(metrics_csv_filename, index=False)


def safe_assign_imputed(df, cols, imputed):
    """
    Safely assign the output of an imputer to DataFrame columns.

    - Handles numpy arrays or DataFrames
    - Guards against shape mismatches and assigns column-wise when needed
    """
    try:
        arr = np.asarray(imputed)
        if arr.ndim != 2:
            logging.warning("Imputed result is not 2D; skipping assignment")
            return df

        n_rows_imputed, n_cols_imputed = arr.shape

        # Align row counts: trim or pad with NaN to match df index length
        target_rows = len(df)
        if n_rows_imputed != target_rows:
            logging.warning(f"Imputed array has {n_rows_imputed} rows but target has {target_rows}; trimming/padding to match.")
            if n_rows_imputed > target_rows:
                arr = arr[:target_rows, :]
            else:
                # pad with NaNs
                pad_rows = target_rows - n_rows_imputed
                pad = np.full((pad_rows, n_cols_imputed), np.nan)
                arr = np.vstack([arr, pad])

        # Build a DataFrame for safe column-wise assignment
        try:
            imputed_df = pd.DataFrame(arr, index=df.index)
        except Exception:
            imputed_df = pd.DataFrame(arr)

        # Assign columns one-by-one to avoid pandas broadcasting/shape errors
        for i, col in enumerate(cols):
            if i < imputed_df.shape[1]:
                try:
                    df[col] = imputed_df.iloc[:, i].values
                except Exception as exc:
                    logging.warning(f"Failed assigning column '{col}' from imputed output: {exc}; setting NaNs")
                    df[col] = np.nan
            else:
                logging.warning(f"Imputer returned fewer columns ({n_cols_imputed}) than expected ({len(cols)}); setting column '{col}' to NaN")
                df[col] = np.nan
    except Exception as e:
        logging.error(f"safe_assign_imputed failed: {e}")
    return df