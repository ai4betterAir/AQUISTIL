import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import seaborn as sns
import scipy.stats as stats
import logging
from PIL import Image, ImageDraw, ImageFont

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Utility function to ensure directory exists
def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        logging.info(f"Creating directory: {directory}")
        os.makedirs(directory)

# Helper function to create subfolder for plot type
def create_plot_subfolder(save_path, plot_type):
    plot_subfolder = os.path.join(save_path, plot_type)
    if not os.path.exists(plot_subfolder):
        logging.info(f"Creating directory: {plot_subfolder}")
        os.makedirs(plot_subfolder)
    return plot_subfolder


def _save_placeholder_plot(path, text):
    # create a small image with text
    try:
        img = Image.new('RGB', (800, 400), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        try:
            f = ImageFont.load_default()
            d.text((20, 180), text, font=f, fill=(0, 0, 0))
        except Exception:
            d.text((20, 180), text, fill=(0, 0, 0))
        img.save(path, dpi=(200, 200))
    except Exception:
        try:
            with open(path, 'w') as f:
                f.write(text)
        except Exception:
            pass

# Save Error Distribution
def save_error_distribution(true_values, imputed_values, save_path, study_site, target_column, define_model_name, missingness):
    # ensure arrays
    try:
        arr_t = np.asarray(true_values)
        arr_i = np.asarray(imputed_values)
    except Exception:
        arr_t = np.array([])
        arr_i = np.array([])

    if arr_t.size == 0 or arr_i.size == 0:
        plot_subfolder = create_plot_subfolder(save_path, "ErrorDistribution")
        error_plot_filename = os.path.join(
            plot_subfolder, f"{study_site}_{target_column}_{define_model_name}_ErrorDistribution_{int(missingness * 100)}.png"
        )
        _save_placeholder_plot(error_plot_filename, f"No data to plot Error Distribution for {study_site} {target_column}")
        return

    errors = arr_t - arr_i
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(errors, bins=30, alpha=0.8, color="orange", edgecolor="black")
    ax.set_title(f"Error Distribution for {target_column}", fontsize=18)
    ax.set_xlabel("Error (Actual - Imputed)", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.grid(True)

    plot_subfolder = create_plot_subfolder(save_path, "ErrorDistribution")
    error_plot_filename = os.path.join(
        plot_subfolder, f"{study_site}_{target_column}_{define_model_name}_ErrorDistribution_{int(missingness * 100)}.png"
    )
    plt.tight_layout()
    plt.savefig(error_plot_filename, dpi=200)
    plt.close()

# Save Residual Plot
def save_residual_plot(true_values, imputed_values, save_path, study_site, target_column, define_model_name, missingness):
    residuals = true_values - imputed_values
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(true_values, residuals, alpha=0.5, color="purple")
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title(f"Residuals vs Actual Values for {target_column}", fontsize=18)
    ax.set_xlabel("Actual Values", fontsize=14)
    ax.set_ylabel("Residuals (Actual - Imputed)", fontsize=14)
    ax.grid(True)

    plot_subfolder = create_plot_subfolder(save_path, "ResidualPlot")
    residual_plot_filename = os.path.join(
        plot_subfolder, f"{study_site}_{target_column}_{define_model_name}_ResidualPlot_{int(missingness * 100)}.png"
    )
    plt.tight_layout()
    plt.savefig(residual_plot_filename, dpi=200)
    plt.close()

# Save Time Series Plot
def save_time_series_plot(impute_data, imputed_df, save_path, study_site, target_column, define_model_name, missingness):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Ensure the DateTime column is present
    if 'DateTime' not in impute_data.columns or 'DateTime' not in imputed_df.columns:
        raise ValueError("'DateTime' column is missing in the input data.")

    # Plot the actual and imputed values using the DateTime index
    ax.plot(impute_data['DateTime'], impute_data[target_column], label="Actual", color="blue", alpha=0.7)
    ax.plot(imputed_df['DateTime'], imputed_df[target_column], label="Imputed", color="green", linestyle="--", alpha=0.7)
    
    # Set the title and labels
    ax.set_title(f"Time Series Plot for {target_column}", fontsize=18)
    ax.set_xlabel("Date and Time", fontsize=14)
    ax.set_ylabel(target_column, fontsize=14)
    ax.legend()
    ax.grid(True)
    
    # Format the x-axis to display date and time properly
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y %H:%M'))  # Format as per your CSV
    fig.autofmt_xdate()  # Rotate date labels for better readability

    # Save the plot
    plot_subfolder = create_plot_subfolder(save_path, "TimeSeriesPlot")
    time_series_plot_filename = os.path.join(
        plot_subfolder, f"{study_site}_{target_column}_{define_model_name}_TimeSeriesPlot_{int(missingness * 100)}.png"
    )
    plt.tight_layout()
    plt.savefig(time_series_plot_filename, dpi=200)
    plt.close()

# Save Scatterplot
def save_scatterplot(true_values, imputed_values, save_path, study_site, target_column, define_model_name, missingness, rmse=None, r=None):
    """
    Save a scatter plot comparing true values and imputed values.

    Args:
        true_values (array-like): Array of true (observed) values.
        imputed_values (array-like): Array of imputed values.
        save_path (str): Directory to save the plot.
        study_site (str): Name of the study site.
        target_column (str): Name of the target column.
        define_model_name (str): Name of the imputation model.
        missingness (float): Missingness level (e.g., 0.1 for 10%).
    """
    # Ensure the input arrays are not empty
    try:
        arr_t = np.asarray(true_values)
        arr_i = np.asarray(imputed_values)
    except Exception:
        arr_t = np.array([])
        arr_i = np.array([])

    if arr_t.size == 0 or arr_i.size == 0:
        plot_subfolder = create_plot_subfolder(save_path, "Scatterplot")
        scatterplot_filename = os.path.join(
            plot_subfolder, f"{study_site}_{target_column}_{define_model_name}_Scatterplots_{int(missingness * 100)}.png"
        )
        _save_placeholder_plot(scatterplot_filename, f"No data to plot Scatter for {study_site} {target_column}")
        print(f"Placeholder scatter plot saved to: {scatterplot_filename}")
        return

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    # Allow caller to supply pre-computed metrics to guarantee consistency
    if rmse is None:
        rmse = np.sqrt(mean_squared_error(true_values, imputed_values))
    if r is None:
        arr_t = np.array(true_values)
        arr_i = np.array(imputed_values)
        if np.std(arr_t) == 0 or np.std(arr_i) == 0:
            r = np.NaN
        else:
            r = np.corrcoef(arr_t, arr_i)[0, 1]

    # Scatter plot of true vs imputed values
    ax.scatter(true_values, imputed_values, s=50, alpha=0.4, color="blue", label="Data Points")

    # Add a best-fit line
    x = np.array(true_values).reshape(-1, 1)
    y = np.array(imputed_values)
    reg = LinearRegression().fit(x, y)
    best_fit_line = reg.predict(x)
    ax.plot(true_values, best_fit_line, color="red", label="Best Fit Line")

    # Set plot title and labels
    # Use 4 decimal places to match printed metric outputs
    ax.set_title(f"{target_column}\nRMSE={rmse:.4f}, R={r:.4f}", fontsize=18)
    ax.set_xlabel("True Values", fontsize=14)
    ax.set_ylabel("Imputed Values", fontsize=14)
    ax.legend()
    ax.grid(True)

    # Save the scatter plot
    plot_subfolder = create_plot_subfolder(save_path, "Scatterplot")
    scatterplot_filename = os.path.join(
        plot_subfolder, f"{study_site}_{target_column}_{define_model_name}_Scatterplots_{int(missingness * 100)}.png"
    )
    plt.tight_layout()
    plt.savefig(scatterplot_filename, dpi=200)
    plt.close()
    print(f"Scatter plot saved to: {scatterplot_filename}")

# Save Histogram
def save_histogram(impute_data, imputed_values, save_path, study_site, target_column, define_model_name, missingness):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(impute_data[target_column].dropna(), bins=20, alpha=0.8, color="red", edgecolor="purple", label="Original")
    ax.hist(imputed_values, bins=20, alpha=0.3, color="blue", edgecolor="green", label="Imputed")
    ax.set_title(f"Histogram for {target_column}", fontsize=18)
    ax.set_xlabel(target_column, fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.legend()
    ax.grid(True)

    plot_subfolder = create_plot_subfolder(save_path, "Histogram")
    histogram_filename = os.path.join(
        plot_subfolder, f"{study_site}_{target_column}_{define_model_name}_Histograms_{int(missingness * 100)}.png"
    )
    plt.tight_layout()
    plt.savefig(histogram_filename, dpi=200)
    plt.close()

# Save QQ Plot
def save_qq_plot(imputed_values, save_path, study_site, target_column, define_model_name, missingness):
    fig, ax = plt.subplots(figsize=(8, 6))
    stats.probplot(imputed_values, dist="norm", plot=ax)
    ax.set_title(f"{target_column} - QQ Plot", fontsize=18)
    ax.grid(True)

    plot_subfolder = create_plot_subfolder(save_path, "QQPlot")
    qq_plot_filename = os.path.join(
        plot_subfolder, f"{study_site}_{target_column}_{define_model_name}_QQPlot_{int(missingness * 100)}.png"
    )
    plt.tight_layout()
    plt.savefig(qq_plot_filename, dpi=200)
    plt.close()

# Save Correlation Heatmap
def save_correlation_heatmap(imputed_df, save_path, study_site, target_column, define_model_name, missingness):
    correlation_matrix = imputed_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title(f"Feature Correlation Heatmap - {study_site}", fontsize=18)

    plot_subfolder = create_plot_subfolder(save_path, "CorrelationHeatmap")
    heatmap_filename = os.path.join(
        plot_subfolder, f"{study_site}_{target_column}_{define_model_name}_CorrelationHeatmap_{int(missingness * 100)}.png"
    )
    plt.tight_layout()
    plt.savefig(heatmap_filename, dpi=200)
    plt.close()

# Save Statistical Summary
def save_statistical_summary(impute_data, imputed_df, save_path, study_site, target_column, define_model_name, missingness):
    """
    Save statistical summary for original and imputed data, including the number of imputed observations.
    Append the results to a single CSV file for each study site and target column.

    Args:
        impute_data (pd.DataFrame): Original data with missing values.
        imputed_df (pd.DataFrame): Imputed data.
        save_path (str): Directory to save the summary.
        study_site (str): Name of the study site.
        target_column (str): Name of the target column.
        define_model_name (str): Name of the imputation model.
        missingness (float): Missingness level (e.g., 0.1 for 10%).
    """
    # Calculate the number of imputed observations
    num_imputed = imputed_df[target_column].isna().sum()

    # Generate statistical summaries
    original_summary = impute_data[target_column].describe()
    imputed_summary = imputed_df[target_column].describe()

    # Create a DataFrame for the summary
    summary_df = pd.DataFrame({"Original": original_summary, "Imputed": imputed_summary})

    # Add the number of imputed observations to the summary
    summary_df.loc["num_imputed"] = [None, num_imputed]

    # Add a column for missingness
    summary_df["Missingness"] = missingness

    # Define the filename for the summary (one file per study site and target column)
    plot_subfolder = create_plot_subfolder(save_path, "StatisticalSummary")
    summary_csv_filename = os.path.join(
        plot_subfolder, f"{study_site}_{target_column}_{define_model_name}_StatisticalSummary.csv"
    )

    # Check if the file already exists
    if os.path.exists(summary_csv_filename):
        # Append to the existing file
        existing_summary = pd.read_csv(summary_csv_filename, index_col=0)
        summary_df = pd.concat([existing_summary, summary_df], axis=0)
    
    # Save the summary to CSV
    summary_df.to_csv(summary_csv_filename)

# Save CDF Plot
def save_cdf_plot(true_values, imputed_values, save_path, study_site, target_column, define_model_name, missingness):
    fig, ax = plt.subplots(figsize=(8, 6))
    arr_t = np.asarray(true_values)
    arr_i = np.asarray(imputed_values)

    # small variance threshold
    eps = 1e-8
    if np.nanstd(arr_t) <= eps:
        logging.info(f"Dataset has 0 variance for Actual; skipping density estimate for {study_site} {target_column}.")
    else:
        try:
            sns.kdeplot(arr_t, label="Actual", fill=True, color="blue", alpha=0.5)
        except Exception as e:
            logging.debug(f"KDE plot for Actual failed: {e}")

    if np.nanstd(arr_i) <= eps:
        logging.info(f"Dataset has 0 variance for Imputed; skipping density estimate for {study_site} {target_column}.")
    else:
        try:
            # Pass warn_singular if seaborn supports it (to avoid backend warnings)
            kwargs = {"label": "Imputed", "fill": True, "color": "red", "alpha": 0.5}
            # seaborn versions differ; guard with try
            sns.kdeplot(arr_i, **kwargs)
        except TypeError:
            # older seaborn signature - try without extras
            try:
                sns.kdeplot(arr_i, label="Imputed", fill=True)
            except Exception as e:
                logging.debug(f"KDE plot for Imputed failed: {e}")
        except Exception as e:
            logging.debug(f"KDE plot for Imputed failed: {e}")

    ax.set_title(f"CDF Plot for {target_column}", fontsize=18)
    ax.set_xlabel(target_column, fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.legend()
    ax.grid(True)

    plot_subfolder = create_plot_subfolder(save_path, "CDFPlot")
    cdf_plot_filename = os.path.join(
        plot_subfolder, f"{study_site}_{target_column}_{define_model_name}_CDFPlot_{int(missingness * 100)}.png"
    )
    plt.tight_layout()
    plt.savefig(cdf_plot_filename, dpi=200)
    plt.close()