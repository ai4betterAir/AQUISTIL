import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from impute_plot import save_scatterplot as _ip_save_scatterplot
except Exception:
    _ip_save_scatterplot = None


def _find_input_dir():
    env = os.environ.get('INPUT_DIRECTORY')
    if env and os.path.isdir(env):
        return env
    # common local folders
    for candidate in ('input', 'inputs', '../input', '../inputs'):
        p = os.path.abspath(candidate)
        if os.path.isdir(p):
            # prefer folder that contains a processed data subfolder
            proc = os.path.join(p, 'Processed_Data_NSW_ALL')
            if os.path.isdir(proc):
                return proc
            return p
    # fallback to known project path
    fallback = '/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/input/Processed_Data_NSW_ALL'
    return fallback


def load_and_preprocess_data(study_site, target_col):
    """Locate CSV for `study_site`, load, select numeric features and scale.

    Returns: (df_processed, feature_cols, scaler)
    df_processed: pandas DataFrame with numeric columns
    feature_cols: list of column names used for imputation
    scaler: fitted StandardScaler instance (on df_processed[feature_cols])
    """
    input_dir = _find_input_dir()
    # find file starting with study_site (search recursively to catch subfolders)
    pattern1 = os.path.join(input_dir, '**', f"{study_site}*.csv")
    candidates = glob.glob(pattern1, recursive=True)
    if not candidates:
        # try case-insensitive search anywhere under input_dir
        all_csv = glob.glob(os.path.join(input_dir, '**', '*.csv'), recursive=True)
        candidates = [p for p in all_csv if os.path.basename(p).lower().startswith(study_site.lower()) or study_site.lower() in os.path.basename(p).lower()]
    if not candidates:
        raise FileNotFoundError(f"No CSV found for study site '{study_site}' in {input_dir}")
    csv_path = candidates[0]
    df = pd.read_csv(csv_path)

    # Ensure DateTime if present
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # select numeric columns for imputation
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric:
        # try to coerce
        try:
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            if target_col not in df.select_dtypes(include=[np.number]).columns:
                raise ValueError
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        except Exception:
            raise ValueError(f"Target column '{target_col}' not numeric or not found in {csv_path}")

    feature_cols = numeric
    scaler = StandardScaler()
    # fillna temporarily with column mean for scaler fit
    tmp = df[feature_cols].fillna(df[feature_cols].mean())
    scaler.fit(tmp.values)
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(tmp.values)

    return df_scaled, feature_cols, scaler


def introduce_missingness(df, target_col, missingness_pct, random_state=42):
    """Introduce MCAR missingness of given percentage (e.g., 30) on target_col.

    Returns: (df_with_missing, missing_mask_boolean_series)
    """
    frac = float(missingness_pct) / 100.0
    rng = np.random.default_rng(random_state)
    obs_idx = df[df[target_col].notna()].index.to_numpy()
    n = int(len(obs_idx) * frac)
    if n <= 0:
        return df, pd.Series(False, index=df.index)
    chosen = rng.choice(obs_idx, size=n, replace=False)
    mask = pd.Series(False, index=df.index)
    mask.loc[chosen] = True
    df2 = df.copy()
    df2.loc[mask, target_col] = np.nan
    return df2, mask


def save_results(results_file, payload):
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(payload, f, indent=2, default=str)


def save_scatterplot(true_vals, imputed_vals, save_path, study_site, target_col, model_name, missingness, rmse=None, r=None):
    os.makedirs(save_path, exist_ok=True)
    if _ip_save_scatterplot is not None:
        try:
            _ip_save_scatterplot(true_vals, imputed_vals, save_path, study_site, target_col, model_name, missingness, rmse=rmse, r=r)
            return
        except Exception:
            pass
    # fallback simple scatter
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(true_vals, imputed_vals, alpha=0.5)
    ax.set_xlabel('True')
    ax.set_ylabel('Imputed')
    ax.set_title(f'{study_site} {target_col} {model_name} ({int(missingness)}%)')
    fname = os.path.join(save_path, f"{study_site}_{target_col}_{model_name}_Scatter_{int(missingness)}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def print_section_header(text, width=80):
    print('\n' + '='*width)
    print(text)
    print('='*width)
