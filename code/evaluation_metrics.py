# /mnt/data/evaluation_metrics.py
# -*- coding: utf-8 -*-
"""
Robust evaluation metrics for spatial–temporal imputation.

Fixes:
- Prevents numeric blow-ups (e.g., 1e16 / 1e32) by using tolerance-based guards.
- Ensures all metrics return finite floats or NaN (never inf / huge unstable values).
- Provides METRIC_FUNCTIONS for gap-stratified evaluation.
- Adds a helper to save metrics rows as:  <target_variable>_Metrics.csv

Author: Dr. Masrur (framework) + robustness fixes (assistant)
Last Updated: 2026-02-28
"""


import os
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Numerical stability controls
# -----------------------------------------------------------------------------
EPS = 1e-10          # tolerance for "near-zero" denominators
MIN_N = 2            # minimum number of valid pairs required for most metrics


def _as_1d(a) -> np.ndarray:
    """Convert array-like to 1D numpy array."""
    return np.asarray(a).reshape(-1)


def filter_nan(s, o):
    """
    Remove non-finite pairs from (s, o).
    Returns: (s_clean, o_clean) as 1D numpy arrays.
    """
    s_arr = _as_1d(s)
    o_arr = _as_1d(o)
    mask = np.isfinite(s_arr) & np.isfinite(o_arr)
    if mask.sum() >= 1:
        return s_arr[mask], o_arr[mask]
    return np.array([], dtype=float), np.array([], dtype=float)


def _safe_float(x):
    """Return float(x) if finite, else np.nan."""
    try:
        x = float(x)
    except Exception:
        return np.nan
    return x if np.isfinite(x) else np.nan


def _safe_div(num, den):
    """Safe division with tolerance guard."""
    den = float(den)
    if not np.isfinite(den) or abs(den) < EPS:
        return np.nan
    out = float(num) / den
    return out if np.isfinite(out) else np.nan


def process_data(observed_values, predicted_values, handle_negative: str = "exclude"):
    """
    Clean data pairs + optionally drop negative values.

    Returns:
        o_clean, s_clean  (observed, predicted)
    """
    s, o = filter_nan(predicted_values, observed_values)
    if s.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if handle_negative == "exclude":
        m = (o >= 0) & (s >= 0)
        o = o[m]
        s = s[m]

    # enforce finite again (defensive)
    m2 = np.isfinite(o) & np.isfinite(s)
    o = o[m2]
    s = s[m2]

    return o, s


# -----------------------------------------------------------------------------
# Core metrics
# -----------------------------------------------------------------------------
def correlation(s, o):
    """Correlation Coefficient (R)."""
    s, o = filter_nan(s, o)
    if s.size < MIN_N:
        return np.nan
    if np.std(s) < EPS or np.std(o) < EPS:
        return np.nan
    r = np.corrcoef(o, s)[0, 1]
    return _safe_float(r)


def calculate_nse(observed_values, predicted_values):
    """Nash–Sutcliffe Efficiency (NSE)."""
    s, o = filter_nan(predicted_values, observed_values)
    if s.size < MIN_N:
        return np.nan
    denom = np.sum((o - np.mean(o)) ** 2)
    if not np.isfinite(denom) or abs(denom) < EPS:
        return np.nan
    nse = 1.0 - (np.sum((s - o) ** 2) / denom)
    return _safe_float(nse)


def calculate_wi(s, o):
    """Index of Agreement (WI) (Willmott)."""
    s, o = filter_nan(s, o)
    if s.size < MIN_N:
        return np.nan
    denom = np.sum((np.abs(s - np.mean(o)) + np.abs(o - np.mean(o))) ** 2)
    if not np.isfinite(denom) or abs(denom) < EPS:
        return np.nan
    wi = 1.0 - (np.sum((o - s) ** 2) / denom)
    return _safe_float(wi)


def calculate_mbe(s, o):
    """Mean Bias Error (MBE): mean(s - o)."""
    s, o = filter_nan(s, o)
    if s.size < 1:
        return np.nan
    mbe = np.mean(s - o)
    return _safe_float(mbe)


def calculate_apb(s, o):
    """Absolute Percent Bias (APB): 100 * sum(|s-o|) / sum(o)."""
    s, o = filter_nan(s, o)
    if s.size < 1:
        return np.nan
    denom = np.sum(o)
    if not np.isfinite(denom) or abs(denom) < EPS:
        return np.nan
    apb = 100.0 * np.sum(np.abs(s - o)) / denom
    return _safe_float(apb)


def calculate_kge(s, o):
    """Kling–Gupta Efficiency (KGE)."""
    s, o = filter_nan(s, o)
    if s.size < MIN_N:
        return np.nan

    cc = correlation(s, o)
    if not np.isfinite(cc):
        return np.nan

    std_o = np.std(o)
    std_s = np.std(s)
    sum_o = np.sum(o)
    sum_s = np.sum(s)

    if (not np.isfinite(std_o)) or (abs(std_o) < EPS):
        return np.nan
    if (not np.isfinite(sum_o)) or (abs(sum_o) < EPS):
        return np.nan

    alpha = std_s / std_o if (np.isfinite(std_s) and abs(std_o) >= EPS) else np.nan
    beta = sum_s / sum_o if (np.isfinite(sum_s) and abs(sum_o) >= EPS) else np.nan

    if (not np.isfinite(alpha)) or (not np.isfinite(beta)):
        return np.nan

    kge = 1.0 - np.sqrt((cc - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return _safe_float(kge)


def calculate_lm(s, o):
    """Legate & McCabe’s Index (LM)."""
    s, o = filter_nan(s, o)
    if s.size < MIN_N:
        return np.nan
    obar = np.mean(o)
    denom = np.sum(np.abs(o - obar))
    if not np.isfinite(denom) or abs(denom) < EPS:
        return np.nan
    lm = 1.0 - (np.sum(np.abs(o - s)) / denom)
    return _safe_float(lm)


def calculate_rmse(s, o):
    """Root Mean Squared Error (RMSE)."""
    s, o = filter_nan(s, o)
    if s.size < 1:
        return np.nan
    rmse = np.sqrt(np.mean((s - o) ** 2))
    return _safe_float(rmse)


def calculate_rrmse(s, o):
    """Relative RMSE (RRMSE): 100 * RMSE / mean(o)."""
    s, o = filter_nan(s, o)
    if s.size < 1:
        return np.nan
    mu = np.mean(o)
    if not np.isfinite(mu) or abs(mu) < EPS:
        return np.nan
    rmse = np.sqrt(np.mean((s - o) ** 2))
    rrmse = 100.0 * rmse / mu
    return _safe_float(rrmse)


def calculate_rmae(s, o):
    """Relative MAE (RMAE): average(|s-o|/|o|)*100 over non-zero o."""
    s, o = filter_nan(s, o)
    if s.size < 1:
        return np.nan
    mask = np.abs(o) >= EPS
    if not np.any(mask):
        return np.nan
    rmae = 100.0 * np.mean(np.abs(s[mask] - o[mask]) / np.abs(o[mask]))
    return _safe_float(rmae)


def calculate_mape(y_true, y_pred):
    """Mean Absolute Percentage Error (MAPE): mean(|(y - yhat)/y|)*100 over non-zero y."""
    y_true = _as_1d(y_true)
    y_pred = _as_1d(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) >= EPS)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size < 1:
        return np.nan
    mape = 100.0 * np.mean(np.abs((y_true - y_pred) / y_true))
    return _safe_float(mape)


def calculate_mae(s, o):
    """Mean Absolute Error (MAE)."""
    s, o = filter_nan(s, o)
    if s.size < 1:
        return np.nan
    mae = np.mean(np.abs(s - o))
    return _safe_float(mae)


def calculate_r2(s, o):
    """Coefficient of Determination (R²)."""
    s, o = filter_nan(s, o)
    if s.size < MIN_N:
        return np.nan
    denom = np.sum((o - np.mean(o)) ** 2)
    if not np.isfinite(denom) or abs(denom) < EPS:
        return np.nan
    r2 = 1.0 - (np.sum((s - o) ** 2) / denom)
    return _safe_float(r2)


def calculate_acc(s, o):
    """Anomaly Correlation Coefficient (ACC): corr(s-mean(s), o-mean(o))."""
    s, o = filter_nan(s, o)
    if s.size < MIN_N:
        return np.nan
    s_an = s - np.mean(s)
    o_an = o - np.mean(o)
    if np.std(s_an) < EPS or np.std(o_an) < EPS:
        return np.nan
    acc = np.corrcoef(s_an, o_an)[0, 1]
    return _safe_float(acc)


def calculate_nrmse(s, o):
    """Mean Normalized RMSE (NRMSE): RMSE / mean(o)."""
    s, o = filter_nan(s, o)
    if s.size < 1:
        return np.nan
    mu = np.mean(o)
    if not np.isfinite(mu) or abs(mu) < EPS:
        return np.nan
    rmse = np.sqrt(np.mean((s - o) ** 2))
    nrmse = rmse / mu
    return _safe_float(nrmse)


# -----------------------------------------------------------------------------
# Main evaluation function (returns EXACT requested metric keys)
# -----------------------------------------------------------------------------
def evaluate_metrics(true_values, imputed_values, handle_negative: str = "exclude"):
    """
    Evaluate all performance metrics for imputation.

    Returns a dict with EXACT metric names your pipeline expects.
    All values are finite floats or np.nan (never inf / explosive).
    """
    o, s = process_data(true_values, imputed_values, handle_negative=handle_negative)
    if o.size < 1:
        # Return all metrics as NaN (consistent schema)
        out = {k: np.nan for k in METRIC_KEYS_ORDERED()}
        return out

    metrics = {
        "Nash-Sutcliffe Efficiency (NSE)": calculate_nse(o, s),
        "Index of Agreement (WI)": calculate_wi(o, s),
        "Mean Bias Error (MBE)": calculate_mbe(o, s),
        "Absolute Percent Bias (APB)": calculate_apb(o, s),
        "Kling-Gupta Efficiency (KGE)": calculate_kge(o, s),
        "Legate's and McCabe's Index (LM)": calculate_lm(o, s),
        "Root Mean Squared Error (RMSE)": calculate_rmse(o, s),
        "Relative Root Mean Squared Error (RRMSE)": calculate_rrmse(o, s),
        "Relative Mean Absolute Error (RMAE)": calculate_rmae(o, s),
        "Mean Absolute Error (MAE)": calculate_mae(o, s),
        "Mean Absolute Percentage Error (MAPE)": calculate_mape(o, s),
        "Correlation Coefficient (R)": correlation(o, s),
        "Coefficient of Determination (R²)": calculate_r2(o, s),
        "Anomaly Correlation Coefficient (ACC)": calculate_acc(o, s),
        "Mean Normalized Root Mean Squared Error (NRMSE)": calculate_nrmse(o, s),
    }

    # Final guard: replace non-finite with NaN
    for k, v in list(metrics.items()):
        metrics[k] = _safe_float(v)

    # Ensure stable column order downstream (optional, but helpful)
    ordered = {k: metrics.get(k, np.nan) for k in METRIC_KEYS_ORDERED()}
    return ordered


def METRIC_KEYS_ORDERED():
    """Canonical ordered metric keys (for consistent CSV columns)."""
    return [
        "Nash-Sutcliffe Efficiency (NSE)",
        "Index of Agreement (WI)",
        "Mean Bias Error (MBE)",
        "Absolute Percent Bias (APB)",
        "Kling-Gupta Efficiency (KGE)",
        "Legate's and McCabe's Index (LM)",
        "Root Mean Squared Error (RMSE)",
        "Relative Root Mean Squared Error (RRMSE)",
        "Relative Mean Absolute Error (RMAE)",
        "Mean Absolute Error (MAE)",
        "Mean Absolute Percentage Error (MAPE)",
        "Correlation Coefficient (R)",
        "Coefficient of Determination (R²)",
        "Anomaly Correlation Coefficient (ACC)",
        "Mean Normalized Root Mean Squared Error (NRMSE)",
    ]


# -----------------------------------------------------------------------------
# Metric descriptions (optional)
# -----------------------------------------------------------------------------
def get_metric_descriptions():
    return {
        "NSE": "Nash–Sutcliffe Efficiency: 1=perfect, 0≈mean predictor, <0=poor",
        "WI": "Willmott Index of Agreement: 0–1, 1=perfect",
        "MBE": "Mean Bias Error: positive=overestimation",
        "APB": "Absolute Percent Bias: 100 * sum(|err|)/sum(obs)",
        "KGE": "Kling–Gupta Efficiency: combines correlation, bias, variability",
        "LM": "Legate–McCabe Index: 1=perfect, lower=worse",
        "RMSE": "Root Mean Squared Error: lower=better",
        "RRMSE": "Relative RMSE: 100*RMSE/mean(obs)",
        "RMAE": "Relative MAE: mean(|err|/|obs|)*100 over non-zero obs",
        "MAE": "Mean Absolute Error: lower=better",
        "MAPE": "Mean Absolute Percentage Error: over non-zero obs",
        "R": "Correlation coefficient",
        "R²": "Coefficient of determination",
        "ACC": "Anomaly correlation coefficient",
        "NRMSE": "RMSE / mean(obs)",
    }


def print_metrics(metrics_dict, title="Performance Metrics"):
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)
    for metric in METRIC_KEYS_ORDERED():
        v = metrics_dict.get(metric, np.nan)
        if np.isfinite(v):
            print(f"{metric:.<60} {v:>14.6f}")
        else:
            print(f"{metric:.<60} {'NaN':>14}")
    print("=" * 80 + "\n")


# -----------------------------------------------------------------------------
# Gap-based evaluation helpers (used by sortdata.py)
# -----------------------------------------------------------------------------
def compute_gap_lengths(simulated_mask: pd.Series) -> pd.Series:
    """
    For each True index in simulated_mask, compute the length of the contiguous
    missing block it belongs to.
    """
    if not isinstance(simulated_mask, pd.Series):
        simulated_mask = pd.Series(simulated_mask)

    mask = simulated_mask.astype(int).values
    n = len(mask)
    gap_len = np.zeros(n, dtype=int)

    i = 0
    while i < n:
        if mask[i] == 1:
            j = i
            while j < n and mask[j] == 1:
                j += 1
            length = j - i
            gap_len[i:j] = length
            i = j
        else:
            i += 1

    return pd.Series(gap_len, index=simulated_mask.index)


def classify_gap_regime(gap_len: int) -> str:
    """Hourly regimes: short <24, medium <72, else long."""
    if gap_len < 24:
        return "short"
    elif gap_len < 72:
        return "medium"
    return "long"


def evaluate_metrics_by_gap(
    y_true: pd.Series,
    y_pred: pd.Series,
    simulated_mask: pd.Series,
    metric_functions: dict,
    min_points: int = 10,
):
    """
    Return a DataFrame of metrics stratified by gap regime.
    """
    if not isinstance(simulated_mask, pd.Series):
        simulated_mask = pd.Series(simulated_mask, index=y_true.index)

    gap_lengths = compute_gap_lengths(simulated_mask)
    regimes = gap_lengths.apply(classify_gap_regime)

    records = []
    for regime in ["short", "medium", "long"]:
        idx = simulated_mask & (regimes == regime)
        n = int(idx.sum())
        if n < min_points:
            continue

        yt = y_true.loc[idx].values
        yp = y_pred.loc[idx].values

        row = {"gap_regime": regime, "n_points": n}
        for name, fn in metric_functions.items():
            try:
                row[name] = _safe_float(fn(yt, yp))
            except Exception:
                row[name] = np.nan

        records.append(row)

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# METRIC_FUNCTIONS: used by sortdata.py (RMSE etc. stratified by gap)
# -----------------------------------------------------------------------------
METRIC_FUNCTIONS = {
    "RMSE": lambda yt, yp: calculate_rmse(yp, yt),
    "MAE": lambda yt, yp: calculate_mae(yp, yt),
    "R": lambda yt, yp: correlation(yp, yt),
    "R2": lambda yt, yp: calculate_r2(yp, yt),
    "NSE": lambda yt, yp: calculate_nse(yt, yp),
}


# -----------------------------------------------------------------------------
# Saving helper: target_variable_Metrics.csv
# -----------------------------------------------------------------------------
def save_target_metrics_csv(
    rows,
    output_dir: str,
    target_variable: str,
    filename_suffix: str = "_Metrics.csv",
    append: bool = True,
):
    """
    Save one or multiple metric rows to:
        <output_dir>/Metrics/<target_variable>_Metrics.csv

    Parameters
    ----------
    rows : dict | list[dict] | pd.DataFrame
        Each row can include BOTH:
          - metric keys (from evaluate_metrics)
          - metadata columns such as:
            Missingness_Regime, Missingness, Total_Features_Used, Base_Features,
            Spatial_Features, Temporal_Features, RMSE, Model, StudySite, etc.
    output_dir : str
    target_variable : str
    append : bool
        If True, append to existing file; else overwrite.
    """
    metrics_dir = os.path.join(output_dir, "Metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
    elif isinstance(rows, dict):
        df = pd.DataFrame([rows])
    else:
        df = pd.DataFrame(list(rows))

    # Ensure metric columns exist (so CSV schema is stable)
    for k in METRIC_KEYS_ORDERED():
        if k not in df.columns:
            df[k] = np.nan

    # Convert inf -> NaN (final protection)
    df = df.replace([np.inf, -np.inf], np.nan)

    # File path
    safe_target = str(target_variable).replace(" ", "_")
    out_fp = os.path.join(metrics_dir, f"{safe_target}{filename_suffix}")

    # Write
    if append and os.path.exists(out_fp):
        df.to_csv(out_fp, mode="a", header=False, index=False)
    else:
        df.to_csv(out_fp, index=False)

    return out_fp
