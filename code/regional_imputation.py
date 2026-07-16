"""Balanced pooled-region imputation and per-study-site evaluation."""

import logging
import os
import json
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from missingness_regimes import apply_missingness


def _safe_correlation(actual, predicted):
    """Pearson correlation without warnings for constant/near-constant data."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if len(actual) < 2:
        return np.nan
    # Median/mean baselines can differ only at floating-point roundoff. Their
    # Pearson correlation is undefined, not zero.
    actual_scale = max(float(np.max(np.abs(actual))), 1.0)
    predicted_scale = max(float(np.max(np.abs(predicted))), 1.0)
    if (
        np.ptp(actual) <= np.finfo(float).eps * actual_scale * 16
        or np.ptp(predicted) <= np.finfo(float).eps * predicted_scale * 16
    ):
        return np.nan
    return float(np.corrcoef(actual, predicted)[0, 1])


def _safe_mean(values):
    """Return NaN for an empty/all-NaN collection without NumPy warnings."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else np.nan


def _balanced_region_mask(data, target, regime, fraction, seed):
    """Mask exactly the same number of observed target rows at every site."""
    observed_counts = data.groupby("Site")[target].apply(lambda s: int(s.notna().sum()))
    if observed_counts.empty or (observed_counts <= 0).any():
        raise ValueError("Every regional study site must contain observed target values")
    n_per_site = min(int(count * fraction) for count in observed_counts.values)
    if n_per_site <= 0:
        raise ValueError("Balanced missingness selected zero rows per site")

    combined = pd.Series(False, index=data.index)
    for site_number, (site, site_data) in enumerate(data.groupby("Site", sort=False)):
        site_seed = seed + site_number * 1009
        _, proposed = apply_missingness(
            site_data.copy(), target, regime=regime, frac=n_per_site / observed_counts[site], seed=site_seed
        )
        proposed = pd.Series(np.asarray(proposed, dtype=bool), index=site_data.index)
        proposed &= site_data[target].notna()
        chosen = proposed[proposed].index.tolist()
        rng = np.random.default_rng(site_seed)
        if len(chosen) > n_per_site:
            chosen = rng.choice(chosen, n_per_site, replace=False).tolist()
        elif len(chosen) < n_per_site:
            available = site_data.index[site_data[target].notna() & ~site_data.index.isin(chosen)]
            supplement = rng.choice(available, n_per_site - len(chosen), replace=False).tolist()
            chosen.extend(supplement)
        combined.loc[chosen] = True

    per_site = combined.groupby(data["Site"]).sum()
    if per_site.nunique() != 1 or int(per_site.iloc[0]) != n_per_site:
        raise AssertionError("Regional mask is not balanced across sites: %s" % per_site.to_dict())
    return combined, n_per_site


def _metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if not len(actual):
        return dict(RMSE=np.nan, MAE=np.nan, R=np.nan, NSE=np.nan, N_Valid=0)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    correlation = _safe_correlation(actual, predicted)
    denominator = np.sum((actual - np.mean(actual)) ** 2)
    nse = float(1 - np.sum((actual - predicted) ** 2) / denominator) if denominator > 0 else np.nan
    return dict(RMSE=rmse, MAE=mae, R=correlation, NSE=nse, N_Valid=int(len(actual)))


def _safe_token(value):
    return str(value).strip().replace(" ", "_").replace(".", "")


def _save_regional_plots(predictions, plots_root, region, target, model_name, plot_types, dpi=300):
    """Create region-model evaluation plots for each site and the pooled region."""
    if predictions.empty:
        return
    region_token, target_token, model_token = map(_safe_token, (region, target, model_name))
    scopes = [("Region_ALL", predictions)]
    scopes.extend(("Site_%s" % _safe_token(site), part) for site, part in predictions.groupby("Site"))

    for scope, scope_data in scopes:
        for (regime, fraction), part in scope_data.groupby(["Regime", "Missingness_Level"]):
            actual = pd.to_numeric(part["Observed"], errors="coerce").to_numpy(float)
            imputed = pd.to_numeric(part["Imputed"], errors="coerce").to_numpy(float)
            valid = np.isfinite(actual) & np.isfinite(imputed)
            actual, imputed = actual[valid], imputed[valid]
            if not len(actual):
                continue
            residual = imputed - actual
            subtitle = "%s | %s | %s | %s | %.0f%%" % (
                region, scope.replace("_", " "), model_name, regime, fraction * 100
            )
            for plot_type in plot_types:
                out_dir = os.path.join(
                    plots_root, plot_type, region_token, target_token, model_token, scope
                )
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, "%s_%02d.png" % (_safe_token(regime), round(fraction * 100)))
                fig, ax = plt.subplots(figsize=(7, 5))
                if plot_type == "error_distribution":
                    ax.hist(residual, bins=40, color="#4472C4", alpha=.85)
                    ax.set_xlabel("Imputed - observed")
                    ax.set_ylabel("Count")
                elif plot_type == "residual_plot":
                    ax.scatter(imputed, residual, s=7, alpha=.35)
                    ax.axhline(0, color="black", linewidth=1)
                    ax.set_xlabel("Imputed")
                    ax.set_ylabel("Residual")
                elif plot_type == "qq_plot":
                    ordered = np.sort(residual)
                    theoretical = np.sort(np.random.default_rng(0).normal(np.mean(residual), np.std(residual), len(residual)))
                    ax.scatter(theoretical, ordered, s=7, alpha=.45)
                    limits = [min(theoretical.min(), ordered.min()), max(theoretical.max(), ordered.max())]
                    ax.plot(limits, limits, "k--", linewidth=1)
                    ax.set_xlabel("Normal theoretical quantiles")
                    ax.set_ylabel("Residual quantiles")
                elif plot_type == "scatterplot":
                    ax.scatter(actual, imputed, s=7, alpha=.35)
                    limits = [min(actual.min(), imputed.min()), max(actual.max(), imputed.max())]
                    ax.plot(limits, limits, "k--", linewidth=1)
                    ax.set_xlabel("Observed")
                    ax.set_ylabel("Imputed")
                elif plot_type == "cdf_plot":
                    for values, label in ((actual, "Observed"), (imputed, "Imputed")):
                        ordered = np.sort(values)
                        ax.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), label=label)
                    ax.set_xlabel(target)
                    ax.set_ylabel("Empirical CDF")
                    ax.legend()
                elif plot_type == "histogram":
                    ax.hist(actual, bins=40, alpha=.55, label="Observed")
                    ax.hist(imputed, bins=40, alpha=.55, label="Imputed")
                    ax.set_xlabel(target)
                    ax.set_ylabel("Count")
                    ax.legend()
                elif plot_type == "correlation_heatmap":
                    cross_correlation = _safe_correlation(actual, imputed)
                    corr = np.array(
                        [
                            [1.0 if np.ptp(actual) > 0 else np.nan, cross_correlation],
                            [cross_correlation, 1.0 if np.ptp(imputed) > 0 else np.nan],
                        ]
                    )
                    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
                    ax.set_xticks([0, 1], ["Observed", "Imputed"])
                    ax.set_yticks([0, 1], ["Observed", "Imputed"])
                    for i in range(2):
                        for j in range(2):
                            ax.text(j, i, "%.3f" % corr[i, j], ha="center", va="center")
                    fig.colorbar(image, ax=ax)
                else:
                    plt.close(fig)
                    continue
                ax.set_title(subtitle)
                fig.tight_layout()
                fig.savefig(out_path, dpi=dpi)
                plt.close(fig)


def run_balanced_regional_task(
    regional_data: pd.DataFrame,
    region: str,
    target: str,
    features: List[str],
    model_name: str,
    impute_callable: Callable,
    regimes: Iterable[str],
    missingness_levels: Iterable[float],
    seeds: Iterable[int],
    output_root: str,
    plots_root: str = None,
    plot_types: Iterable[str] = (),
    plot_dpi: int = 300,
    parameters: Dict = None,
) -> Dict[str, pd.DataFrame]:
    """Run one model on one pooled region and save predictions/metrics."""
    task_root = os.path.join(output_root, model_name, region.replace(" ", "_"), target.replace(".", ""))
    os.makedirs(task_root, exist_ok=True)
    data = regional_data.copy().sort_values(["Site", "DateTime"]).reset_index(drop=True)
    features = list(dict.fromkeys([f for f in features if f in data.columns and f != target] + ["SiteCode"]))
    parameter_values = dict(parameters or {})
    parameter_values["features"] = features
    parameters_json = json.dumps(parameter_values, sort_keys=True, default=str)
    feature_list = ",".join(features)
    data["SiteCode"] = pd.Categorical(data["Site"]).codes.astype(float)
    metric_rows, prediction_parts = [], []

    for regime in regimes:
        for fraction in missingness_levels:
            for seed in seeds:
                mask, n_per_site = _balanced_region_mask(data, target, regime, fraction, seed)
                work = data.copy()
                truth = pd.to_numeric(data[target], errors="coerce")
                work.loc[mask, target] = np.nan
                logging.info(
                    "Regional task %s/%s/%s | %s %.0f%% seed=%d | balanced masked/site=%d",
                    region, target, model_name, regime, fraction * 100, seed, n_per_site,
                )
                result = impute_callable(
                    work,
                    target,
                    features,
                    custom_strategies=None,
                    site_name=region.replace(" ", "_"),
                    model_name="%s_%s" % (model_name, region.replace(" ", "_")),
                    out_dir=task_root,
                )
                if result is None or target not in result.columns:
                    raise RuntimeError("%s returned no imputed target for %s" % (model_name, region))
                predicted = pd.to_numeric(result[target], errors="coerce").reset_index(drop=True)
                prediction = data.loc[mask, ["DateTime", "Region", "Site"]].copy()
                prediction["Target"] = target
                prediction["Model"] = model_name
                prediction["Regime"] = regime
                prediction["Missingness_Level"] = fraction
                prediction["Seed"] = seed
                prediction["Observed"] = truth.loc[mask]
                prediction["Imputed"] = predicted.loc[mask]
                prediction["Was_Artificially_Masked"] = True
                prediction_parts.append(prediction)

                site_metrics = []
                for site, site_prediction in prediction.groupby("Site", sort=False):
                    values = _metrics(site_prediction["Observed"], site_prediction["Imputed"])
                    row = dict(Region=region, Site=site, Target=target, Model=model_name,
                               Parameters=parameters_json, Features=feature_list,
                               Feature_Count=len(features), Regime=regime,
                               Missingness_Level=fraction, Missingness_Percent=fraction * 100,
                               Seed=seed, N_Masked=n_per_site, Scope="Site", **values)
                    metric_rows.append(row)
                    site_metrics.append(row)
                micro = _metrics(prediction["Observed"], prediction["Imputed"])
                metric_rows.append(dict(Region=region, Site="ALL", Target=target, Model=model_name,
                                        Parameters=parameters_json, Features=feature_list,
                                        Feature_Count=len(features), Regime=regime,
                                        Missingness_Level=fraction, Missingness_Percent=fraction * 100,
                                        Seed=seed, N_Masked=int(mask.sum()), Scope="Region_Micro", **micro))
                metric_rows.append(dict(
                    Region=region, Site="ALL", Target=target, Model=model_name, Regime=regime,
                    Parameters=parameters_json, Features=feature_list, Feature_Count=len(features),
                    Missingness_Percent=fraction * 100,
                    Missingness_Level=fraction, Seed=seed, N_Masked=int(mask.sum()), Scope="Region_Macro",
                    RMSE=_safe_mean([r["RMSE"] for r in site_metrics]),
                    MAE=_safe_mean([r["MAE"] for r in site_metrics]),
                    R=_safe_mean([r["R"] for r in site_metrics]),
                    NSE=_safe_mean([r["NSE"] for r in site_metrics]),
                    N_Valid=int(sum(r["N_Valid"] for r in site_metrics)),
                ))

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    metrics.to_csv(os.path.join(task_root, "metrics_by_site_and_region.csv"), index=False)
    predictions.to_csv(os.path.join(task_root, "masked_predictions_by_site.csv"), index=False)
    if plots_root and plot_types:
        _save_regional_plots(
            predictions, plots_root, region, target, model_name, list(plot_types), dpi=plot_dpi
        )
    return {"metrics": metrics, "predictions": predictions}
