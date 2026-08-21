"""Balanced pooled-region imputation and per-study-site evaluation."""

import logging
import os
import json
import hashlib
import threading
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluation_metrics import evaluate_metrics
from missingness_regimes import (
    GAP_REGIME_BOUNDS,
    apply_missingness,
    assert_effective_gap_purity,
    effective_gap_mask_diagnostics,
    gap_mask_diagnostics,
    has_valid_gap_candidate,
    mask_gap_lengths,
)


# Short, analysis-friendly column names used consistently by both central CSVs.
METRIC_COLUMN_MAP = {
    "Nash-Sutcliffe Efficiency (NSE)": "NSE",
    "Index of Agreement (WI)": "WI",
    "Mean Bias Error (MBE)": "MBE",
    "Absolute Percent Bias (APB)": "APB",
    "Kling-Gupta Efficiency (KGE)": "KGE",
    "Legate's and McCabe's Index (LM)": "LM",
    "Root Mean Squared Error (RMSE)": "RMSE",
    "Relative Root Mean Squared Error (RRMSE)": "RRMSE",
    "Relative Mean Absolute Error (RMAE)": "RMAE",
    "Mean Absolute Error (MAE)": "MAE",
    "Mean Absolute Percentage Error (MAPE)": "MAPE",
    "Correlation Coefficient (R)": "R",
    "Coefficient of Determination (R²)": "R2",
    "Anomaly Correlation Coefficient (ACC)": "ACC",
    "Mean Normalized Root Mean Squared Error (NRMSE)": "NRMSE",
}
REGIONAL_METRIC_COLUMNS = list(METRIC_COLUMN_MAP.values())
_PLOT_LOCK = threading.Lock()


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


def _balanced_region_mask(data, target, regime, fraction, seed, region=None):
    """Build per-site percentage masks without compromising gap-regime purity."""
    observed_counts = data.groupby("Site")[target].apply(lambda s: int(s.notna().sum()))

    combined = pd.Series(False, index=data.index)
    diagnostics_by_site = {}
    exclusions = []
    for site_number, (site, site_data) in enumerate(data.groupby("Site", sort=False)):
        site_seed = seed + site_number * 1009
        rng = np.random.default_rng(site_seed)
        original_missing = site_data[target].isna()
        observed_count = int((~original_missing).sum())
        if observed_count <= 0:
            exclusions.append(
                dict(Region=region, Site=site, Target=target, Regime=regime, Seed=seed,
                     Requested_Missingness=fraction, Reason="target entirely NaN")
            )
            continue
        if regime in GAP_REGIME_BOUNDS and not has_valid_gap_candidate(site_data[target], regime):
            exclusions.append(
                dict(Region=region, Site=site, Target=target, Regime=regime, Seed=seed,
                     Requested_Missingness=fraction, Reason="no valid anchored synthetic gap candidates")
            )
            continue

        if regime == "event":
            if "DateTime" not in data.columns:
                raise ValueError("Event missingness requires a DateTime column to select daily maxima")
            y = pd.to_numeric(site_data[target], errors="coerce")
            dates = pd.to_datetime(site_data["DateTime"], errors="coerce").dt.date
            event_frame = pd.DataFrame({"Date": dates, "Value": y}, index=site_data.index)
            event_frame = event_frame.loc[event_frame["Value"].notna() & event_frame["Date"].notna()]
            candidates = event_frame.groupby("Date", sort=False)["Value"].idxmax().to_numpy()
            site_target_missing = int(round(len(candidates) * fraction))
            if fraction > 0 and site_target_missing == 0 and len(candidates):
                site_target_missing = 1
            if site_target_missing <= 0:
                exclusions.append(
                    dict(Region=region, Site=site, Target=target, Regime=regime, Seed=seed,
                         Requested_Missingness=fraction, Reason="insufficient event candidates")
                )
                continue
            chosen = rng.choice(candidates, site_target_missing, replace=False).tolist()
        else:
            _, proposed = apply_missingness(
                site_data.copy(), target, regime=regime, frac=fraction, seed=site_seed
            )
            proposed = pd.Series(np.asarray(proposed, dtype=bool), index=site_data.index)
            proposed &= site_data[target].notna()
            chosen = proposed[proposed].index.tolist()
            if regime in GAP_REGIME_BOUNDS:
                if not chosen:
                    exclusions.append(
                        dict(Region=region, Site=site, Target=target, Regime=regime, Seed=seed,
                             Requested_Missingness=fraction, Reason="no valid pure gaps selected")
                    )
                    continue
                assert_effective_gap_purity(proposed, original_missing, regime)
        combined.loc[chosen] = True
        local_mask = site_data.index.isin(chosen)
        diagnostics = effective_gap_mask_diagnostics(
            local_mask,
            original_missing.to_numpy(dtype=bool),
            regime,
            requested_fraction=fraction,
            observed_count=observed_count,
        )
        diagnostics_by_site[site] = diagnostics

    if not diagnostics_by_site:
        raise ValueError(
            "No eligible sites for %s/%s/%s %.0f%% seed=%d"
            % (region, target, regime, fraction * 100, seed)
        )
    per_site = combined.groupby(data["Site"]).sum().to_dict()
    logging.info(
        "%s %.0f%% seed=%d per-site artificial counts: %s",
        regime,
        fraction * 100,
        seed,
        per_site,
    )
    return combined, diagnostics_by_site, exclusions


def _metrics(actual, predicted, handle_negatives="exclude"):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[valid], predicted[valid]
    if not len(actual):
        return {**{name: np.nan for name in REGIONAL_METRIC_COLUMNS}, "N_Valid": 0}
    complete = evaluate_metrics(actual, predicted, handle_negative=handle_negatives)
    values = {short: complete.get(long, np.nan) for long, short in METRIC_COLUMN_MAP.items()}
    values["N_Valid"] = int(len(actual))
    return values


def _safe_token(value):
    return str(value).strip().replace(" ", "_").replace(".", "")


def _mask_sha256(data: pd.DataFrame, mask: pd.Series) -> str:
    """Return a stable fingerprint of the exact site/timestamp mask rows."""
    selected = data.loc[mask, ["Site", "DateTime"]].copy()
    selected.insert(0, "Row_Index", selected.index.astype(int))
    selected["DateTime"] = pd.to_datetime(selected["DateTime"], errors="coerce").map(
        lambda value: value.isoformat() if pd.notna(value) else "NaT"
    )
    payload = selected.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    strict_feature_list: bool = False,
    model_n_jobs: int = None,
    handle_negatives: str = "exclude",
) -> Dict[str, pd.DataFrame]:
    """Run one model on one pooled region and save predictions/metrics."""
    task_root = os.path.join(output_root, model_name, region.replace(" ", "_"), target.replace(".", ""))
    os.makedirs(task_root, exist_ok=True)
    data = regional_data.copy().sort_values(["Site", "DateTime"]).reset_index(drop=True)
    observed_counts = data.groupby("Site")[target].apply(lambda s: int(s.notna().sum()))
    zero_observed_sites = observed_counts[observed_counts <= 0]
    if not zero_observed_sites.empty:
        logging.warning(
            "Skipping %d regional study site(s) with no observed %s values for %s/%s: %s",
            len(zero_observed_sites),
            target,
            region,
            model_name,
            ", ".join(map(str, zero_observed_sites.index)),
        )
        data = (
            data.loc[data["Site"].isin(observed_counts[observed_counts > 0].index)]
            .copy()
            .reset_index(drop=True)
        )
        if data.empty:
            raise ValueError(
                "No regional study sites contain observed %s values for %s" % (target, region)
            )

    features = list(dict.fromkeys([f for f in features if f in data.columns and f != target] + ["SiteCode"]))
    parameter_values = dict(parameters or {})
    parameter_values["features"] = features
    parameters_json = json.dumps(parameter_values, sort_keys=True, default=str)
    feature_list = ",".join(features)
    data["SiteCode"] = pd.Categorical(data["Site"]).codes.astype(float)
    metric_rows, prediction_parts = [], []
    diagnostic_rows, exclusion_rows = [], []

    for regime in regimes:
        for fraction in missingness_levels:
            for seed in seeds:
                try:
                    mask, mask_diagnostics, exclusions = _balanced_region_mask(
                        data, target, regime, fraction, seed, region=region
                    )
                except ValueError as exc:
                    logging.warning("Skipping regional mask combination: %s", exc)
                    exclusion_rows.append(
                        dict(
                            Region=region,
                            Site="ALL",
                            Target=target,
                            Regime=regime,
                            Seed=seed,
                            Requested_Missingness=fraction,
                            Reason=str(exc),
                        )
                    )
                    continue
                exclusion_rows.extend(exclusions)
                mask_sha256 = _mask_sha256(data, mask)
                for site, diagnostics in mask_diagnostics.items():
                    diagnostic_rows.append(
                        dict(
                            Region=region,
                            Site=site,
                            Target=target,
                            Regime=regime,
                            Seed=seed,
                            Mask_SHA256=mask_sha256,
                            **diagnostics,
                        )
                    )
                work = data.copy()
                truth = pd.to_numeric(data[target], errors="coerce")
                work.loc[mask, target] = np.nan
                logging.info(
                    "Regional task %s/%s/%s | %s %.0f%% seed=%d | masked=%d eligible_sites=%d",
                    region,
                    target,
                    model_name,
                    regime,
                    fraction * 100,
                    seed,
                    int(mask.sum()),
                    len(mask_diagnostics),
                )
                imputer_kwargs = {
                    "site_name": region.replace(" ", "_"),
                    "model_name": "%s_%s" % (model_name, region.replace(" ", "_")),
                    "out_dir": task_root,
                    "missingness_regime": regime,
                    "missingness_fraction": fraction,
                    "seed": seed,
                    "strict_feature_list": bool(strict_feature_list),
                }
                if model_n_jobs is not None:
                    imputer_kwargs["n_jobs"] = int(model_n_jobs)
                result = impute_callable(
                    work,
                    target,
                    features,
                    custom_strategies=None,
                    **imputer_kwargs,
                )
                if result is None or target not in result.columns:
                    raise RuntimeError("%s returned no imputed target for %s" % (model_name, region))
                predicted = pd.to_numeric(result[target], errors="coerce").reindex(data.index)
                prediction = data.loc[mask, ["DateTime", "Region", "Site"]].copy()
                prediction["Target"] = target
                prediction["Model"] = model_name
                prediction["Regime"] = regime
                prediction["Missingness_Level"] = fraction
                prediction["Seed"] = seed
                prediction["Mask_SHA256"] = mask_sha256
                prediction["Observed"] = truth.loc[mask]
                prediction["Imputed"] = predicted.loc[mask]
                prediction["Was_Artificially_Masked"] = True
                prediction["Artificial_Gap_Length"] = 0
                prediction["Effective_Gap_Length"] = 0
                prediction["Artificial_Gap_In_Regime"] = np.nan
                prediction["Effective_Gap_In_Regime"] = np.nan
                bounds = GAP_REGIME_BOUNDS.get(regime)
                for site, site_data in data.groupby("Site", sort=False):
                    local_mask = mask.loc[site_data.index].to_numpy(dtype=bool)
                    original_missing = site_data[target].isna().to_numpy(dtype=bool)
                    artificial_lengths = mask_gap_lengths(local_mask)
                    effective_lengths = mask_gap_lengths(original_missing | local_mask)
                    selected = local_mask
                    prediction_rows = prediction.index[prediction["Site"].eq(site)]
                    prediction.loc[prediction_rows, "Artificial_Gap_Length"] = artificial_lengths[selected]
                    prediction.loc[prediction_rows, "Effective_Gap_Length"] = effective_lengths[selected]
                    if bounds:
                        prediction.loc[prediction_rows, "Artificial_Gap_In_Regime"] = (
                            (artificial_lengths[selected] >= bounds[0]) & (artificial_lengths[selected] <= bounds[1])
                        ).astype(float)
                        prediction.loc[prediction_rows, "Effective_Gap_In_Regime"] = (
                            (effective_lengths[selected] >= bounds[0]) & (effective_lengths[selected] <= bounds[1])
                        ).astype(float)
                prediction_parts.append(prediction)

                site_metrics = []
                for site, site_prediction in prediction.groupby("Site", sort=False):
                    values = _metrics(
                        site_prediction["Observed"],
                        site_prediction["Imputed"],
                        handle_negatives=handle_negatives,
                    )
                    diagnostics = mask_diagnostics[site]
                    row = dict(Region=region, Site=site, Target=target, Model=model_name,
                               Parameters=parameters_json, Features=feature_list,
                               Feature_Count=len(features), Regime=regime,
                               Missingness_Level=fraction, Missingness_Percent=fraction * 100,
                               Seed=seed, Mask_SHA256=mask_sha256,
                               N_Masked=len(site_prediction), Scope="Site",
                               **diagnostics, **values)
                    metric_rows.append(row)
                    site_metrics.append(row)
                micro = _metrics(
                    prediction["Observed"], prediction["Imputed"],
                    handle_negatives=handle_negatives,
                )
                masked_total = int(mask.sum())
                gap_total = sum(
                    item["Number_of_Artificial_Gaps"] for item in mask_diagnostics.values()
                )
                out_of_regime = sum(
                    item["Out_of_Regime_Effective_Points"] for item in mask_diagnostics.values()
                )
                nonempty = [
                    item for item in mask_diagnostics.values() if item["Number_of_Artificial_Gaps"]
                ]
                pooled_diagnostics = {
                    "Requested_Missingness": fraction,
                    "Achieved_Missingness": masked_total
                    / max(int(data[target].notna().sum()), 1),
                    "Number_of_Artificial_Gaps": gap_total,
                    "Min_Artificial_Gap": min((item["Min_Artificial_Gap"] for item in nonempty), default=0),
                    "Median_Artificial_Gap": float(
                        np.median([item["Median_Artificial_Gap"] for item in nonempty])
                    ) if nonempty else 0.0,
                    "Mean_Artificial_Gap": masked_total / gap_total if gap_total else 0.0,
                    "Max_Artificial_Gap": max((item["Max_Artificial_Gap"] for item in nonempty), default=0),
                    "Min_Effective_Gap": min((item["Min_Effective_Gap"] for item in nonempty), default=0),
                    "Median_Effective_Gap": float(
                        np.median([item["Median_Effective_Gap"] for item in nonempty])
                    ) if nonempty else 0.0,
                    "Mean_Effective_Gap": float(
                        np.mean([item["Mean_Effective_Gap"] for item in nonempty])
                    ) if nonempty else 0.0,
                    "Max_Effective_Gap": max((item["Max_Effective_Gap"] for item in nonempty), default=0),
                    "Isolated_Masked_Points": sum(
                        item["Isolated_Masked_Points"] for item in mask_diagnostics.values()
                    ),
                    "Out_of_Regime_Artificial_Points": sum(
                        item["Out_of_Regime_Artificial_Points"] for item in mask_diagnostics.values()
                    ),
                    "Out_of_Regime_Effective_Points": out_of_regime,
                    "Artificial_Gap_Purity": (
                        (masked_total - sum(
                            item["Out_of_Regime_Artificial_Points"] for item in mask_diagnostics.values()
                        )) / masked_total if masked_total else np.nan
                    ),
                    "Effective_Gap_Purity": (
                        (masked_total - out_of_regime) / masked_total
                        if masked_total else np.nan
                    ),
                }
                pooled_diagnostics.update(
                    {
                        "Number_of_Gaps": pooled_diagnostics["Number_of_Artificial_Gaps"],
                        "Min_Gap": pooled_diagnostics["Min_Artificial_Gap"],
                        "Median_Gap": pooled_diagnostics["Median_Artificial_Gap"],
                        "Mean_Gap": pooled_diagnostics["Mean_Artificial_Gap"],
                        "Max_Gap": pooled_diagnostics["Max_Artificial_Gap"],
                        "Out_of_Regime_Points": pooled_diagnostics["Out_of_Regime_Effective_Points"],
                        "Gap_Purity": pooled_diagnostics["Effective_Gap_Purity"],
                    }
                )
                metric_rows.append(dict(Region=region, Site="ALL", Target=target, Model=model_name,
                                        Parameters=parameters_json, Features=feature_list,
                                        Feature_Count=len(features), Regime=regime,
                                        Missingness_Level=fraction, Missingness_Percent=fraction * 100,
                                        Seed=seed, Mask_SHA256=mask_sha256,
                                        N_Masked=int(mask.sum()), Scope="Region_Micro",
                                        **pooled_diagnostics, **micro))
                macro_metrics = {
                    metric: _safe_mean([row[metric] for row in site_metrics])
                    for metric in REGIONAL_METRIC_COLUMNS
                }
                macro_metrics["N_Valid"] = int(sum(r["N_Valid"] for r in site_metrics))
                metric_rows.append(dict(
                    Region=region, Site="ALL", Target=target, Model=model_name, Regime=regime,
                    Parameters=parameters_json, Features=feature_list, Feature_Count=len(features),
                    Missingness_Percent=fraction * 100,
                    Missingness_Level=fraction, Seed=seed, Mask_SHA256=mask_sha256,
                    N_Masked=int(mask.sum()), Scope="Region_Macro",
                    **pooled_diagnostics, **macro_metrics,
                ))
                logging.info(
                    "Completed regional task %s/%s/%s | %s %.0f%% seed=%d | RMSE=%.4f R2=%.4f",
                    region,
                    target,
                    model_name,
                    regime,
                    fraction * 100,
                    seed,
                    micro.get("RMSE", np.nan),
                    micro.get("R2", np.nan),
                )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    metrics.to_csv(os.path.join(task_root, "metrics_by_site_and_region.csv"), index=False)
    predictions.to_csv(os.path.join(task_root, "masked_predictions_by_site.csv"), index=False)
    diagnostics = pd.DataFrame(diagnostic_rows)
    exclusions = pd.DataFrame(exclusion_rows)
    diagnostics.to_csv(os.path.join(task_root, "gap_mask_diagnostics.csv"), index=False)
    exclusions.to_csv(os.path.join(task_root, "site_target_exclusions.csv"), index=False)
    if plots_root and plot_types:
        with _PLOT_LOCK:
            _save_regional_plots(
                predictions, plots_root, region, target, model_name, list(plot_types), dpi=plot_dpi
            )
    return {"metrics": metrics, "predictions": predictions, "diagnostics": diagnostics, "exclusions": exclusions}
