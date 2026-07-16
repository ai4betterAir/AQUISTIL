"""
Comprehensive Model Evaluation Framework
Tests multiple feature configurations to determine optimal approach

Configurations tested:
1. Baseline: Only local features + temporal
2. Spatial_AI: Local + temporal + spatial features (CO, NO, NOX, PM10 from other stations)
3. Spatial_AI_Full: Local + temporal + ALL spatial features (including PM2.5)
4. Spatial_IDW: Local + temporal + IDW-based spatial feature
5. Spatial_Kriging: Local + temporal + Kriging-based spatial feature
6. Temporal_Only: Only temporal features (no spatial, minimal local)

Purpose:  Determine if spatial features genuinely improve performance
         and which spatial approach is best for practical deployment

Author: Dr.  Masrur
Date: 2026-01-21
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Import your modules
from evaluation_metrics import evaluate_metrics, evaluate_metrics_by_gap, METRIC_FUNCTIONS
from missingness_regimes import apply_missingness
import config_spatial as config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("comprehensive_evaluation.log"),
        logging.StreamHandler()
    ]
)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 300

# Suppress noisy sklearn/UserWarnings about feature names during iterative imputation
warnings.filterwarnings(
    'ignore',
    message='X does not have valid feature names',
    category=UserWarning
)

# Suppress FutureWarning from deprecated fillna(method=... ) if any older modules still use it
warnings.filterwarnings(
    'ignore',
    message="DataFrame.fillna with 'method' is deprecated",
    category=FutureWarning
)

# ============================================================================
# SPATIAL FEATURE GENERATORS
# ============================================================================

def compute_idw_feature(target_site, target_datetime_index, neighbor_data, coordinates, power=2, max_distance=100):
    """
    Compute IDW-based spatial feature (single column)
    
    Returns:
        pd.Series: IDW-interpolated values for each timestamp
    """
    from math import radians, sin, cos, sqrt, atan2
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c
    
    if target_site not in coordinates:
        logging.warning(f"Target site {target_site} not in coordinates")
        return pd.Series(np.nan, index=target_datetime_index)
    
    target_lat = coordinates[target_site]['lat']
    target_lon = coordinates[target_site]['lon']
    
    # Calculate distances and weights
    neighbor_weights = {}
    for site, data in neighbor_data.items():
        if site in coordinates and site != target_site:
            distance = haversine(target_lat, target_lon, 
                               coordinates[site]['lat'], 
                               coordinates[site]['lon'])
            if 0 < distance <= max_distance:
                neighbor_weights[site] = 1.0 / (distance ** power)
    
    if not neighbor_weights:
        logging.warning(f"No neighbors within {max_distance}km for {target_site}")
        return pd.Series(np.nan, index=target_datetime_index)
    
    # Compute weighted average for each timestamp
    idw_values = []
    for timestamp in target_datetime_index: 
        weighted_sum = 0
        weight_sum = 0
        
        for site, weight in neighbor_weights.items():
            if site in neighbor_data and timestamp in neighbor_data[site]. index:
                value = neighbor_data[site]. loc[timestamp]
                if pd.notna(value):
                    weighted_sum += value * weight
                    weight_sum += weight
        
        if weight_sum > 0:
            idw_values.append(weighted_sum / weight_sum)
        else:
            idw_values.append(np.nan)
    
    return pd.Series(idw_values, index=target_datetime_index)


def compute_kriging_feature(target_site, target_datetime_index, neighbor_data, coordinates, max_distance=100):
    """
    Compute Kriging-based spatial feature (single column)
    
    Returns:
        pd.Series: Kriging-interpolated values for each timestamp
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from math import radians, sin, cos, sqrt, atan2
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c
    
    if target_site not in coordinates: 
        return pd.Series(np.nan, index=target_datetime_index)
    
    target_lat = coordinates[target_site]['lat']
    target_lon = coordinates[target_site]['lon']
    
    # Find nearby neighbors
    nearby_sites = []
    for site in neighbor_data. keys():
        if site in coordinates and site != target_site: 
            distance = haversine(target_lat, target_lon,
                               coordinates[site]['lat'],
                               coordinates[site]['lon'])
            if distance <= max_distance:
                nearby_sites.append(site)
    
    if len(nearby_sites) < 3:
        # Fallback: if we have 1-2 neighbors, use IDW instead of failing kriging
        if len(nearby_sites) == 0:
            logging.warning(f"Not enough neighbors for kriging at {target_site}")
            return pd.Series(np.nan, index=target_datetime_index)
        else:
            logging.info(f"Only {len(nearby_sites)} neighbors available for kriging at {target_site}; falling back to IDW")
            return compute_idw_feature(target_site, target_datetime_index, neighbor_data, coordinates, power=2, max_distance=max_distance)
    
    # Compute kriging for each timestamp
    kriging_values = []
    kernel = ConstantKernel(1.0) * RBF(length_scale=10.0)
    
    for timestamp in target_datetime_index: 
        X_train = []
        y_train = []
        
        for site in nearby_sites:
            if site in neighbor_data and timestamp in neighbor_data[site].index:
                value = neighbor_data[site].loc[timestamp]
                if pd.notna(value):
                    X_train.append([coordinates[site]['lat'], coordinates[site]['lon']])
                    y_train.append(value)
        
        if len(y_train) >= 3:
            try:
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=0.1)
                gp. fit(X_train, y_train)
                pred = gp.predict([[target_lat, target_lon]])[0]
                kriging_values.append(pred)
            except:
                kriging_values.append(np. mean(y_train))
        else:
            kriging_values. append(np.nan)
    
    return pd. Series(kriging_values, index=target_datetime_index)


# ============================================================================
# FEATURE CONFIGURATION BUILDER
# ============================================================================

def load_neighbor_data(target_site, target_variable="PM2.5"):
    """Load PM2.5 data from all neighbor sites"""
    neighbor_data = {}
    
    for filename in os.listdir(config. INPUT_DIRECTORY):
        if not filename.endswith('.csv'):
            continue
        
        site_name = filename.split('_')[0]
        if site_name == target_site: 
            continue
        
        try:
            filepath = os.path.join(config.INPUT_DIRECTORY, filename)
            df = pd.read_csv(filepath, low_memory=False)
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df.set_index('DateTime', inplace=True)
            
            if target_variable in df.columns:
                neighbor_data[site_name] = pd.to_numeric(df[target_variable], errors='coerce')
        except Exception as e:
            logging.debug(f"Could not load {filename}: {e}")
            continue
    
    return neighbor_data


def build_feature_set(data, target_site, target_column, config_name):
    """
    Build feature set based on configuration name
    
    Args:
        data: DataFrame with DateTime and target column
        target_site: Name of target site
        target_column:  Target variable name
        config_name:  One of: 
            - "Baseline": Local + temporal only
            - "Spatial_AI": Local + temporal + spatial (no PM2.5)
            - "Spatial_AI_Full":  Local + temporal + spatial (with PM2.5)
            - "Spatial_IDW": Local + temporal + IDW feature
            - "Spatial_Kriging": Local + temporal + Kriging feature
            - "Temporal_Only": Only temporal features
    
    Returns:
        DataFrame with features
    """
    from spatial import add_temporal_features, load_spatial_features
    
    df = data.copy()
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    feature_list = []
    
    # ========================================================================
    # Configuration 1: Baseline (Local + Temporal)
    # ========================================================================
    if config_name == "Baseline":
        # Local predictors
        for col in config.INPUT_COLUMNS:
            if col in df.columns and col != target_column:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                feature_list.append(col)
        
        # Temporal features
        df = add_temporal_features(df, datetime_column='DateTime')
        temporal_cols = ['Hour', 'Day', 'Month', 'DayOfWeek', 'DayOfYear', 'WeekOfYear',
                        'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos']
        feature_list.extend([c for c in temporal_cols if c in df.columns])
    
    # ========================================================================
    # Configuration 2: Spatial_AI (No PM2.5 from other stations)
    # ========================================================================
    elif config_name == "Spatial_AI":
        # Local predictors
        for col in config.INPUT_COLUMNS: 
            if col in df.columns and col != target_column: 
                df[col] = pd. to_numeric(df[col], errors='coerce')
                feature_list.append(col)
        
        # Temporal features
        df = add_temporal_features(df, datetime_column='DateTime')
        temporal_cols = ['Hour', 'Day', 'Month', 'DayOfWeek', 'DayOfYear', 'WeekOfYear',
                        'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos']
        feature_list. extend([c for c in temporal_cols if c in df.columns])
        
        # Spatial features (exclude PM2.5)
        df_temp = df.set_index('DateTime')
        spatial_features = load_spatial_features(
            config.INPUT_DIRECTORY,
            target_site,
            target_column,  # Will be excluded by pollutants_except_target mode
            df_temp.index
        )

        # Merge spatial features in a single concat to avoid fragmented inserts
        if spatial_features is not None and not spatial_features.empty:
            merged = pd.concat([df_temp, spatial_features], axis=1)
            # reset back to original df shape with DateTime column
            df = merged.reset_index()
            # add all spatial columns to feature list (they should already exclude target)
            for col in spatial_features.columns:
                feature_list.append(col)
    
    # ========================================================================
    # Configuration 3: Spatial_AI_Full (With PM2.5 from other stations)
    # ========================================================================
    elif config_name == "Spatial_AI_Full":
        # Local predictors
        for col in config.INPUT_COLUMNS:
            if col in df.columns and col != target_column:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                feature_list.append(col)
        
        # Temporal features
        df = add_temporal_features(df, datetime_column='DateTime')
        temporal_cols = ['Hour', 'Day', 'Month', 'DayOfWeek', 'DayOfYear', 'WeekOfYear',
                        'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos']
        feature_list.extend([c for c in temporal_cols if c in df.columns])
        
        # Spatial features (include PM2.5)
        df_temp = df.set_index('DateTime')
        
        # Temporarily override config
        original_mode = config.SPATIAL_FEATURE_MODE
        config. SPATIAL_FEATURE_MODE = "all_pollutants"
        
        spatial_features = load_spatial_features(
            config.INPUT_DIRECTORY,
            target_site,
            target_column,
            df_temp.index
        )

        config.SPATIAL_FEATURE_MODE = original_mode  # Restore

        if spatial_features is not None and not spatial_features.empty:
            merged = pd.concat([df_temp, spatial_features], axis=1)
            df = merged.reset_index()
            for col in spatial_features.columns:
                feature_list.append(col)
    
    # ========================================================================
    # Configuration 4: Spatial_IDW (Local + Temporal + IDW feature)
    # ========================================================================
    elif config_name == "Spatial_IDW":
        # Local predictors
        for col in config.INPUT_COLUMNS: 
            if col in df.columns and col != target_column:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                feature_list.append(col)
        
        # Temporal features
        df = add_temporal_features(df, datetime_column='DateTime')
        temporal_cols = ['Hour', 'Day', 'Month', 'DayOfWeek', 'DayOfYear', 'WeekOfYear',
                        'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos']
        feature_list.extend([c for c in temporal_cols if c in df.columns])
        
        # IDW spatial feature
        neighbor_data = load_neighbor_data(target_site, target_column)
        if neighbor_data and config.SITE_COORDINATES:
            df_temp = df.set_index('DateTime')
            idw_feature = compute_idw_feature(
                target_site,
                df_temp.index,
                neighbor_data,
                config. SITE_COORDINATES,
                power=2,
                max_distance=100
            )
            # assign as Series aligned by index in one operation
            df_temp['IDW_Spatial_PM25'] = idw_feature
            df = df_temp.reset_index()
            feature_list.append('IDW_Spatial_PM25')
    
    # ========================================================================
    # Configuration 5: Spatial_Kriging (Local + Temporal + Kriging feature)
    # ========================================================================
    elif config_name == "Spatial_Kriging": 
        # Local predictors
        for col in config.INPUT_COLUMNS:
            if col in df. columns and col != target_column: 
                df[col] = pd.to_numeric(df[col], errors='coerce')
                feature_list.append(col)
        
        # Temporal features
        df = add_temporal_features(df, datetime_column='DateTime')
        temporal_cols = ['Hour', 'Day', 'Month', 'DayOfWeek', 'DayOfYear', 'WeekOfYear',
                        'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos']
        feature_list.extend([c for c in temporal_cols if c in df.columns])
        
        # Kriging spatial feature
        neighbor_data = load_neighbor_data(target_site, target_column)
        if neighbor_data and config. SITE_COORDINATES:
            df_temp = df.set_index('DateTime')
            kriging_feature = compute_kriging_feature(
                target_site,
                df_temp.index,
                neighbor_data,
                config.SITE_COORDINATES,
                max_distance=100
            )
            df_temp['Kriging_Spatial_PM25'] = kriging_feature
            df = df.reset_index()
            feature_list.append('Kriging_Spatial_PM25')
    
    # ========================================================================
    # Configuration 6: Temporal_Only (Minimal baseline)
    # ========================================================================
    elif config_name == "Temporal_Only":
        # Only temporal features
        df = add_temporal_features(df, datetime_column='DateTime')
        temporal_cols = ['Hour', 'Day', 'Month', 'DayOfWeek', 'DayOfYear', 'WeekOfYear',
                        'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos']
        feature_list.extend([c for c in temporal_cols if c in df.columns])
    
    else:
        raise ValueError(f"Unknown configuration: {config_name}")
    
    # Clean features
    for col in feature_list:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values in features
        df[feature_list] = df[feature_list].ffill().bfill().fillna(0)
    
    logging.info(f"  {config_name}:  {len(feature_list)} features")
    
    return df, feature_list


# ============================================================================
# MODEL EVALUATION WITH DIFFERENT CONFIGURATIONS
# ============================================================================

def evaluate_configuration(data, target_site, target_column, config_name, model_function, 
                           regime, miss_level, seed=42):
    """
    Evaluate a single configuration
    
    Args:  
        data: DataFrame with original data
        target_site: Site name
        target_column: Target variable
        config_name: Configuration name
        model_function:   Imputation function to use
        regime: Missingness regime
        miss_level:   Missingness fraction
        seed: Random seed
    
    Returns:
        dict:   Evaluation metrics
    """
    # Build features for this configuration
    df_features, feature_list = build_feature_set(data, target_site, target_column, config_name)
    
    # Apply missingness
    original_missing = df_features[target_column].isna()
    df_with_missing, simulated_mask = apply_missingness(
        df_features,
        target_column,
        regime=regime,
        frac=miss_level,
        seed=seed
    )
    
    simulated_mask = simulated_mask & (~original_missing)
    
    if simulated_mask.sum() == 0:
        logging.warning(f"No simulated missing values for {config_name}")
        return None
    
    # ✅ FIX: Pass site_name and model_name to impute function
    try:
        df_imputed = model_function(
            df_with_missing,
            target_column,
            feature_list,
            custom_strategies=config.CUSTOM_STRATEGIES,
            site_name=target_site,  # ✅ ADD THIS
            model_name=f"LightGBM_{target_site}",  # ✅ ADD THIS
            out_dir=str(Path("comprehensive_evaluation_results"))  # ✅ ADD THIS
        )
    except Exception as e:
        logging.error(f"Imputation failed for {config_name}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Evaluate
    true_values = df_features. loc[simulated_mask, target_column]. values
    imputed_values = df_imputed.loc[simulated_mask, target_column].values
    
    # Remove invalid values
    valid_mask = np.isfinite(true_values) & np.isfinite(imputed_values)
    if config.HANDLE_NEGATIVES == 'exclude':
        valid_mask = valid_mask & (true_values >= 0) & (imputed_values >= 0)
    
    if valid_mask.sum() < 10:
        logging.warning(f"Too few valid values for {config_name}")
        return None
    
    true_clean = true_values[valid_mask]
    imputed_clean = imputed_values[valid_mask]
    
    # Compute metrics
    metrics = evaluate_metrics(true_clean, imputed_clean, handle_negative=config.HANDLE_NEGATIVES)
    
    metrics. update({
        'Site': target_site,
        'Configuration': config_name,
        'Regime': regime,
        'Missingness_Pct': miss_level * 100,
        'N_Features': len(feature_list),
        'N_Samples': len(true_clean)
    })
    
    return metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_comprehensive_evaluation(site_file, target_column="PM2.5", model_name="LightGBM"):
    """
    Run comprehensive evaluation for a single site
    
    Tests all 6 configurations:
    1. Baseline
    2. Spatial_AI
    3. Spatial_AI_Full
    4. Spatial_IDW
    5. Spatial_Kriging
    6. Temporal_Only
    """
    site_name = os.path.basename(site_file).split('. ')[0].split('_')[0]
    
    logging.info(f"\n{'='*80}")
    logging.info(f"COMPREHENSIVE EVALUATION:  {site_name}")
    logging.info(f"{'='*80}")
    
    # Load data
    data = pd.read_csv(site_file)
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    
    if target_column not in data.columns:
        logging.error(f"Target column {target_column} not found")
        return
    
    # Load model
    if model_name == "LightGBM": 
        from Model.LightGBM import impute_mice as model_function
    elif model_name == "LGBM_AQ_Plus_SpatialIter":
        from Model.LGBM_AQ_Plus_SpatialIter import impute_mice as model_function
    else:
        logging.error(f"Unknown model: {model_name}")
        return
    
    # Configurations to test
    configurations = [
        "Baseline",
        "Spatial_AI",
        "Spatial_AI_Full",
        "Spatial_IDW",
        "Spatial_Kriging",
        "Temporal_Only"
    ]
    
    all_results = []
    
    # Test each configuration
    for config_name in configurations:
        logging.info(f"\nTesting configuration: {config_name}")
        
        for regime in config.MISSINGNESS_REGIMES:
            for miss_level in config. MISSINGNESS_LEVELS:
                logging.info(f"  {regime} @ {int(miss_level*100)}%")
                
                metrics = evaluate_configuration(
                    data,
                    site_name,
                    target_column,
                    config_name,
                    model_function,
                    regime,
                    miss_level,
                    seed=42
                )
                
                if metrics: 
                    all_results.append(metrics)
                    logging.info(f"    RMSE: {metrics. get('Root Mean Squared Error (RMSE)', np.nan):.2f}, "
                               f"R:  {metrics.get('Correlation Coefficient (R)', np.nan):.3f}")
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_dir = Path("comprehensive_evaluation_results")
        output_dir. mkdir(exist_ok=True)
        
        output_file = output_dir / f"{site_name}_{target_column}_{model_name}_comprehensive.csv"
        results_df.to_csv(output_file, index=False)
        logging.info(f"\n✅ Results saved to: {output_file}")
        
        return results_df
    
    return None


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_configuration_comparison(results_df, output_dir="comprehensive_evaluation_plots"):
    """
    Generate comparison plots across configurations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: RMSE by configuration
    fig, ax = plt.subplots(figsize=(12, 6))
    
    config_order = ["Temporal_Only", "Baseline", "Spatial_IDW", "Spatial_Kriging", 
                   "Spatial_AI", "Spatial_AI_Full"]
    
    rmse_by_config = results_df.groupby('Configuration')['Root Mean Squared Error (RMSE)'].agg(['mean', 'std'])
    rmse_by_config = rmse_by_config.reindex([c for c in config_order if c in rmse_by_config. index])
    
    x = np.arange(len(rmse_by_config))
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen']
    
    bars = ax.bar(x, rmse_by_config['mean'], yerr=rmse_by_config['std'], 
                  capsize=5, color=colors[: len(rmse_by_config)], 
                  edgecolor='black', linewidth=1)
    
    ax.set_xticks(x)
    ax.set_xticklabels(rmse_by_config.index, rotation=45, ha='right')
    ax.set_ylabel('RMSE (μg/m³)', fontweight='bold')
    ax.set_title('Performance Comparison Across Feature Configurations', 
                fontweight='bold', fontsize=14)
    try:
        ax.grid(axis='y', alpha=0.3)
    except ValueError as e:
        logging.warning(f"Failed to set grid on RMSE plot: {e}")
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "rmse_by_configuration.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: R coefficient by configuration
    fig, ax = plt.subplots(figsize=(12, 6))
    
    r_by_config = results_df.groupby('Configuration')['Correlation Coefficient (R)'].agg(['mean', 'std'])
    r_by_config = r_by_config.reindex([c for c in config_order if c in r_by_config.index])
    
    bars = ax.bar(x, r_by_config['mean'], yerr=r_by_config['std'],
                  capsize=5, color=colors[:len(r_by_config)],
                  edgecolor='black', linewidth=1)
    
    ax.set_xticks(x)
    ax.set_xticklabels(r_by_config.index, rotation=45, ha='right')
    ax.set_ylabel('Correlation Coefficient (R)', fontweight='bold')
    ax.set_title('Correlation Comparison Across Feature Configurations',
                fontweight='bold', fontsize=14)
    ax.set_ylim(0, 1.0)
    try:
        ax.grid(axis='y', alpha=0.3)
    except ValueError as e:
        logging.warning(f"Failed to set grid on R plot: {e}")
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "r_by_configuration.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Regime-wise comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.flatten()
    
    for idx, regime in enumerate(config. MISSINGNESS_REGIMES):
        ax = axes[idx]
        
        df_regime = results_df[results_df['Regime'] == regime]
        rmse_regime = df_regime. groupby('Configuration')['Root Mean Squared Error (RMSE)'].mean()
        rmse_regime = rmse_regime.reindex([c for c in config_order if c in rmse_regime.index])
        
        x_regime = np.arange(len(rmse_regime))
        ax.bar(x_regime, rmse_regime. values, color=colors[: len(rmse_regime)], 
              edgecolor='black', linewidth=1)
        
        ax.set_title(regime.replace('_', ' ').title(), fontweight='bold')
        ax.set_xticks(x_regime)
        ax.set_xticklabels(rmse_regime. index, rotation=45, ha='right', fontsize=8)
        
        if idx % 3 == 0:
            ax.set_ylabel('RMSE (μg/m³)', fontweight='bold')
        
        try:
            ax.grid(axis='y', alpha=0.3)
        except ValueError as e:
            logging.warning(f"Failed to set grid on regime plot for {regime}: {e}")
    
    # Remove extra subplot
    fig.delaxes(axes[-1])
    
    fig.suptitle('Performance by Missingness Regime', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "rmse_by_regime.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Statistical significance test
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Pairwise t-tests
    configs = results_df['Configuration'].unique()
    n = len(configs)
    p_matrix = np.ones((n, n))
    
    for i, config1 in enumerate(configs):
        for j, config2 in enumerate(configs):
            if i >= j:
                continue
            
            vals1 = results_df[results_df['Configuration'] == config1]['Root Mean Squared Error (RMSE)'].dropna()
            vals2 = results_df[results_df['Configuration'] == config2]['Root Mean Squared Error (RMSE)'].dropna()
            
            if len(vals1) > 1 and len(vals2) > 1:
                try:
                    _, p = stats.ttest_ind(vals1, vals2)
                    p_matrix[i, j] = p
                    p_matrix[j, i] = p
                except:
                    pass
    
    sns.heatmap(p_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
               xticklabels=configs, yticklabels=configs,
               vmin=0, vmax=0.10, center=0.05,
               cbar_kws={'label': 'p-value'}, ax=ax)
    
    ax.set_title('Statistical Significance (p-values)\nGreen = Significant Difference (p<0.05)',
                fontweight='bold', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(output_dir / "statistical_significance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"✅ Plots saved to {output_dir}")


def generate_summary_report(results_df):
    """
    Generate text summary report
    """
    report = []
    report.append("\n" + "="*80)
    report.append("COMPREHENSIVE EVALUATION SUMMARY")
    report.append("="*80)
    
    # Overall ranking
    report.append("\n1. OVERALL RANKING (by RMSE)")
    report.append("-" * 80)
    
    overall_perf = results_df.groupby('Configuration').agg({
        'Root Mean Squared Error (RMSE)': ['mean', 'std'],
        'Correlation Coefficient (R)': ['mean', 'std'],
        'Nash-Sutcliffe Efficiency (NSE)': ['mean', 'std']
    }).round(3)
    
    overall_perf = overall_perf.sort_values(('Root Mean Squared Error (RMSE)', 'mean'))
    
    for idx, (config, row) in enumerate(overall_perf.iterrows(), 1):
        report.append(f"\n{idx}.  {config}")
        report.append(f"   RMSE: {row[('Root Mean Squared Error (RMSE)', 'mean')]:.2f} ± {row[('Root Mean Squared Error (RMSE)', 'std')]:.2f} μg/m³")
        report.append(f"   R:     {row[('Correlation Coefficient (R)', 'mean')]:.3f} ± {row[('Correlation Coefficient (R)', 'std')]:.3f}")
        report.append(f"   NSE:  {row[('Nash-Sutcliffe Efficiency (NSE)', 'mean')]:.3f} ± {row[('Nash-Sutcliffe Efficiency (NSE)', 'std')]:.3f}")
    
    # Key findings
    report.append("\n" + "="*80)
    report.append("2. KEY FINDINGS")
    report.append("="*80)
    
    baseline_rmse = results_df[results_df['Configuration'] == 'Baseline']['Root Mean Squared Error (RMSE)'].mean()
    
    for config in ['Spatial_AI', 'Spatial_AI_Full', 'Spatial_IDW', 'Spatial_Kriging']: 
        if config in results_df['Configuration']. values:
            config_rmse = results_df[results_df['Configuration'] == config]['Root Mean Squared Error (RMSE)'].mean()
            improvement = (baseline_rmse - config_rmse) / baseline_rmse * 100
            
            if improvement > 0:
                report.append(f"\n✅ {config}:  {improvement:.1f}% improvement over Baseline")
            else:
                report.append(f"\n❌ {config}: {-improvement:.1f}% worse than Baseline")
    
    # Spatial PM2.5 leakage check
    if 'Spatial_AI_Full' in results_df['Configuration'].values:
        ai_full_rmse = results_df[results_df['Configuration'] == 'Spatial_AI_Full']['Root Mean Squared Error (RMSE)'].mean()
        ai_rmse = results_df[results_df['Configuration'] == 'Spatial_AI']['Root Mean Squared Error (RMSE)'].mean()
        
        if ai_full_rmse < ai_rmse * 0.7:  # More than 30% improvement
            report.append("\n⚠️  WARNING: Including spatial PM2.5 causes large performance jump")
            report.append("   This suggests strong spatial correlation (potential leakage effect)")
            report.append("   Recommend using Spatial_AI (without PM2.5) for fair evaluation")
    
    # Recommendation
    report.append("\n" + "="*80)
    report.append("3. RECOMMENDATION FOR PRODUCTION")
    report.append("="*80)
    
    best_config = overall_perf.index[0]
    best_rmse = overall_perf.iloc[0][('Root Mean Squared Error (RMSE)', 'mean')]
    
    report.append(f"\n🎯 Best configuration: {best_config}")
    report.append(f"   Expected RMSE: {best_rmse:.2f} μg/m³")
    
    if 'Spatial_AI_Full' in best_config:
        report.append("\n   ⚠️  Note: This configuration uses PM2.5 from other stations")
        report.append("   Only use if neighbor station data will be reliably available in production")
    
    report.append("\n" + "="*80)
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save to file
    with open("comprehensive_evaluation_summary.txt", "w") as f:
        f.write(report_text)
    
    logging.info("✅ Summary report saved to comprehensive_evaluation_summary.txt")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║         COMPREHENSIVE MODEL EVALUATION FRAMEWORK                     ║
    ║                                                                      ║
    ║  Tests 6 configurations:                                              ║
    ║    1. Temporal_Only - Only time patterns                             ║
    ║    2. Baseline - Local + Temporal features                           ║
    ║    3. Spatial_IDW - Baseline + IDW spatial feature                   ║
    ║    4. Spatial_Kriging - Baseline + Kriging spatial feature           ║
    ║    5. Spatial_AI - Baseline + AI spatial features (no PM2.5)         ║
    ║    6. Spatial_AI_Full - Baseline + AI spatial (with PM2.5)           ║
    ║                                                                      ║
    ║  Purpose: Determine if spatial features genuinely improve            ║
    ║           performance and which approach is best                     ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Get site to process
    if config.TARGET_SITES:
        sites_to_process = config.TARGET_SITES
    else:
        from spatial import get_available_sites
        sites_to_process = get_available_sites(config.INPUT_DIRECTORY)
    
    # Process each site
    all_results = []
    
    for site in sites_to_process:
        # Find site file
        site_file = None
        for filename in os.listdir(config. INPUT_DIRECTORY):
            if filename.lower().startswith(site.lower()) and filename.endswith('.csv'):
                site_file = os.path.join(config.INPUT_DIRECTORY, filename)
                break
        
        if site_file:
            results_df = run_comprehensive_evaluation(
                site_file, 
                target_column="PM2.5",
                model_name="LightGBM"  # or "LGBM_AQ_Plus_SpatialIter"
            )
            
            if results_df is not None:
                all_results.append(results_df)
        else:
            logging.warning(f"File not found for site: {site}")
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        combined_results. to_csv("comprehensive_evaluation_all_sites.csv", index=False)
        logging.info("✅ Combined results saved to comprehensive_evaluation_all_sites.csv")
        
        # Generate visualizations
        plot_configuration_comparison(combined_results)
        
        # Generate summary report
        generate_summary_report(combined_results)
        
        print("\n" + "="*80)
        print("✅ COMPREHENSIVE EVALUATION COMPLETE")
        print("="*80)
        print("\n📊 Results saved to:")
        print("   - comprehensive_evaluation_results/")
        print("   - comprehensive_evaluation_all_sites.csv")
        print("   - comprehensive_evaluation_summary.txt")
        print("   - comprehensive_evaluation_plots/")
    else:
        logging.error("No results generated")


if __name__ == "__main__":
    main()