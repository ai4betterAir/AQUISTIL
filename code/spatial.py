import os
import pandas as pd
import numpy as np
import logging
import pickle
import hashlib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging. INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Simple in-memory cache for spatial features to avoid recomputing during a run
_spatial_features_cache = {}

# Additional memory cache and disk cache helpers
_spatial_cache_memory = {}

def _spatial_cache_dir():
    try:
        # lazy import config to avoid import-time cycles
        import codigal.config_spatial as cfg
        base = getattr(cfg, 'OUTPUT_DIRECTORY', None)
    except Exception:
        base = None
    if not base:
        base = os.environ.get('SPATIAL_CACHE_DIR', '/tmp/spatial_cache')
    cache_dir = Path(base) / '.cache_spatial'
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return cache_dir

def _spatial_cache_key(target_site, variables, max_distance, max_sites, input_directory):
    key_parts = [str(os.path.abspath(input_directory)), str(target_site), ','.join(map(str, sorted(variables))), str(float(max_distance)), str(int(max_sites))]
    raw = '|'.join(key_parts)
    return hashlib.sha1(raw.encode('utf-8')).hexdigest()

def _save_spatial_cache(cache_key, df):
    try:
        _spatial_cache_memory[cache_key] = df.copy()
        p = _spatial_cache_dir() / f"{cache_key}.pkl"
        # use pandas to_pickle for robustness
        df.to_pickle(p)
        return True
    except Exception as e:
        logging.debug(f"Could not save spatial cache to disk: {e}")
        return False

def _load_spatial_cache(cache_key):
    # memory first
    if cache_key in _spatial_cache_memory:
        return _spatial_cache_memory[cache_key].copy()
    # disk fallback
    try:
        p = _spatial_cache_dir() / f"{cache_key}.pkl"
        if p.exists():
            try:
                df = pd.read_pickle(p)
                _spatial_cache_memory[cache_key] = df.copy()
                return df.copy()
            except Exception:
                return None
    except Exception:
        return None
    return None

def extract_site_name(filename):
    """
    Extract site name from filename.  
    
    Args: 
        filename (str): CSV filename
        
    Returns: 
        str: Site name
    """
    # Assuming format:  SiteName_*. csv
    return filename.split('_')[0]


def load_spatial_features(input_directory, target_site, target_variable, current_datetime_index, max_distance=None):
    """
    Load spatial features from other sites, constrained by distance and selected variable list.

    Returns a DataFrame with columns named spatial_<site>_<variable>.
    Caches results in-memory to avoid repeated expensive loads for the same selection.
    """
    import os
    import logging
    import numpy as np
    import pandas as pd
    import config_spatial as config

    # Helper: haversine distance in km
    def haversine_km(lat1, lon1, lat2, lon2):
        from math import radians, sin, cos, sqrt, atan2
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2.0)**2 + cos(lat1) * cos(lat2) * sin(dlon/2.0)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    # Use config value if not passed
    if max_distance is None:
        max_distance = getattr(config, "MAX_SPATIAL_DISTANCE", 100)
    try:
        max_distance = float(max_distance)
    except Exception:
        max_distance = 100.0

    # Normalize coordinate dict keys (uppercase) for matching
    site_coords = getattr(config, "SITE_COORDINATES", {}) or {}
    coords_normalized = {str(k).strip().upper(): v for k, v in site_coords.items() if isinstance(k, str)}

    # Robustly map target_site -> coordinate key
    raw_target = str(target_site).strip()
    def _norm_token(s):
        return ''.join(ch for ch in str(s).upper() if ch.isalnum())
    candidates = set()
    candidates.add(raw_target.upper())
    parts = [raw_target.split('_')[0], raw_target.split('-')[0]]
    for p in parts:
        if p:
            candidates.add(p.upper())
            candidates.add(_norm_token(p))
    candidates.add(_norm_token(raw_target))
    for suf in ['AQMS', 'PROCESSED', 'STATION', 'SITE']:
        if raw_target.upper().endswith('_' + suf):
            candidates.add(raw_target.upper().rsplit('_' + suf, 1)[0])
            candidates.add(_norm_token(raw_target.upper().rsplit('_' + suf, 1)[0]))

    matched_key = None
    for cand in candidates:
        if cand in coords_normalized:
            matched_key = cand
            break
    if matched_key is None:
        # substring fallback
        for ck in coords_normalized.keys():
            for cand in candidates:
                if cand and cand in ck:
                    matched_key = ck
                    break
            if matched_key:
                break

    if matched_key:
        coord_key = matched_key
        logging.info(f"Spatial site matching: raw_target='{raw_target}' resolved to coord_key='{coord_key}'")
    else:
        coord_key = None
        logging.info(f"Spatial site matching: raw_target='{raw_target}' could not be resolved to coordinates (will skip distance filtering)")

    # Determine variables to load according to mode (and restrict by INPUT_COLUMNS when sensible)
    mode = getattr(config, 'SPATIAL_FEATURE_MODE', 'target_only')
    cfg_spatial_cols = getattr(config, 'SPATIAL_FEATURE_COLUMNS', []) or []
    cfg_input_cols = getattr(config, 'INPUT_COLUMNS', []) or []

    if mode == "target_only":
        variables_to_load = [target_variable]

    elif mode == "pollutants_except_target":
        all_pollutants = ["CO", "NO", "NOX", "PM10", "PM2.5", "SO2", "O3", "NO2"]
        if cfg_spatial_cols:
            variables_to_load = [v for v in cfg_spatial_cols if v != target_variable]
        else:
            # restrict to pollutants that are in INPUT_COLUMNS to avoid loading unwanted pollutants
            variables_to_load = [v for v in all_pollutants if v != target_variable and v in cfg_input_cols]

    elif mode == "all_pollutants":
        variables_to_load = cfg_spatial_cols.copy() if cfg_spatial_cols else ["CO", "NO", "NOX", "PM10", "PM2.5", "SO2", "O3", "NO2"]

    elif mode == "all_meteorological":
        variables_to_load = ["HUMID", "TEMP", "WSP", "RAIN", "WDR", "WGU"]

    elif mode == "all":
        variables_to_load = config.INPUT_COLUMNS.copy()

    elif mode == "custom":
        variables_to_load = getattr(config, 'SPATIAL_FEATURE_COLUMNS', [target_variable])

    else:
        variables_to_load = [target_variable]

    logging.info(f"Loading spatial features for {target_site} (mode={mode}) -> variables: {variables_to_load}")

    # ------------------------
    # Caching: avoid repeated expensive loads for same (input_dir, site, vars, distance, max_sites)
    # Uses both in-memory and optional on-disk cache (best-effort)
    # ------------------------
    try:
        def _simple_norm_token(s):
            return ''.join(ch for ch in str(s).upper() if ch.isalnum())
        target_token = _simple_norm_token(raw_target)
        vars_tuple = tuple(sorted(variables_to_load))
        max_sites_cfg = getattr(config, "MAX_SPATIAL_SITES", 0)
        cache_key_hex = _spatial_cache_key(raw_target, variables_to_load, float(max_distance), int(max_sites_cfg), input_directory)

        cached_df = _load_spatial_cache(cache_key_hex)
        if cached_df is not None:
            logging.info(f"Spatial cache hit for target='{raw_target}' vars={vars_tuple} max_distance={max_distance} max_sites={max_sites_cfg}")
            try:
                return cached_df.reindex(current_datetime_index).copy()
            except Exception:
                return cached_df.copy()
    except Exception:
        cache_key_hex = None
        max_sites_cfg = getattr(config, "MAX_SPATIAL_SITES", 0)

    # Build allowed_sites (site_key -> distance). If coord missing => allow all.
    allowed_sites = None
    if coord_key and coord_key in coords_normalized:
        tlat = coords_normalized[coord_key]['lat']
        tlon = coords_normalized[coord_key]['lon']
        distances = {}
        for s_key, v in coords_normalized.items():
            if s_key == coord_key:
                continue
            try:
                d = haversine_km(tlat, tlon, v['lat'], v['lon'])
                if 0 < d <= max_distance:
                    distances[s_key] = d
            except Exception:
                continue
        # limit by MAX_SPATIAL_SITES if configured
        max_sites_cfg = getattr(config, "MAX_SPATIAL_SITES", 0)
        if max_sites_cfg and max_sites_cfg > 0:
            sorted_items = sorted(distances.items(), key=lambda x: x[1])
            distances = dict(sorted_items[:max_sites_cfg])
        allowed_sites = distances
        logging.info(f"Distance filter: target='{coord_key}' max_distance={max_distance}km -> {len(allowed_sites)} candidate sites")
    else:
        allowed_sites = None

    # Iterate files and collect per-variable series only for allowed sites
    col_series = {}
    loaded_sites = []
    loaded_variable_counts = {var: 0 for var in variables_to_load}

    for filename in os.listdir(input_directory):
        if not filename.lower().endswith('.csv'):
            continue

        site_name = extract_site_name(filename).strip()
        site_key = site_name.upper()

        # Skip self
        if coord_key and site_key == coord_key:
            continue

        # If allowed_sites configured, skip sites not in allowed_sites
        if allowed_sites is not None:
            if not allowed_sites:
                continue
            if site_key not in allowed_sites:
                # try normalized key match as well
                if _norm_token(site_key) not in allowed_sites and site_key not in allowed_sites:
                    continue

        filepath = os.path.join(input_directory, filename)
        try:
            df = pd.read_csv(filepath, low_memory=False)
            # header heuristics
            try:
                preview = pd.read_csv(filepath, nrows=12, header=None, low_memory=False)
                preview_text = preview.astype(str).apply(lambda col: ' '.join(col.dropna().values), axis=0).str.cat(sep=' ')
                if '1h average' in preview_text or 'average [' in preview_text or 'µg/m' in preview_text:
                    header_row = None
                    for i in range(len(preview)):
                        row_vals = preview.iloc[i].astype(str).str.strip().tolist()
                        if any((str(v).strip() == 'DateTime') or (str(v).strip().lower().startswith('datetime')) for v in row_vals):
                            header_row = i
                            break
                    if header_row is not None:
                        df = pd.read_csv(filepath, header=header_row, low_memory=False)
                    else:
                        df = pd.read_csv(filepath, skiprows=2, low_memory=False)
            except Exception:
                pass

            # Ensure DateTime index
            if 'DateTime' not in df.columns:
                if 'Date' in df.columns and 'Time' in df.columns:
                    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M', errors='coerce')
                elif 'datetime' in df.columns:
                    df['DateTime'] = pd.to_datetime(df['datetime'], errors='coerce')
                else:
                    logging.debug(f"No DateTime in {filename}; skipping")
                    continue
            else:
                df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

            df = df.loc[df['DateTime'].notna()]
            if df.empty:
                continue

            df.set_index('DateTime', inplace=True)

            if not any(v in df.columns for v in variables_to_load):
                continue

            if df.index.duplicated().any():
                try:
                    df_numeric = df.copy()
                    for col in df_numeric.columns:
                        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
                    df = df_numeric.groupby(df_numeric.index).mean()
                except Exception:
                    try:
                        df = df.groupby(df.index).agg(lambda s: pd.to_numeric(s, errors='coerce').mean())
                    except Exception:
                        df = df.loc[~df.index.duplicated(keep='first')]

            site_loaded_any = False
            for variable in variables_to_load:
                if variable not in df.columns:
                    continue
                df[variable] = pd.to_numeric(df[variable], errors='coerce')
                try:
                    df_aligned = df[[variable]].reindex(current_datetime_index)
                except Exception:
                    df_aligned = df[[variable]].loc[~df.index.duplicated(keep='first')].reindex(current_datetime_index)

                col_name = f"spatial_{site_name}_{variable}"
                val = df_aligned[variable]

                try:
                    if isinstance(val, pd.DataFrame):
                        num_cols = val.select_dtypes(include=[np.number]).columns.tolist()
                        if num_cols:
                            chosen = val[num_cols[0]]
                            col_series[col_name] = pd.to_numeric(chosen, errors='coerce')
                        else:
                            first_col = val.iloc[:, 0]
                            col_series[col_name] = pd.to_numeric(first_col, errors='coerce')
                    else:
                        arr = val.values if hasattr(val, "values") else np.array(val)
                        if getattr(arr, "ndim", 1) > 1:
                            ser = pd.Series(arr[:, 0], index=current_datetime_index)
                            col_series[col_name] = pd.to_numeric(ser, errors='coerce')
                        else:
                            ser = pd.Series(val, index=current_datetime_index)
                            col_series[col_name] = pd.to_numeric(ser, errors='coerce')
                except Exception as e:
                    logging.warning(f"Failed to normalize spatial column {col_name}: {e}; skipping")
                    continue

                loaded_variable_counts[variable] += 1
                site_loaded_any = True

            if site_loaded_any:
                loaded_sites.append(site_name)

        except Exception as e:
            logging.warning(f"Could not load spatial features from {filename}: {e}")
            continue

    # Concatenate into DataFrame
    if col_series:
        normalized = {}
        for k, v in col_series.items():
            try:
                if isinstance(v, pd.Series):
                    normalized[k] = v
                elif isinstance(v, pd.DataFrame):
                    num_cols = v.select_dtypes(include=[np.number]).columns.tolist()
                    if num_cols:
                        normalized[k] = pd.to_numeric(v[num_cols[0]], errors='coerce')
                    else:
                        normalized[k] = pd.to_numeric(v.iloc[:, 0], errors='coerce')
                else:
                    arr = np.array(v)
                    if arr.ndim > 1:
                        arr = arr[:, 0]
                    ser = pd.Series(arr, index=current_datetime_index)
                    normalized[k] = pd.to_numeric(ser, errors='coerce')
            except Exception as e:
                logging.warning(f"Failed to normalize spatial series {k}: {e}; skipping")
        spatial_features = pd.concat(normalized, axis=1) if normalized else pd.DataFrame(index=current_datetime_index)
    else:
        spatial_features = pd.DataFrame(index=current_datetime_index)

    # Optional defensive filter: keep only spatial columns for variables_to_load
    try:
        if variables_to_load:
            allowed_vars = set(variables_to_load)
            cols_to_keep = [c for c in spatial_features.columns if any(c.endswith(f"_{v}") for v in allowed_vars)]
            if cols_to_keep:
                spatial_features = spatial_features[cols_to_keep]
    except Exception:
        pass

    # Prevent leakage: drop any columns that include the target variable name (just in case)
    target_cols = [col for col in spatial_features.columns if target_variable in col]
    if target_cols:
        logging.warning(f"⚠️ DATA LEAKAGE WARNING: {len(target_cols)} columns contain '{target_variable}' from other sites! Dropping them.")
        try:
            spatial_features = spatial_features.drop(columns=target_cols, errors='ignore')
            logging.info(f"Dropped {len(target_cols)} leaking spatial columns")
        except Exception as e:
            logging.error(f"Failed to drop leaking columns: {e}")

    # Cache store (best-effort) — save to in-memory + disk cache using hashed key
    try:
        if 'cache_key_hex' in locals() and cache_key_hex is not None:
            _spatial_cache_memory[cache_key_hex] = spatial_features.copy()
            _save_spatial_cache(cache_key_hex, spatial_features)
            logging.info(f"Saved spatial features to cache for target='{raw_target}' vars={vars_tuple} max_distance={max_distance} max_sites={max_sites_cfg}")
    except Exception:
        logging.debug("Failed to store spatial features in cache", exc_info=True)

    # Final summary
    logging.info(f"✅ Loaded spatial features from {len(loaded_sites)} sites (after filters).")
    for var, count in {k: v for k, v in loaded_variable_counts.items() if v > 0}.items():
        logging.info(f"   {var}: {count} sites")
    if len(loaded_sites) > 0:
        logging.info(f"   Sites: {', '.join(loaded_sites[:5])}" + (f" and {len(loaded_sites)-5} more" if len(loaded_sites) > 5 else ""))

    return spatial_features

def add_temporal_features(data, datetime_column='DateTime'):
    """
    Add temporal features to the dataset.
    
    Args:
        data (pd.DataFrame): Input data
        datetime_column (str): Name of the datetime column
        
    Returns:  
        pd.DataFrame: Data with added temporal features
    """
    logging.info("Adding temporal features...")
    
    df = data.copy()
    
    # Ensure datetime column is datetime type
    if datetime_column in df.columns:
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        datetime_index = df[datetime_column]
    elif df. index.name == datetime_column or isinstance(df.index, pd.DatetimeIndex):
        datetime_index = df.index
    else:
        raise ValueError(f"DateTime column '{datetime_column}' not found in data")
    
    # Extract temporal features (handle both Series and DatetimeIndex)
    if isinstance(datetime_index, pd.Series):
        df['Hour'] = datetime_index.dt. hour
        df['Day'] = datetime_index.dt.day
        df['Month'] = datetime_index.dt.month
        df['DayOfWeek'] = datetime_index.dt.dayofweek
        df['DayOfYear'] = datetime_index.dt.dayofyear
        week = datetime_index.dt.isocalendar().week
        df['WeekOfYear'] = week. fillna(0).astype(int)
    else:
        df['Hour'] = datetime_index. hour
        df['Day'] = datetime_index.day
        df['Month'] = datetime_index. month
        df['DayOfWeek'] = datetime_index. dayofweek
        df['DayOfYear'] = datetime_index.dayofyear
        week = datetime_index.isocalendar().week
        df['WeekOfYear'] = week.fillna(0).astype(int)
    
    # Cyclical encoding for better representation
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np. cos(2 * np.pi * df['Hour'] / 24)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    temporal_features = ['Hour', 'Day', 'Month', 'DayOfWeek', 'DayOfYear', 'WeekOfYear',
                        'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 
                        'DayOfWeek_sin', 'DayOfWeek_cos']
    
    logging.info(f"Added {len(temporal_features)} temporal features")
    
    return df


def add_lagged_features(data, target_column, lags=[1, 2, 3, 6, 12, 24]):
    """
    Add lagged features for time series context.
    
    Args:
        data (pd.DataFrame): Input data
        target_column (str): Column to create lags for
        lags (list): List of lag values (in hours)
        
    Returns: 
        pd.DataFrame: Data with added lagged features
    """
    logging.info(f"Adding lagged features for {target_column}...")
    
    df = data.copy()
    
    if target_column not in df.columns:
        logging.warning(f"Target column '{target_column}' not found, skipping lag features")
        return df
    
    for lag in lags:
        df[f'{target_column}_lag_{lag}'] = df[target_column]. shift(lag)
    
    logging.info(f"Added {len(lags)} lagged features")
    
    return df


def add_rolling_features(data, target_column, windows=[3, 6, 12, 24]):
    """
    Add rolling statistics features.
    
    Args:
        data (pd.DataFrame): Input data
        target_column (str): Column to create rolling features for
        windows (list): List of window sizes (in hours)
        
    Returns: 
        pd.DataFrame: Data with added rolling features
    """
    logging.info(f"Adding rolling features for {target_column}...")
    
    df = data.copy()
    
    if target_column not in df. columns:
        logging.warning(f"Target column '{target_column}' not found, skipping rolling features")
        return df
    
    for window in windows:
        df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window=window, min_periods=1).mean()
        df[f'{target_column}_rolling_std_{window}'] = df[target_column]. rolling(window=window, min_periods=1).std()
        df[f'{target_column}_rolling_min_{window}'] = df[target_column].rolling(window=window, min_periods=1).min()
        df[f'{target_column}_rolling_max_{window}'] = df[target_column].rolling(window=window, min_periods=1).max()
    
    logging.info(f"Added {len(windows) * 4} rolling features")
    
    return df


def prepare_spatial_temporal_data(data, target_column, input_columns, spatial_config):
    """
    Prepare data with spatial and temporal features.
    Robustly drops any feature columns that are 100% NaN before returning.
    """
    logging.info("Preparing spatial-temporal data...")
    
    df = data.copy()
    all_features = input_columns.copy()
    
    # Ensure DateTime column exists and is properly formatted
    if 'DateTime' not in df.columns:
        raise ValueError("DateTime column is required for spatial-temporal features")
    
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)
    
    # Add spatial features (from other sites)
    if spatial_config.get('use_spatial', False):
        input_directory = spatial_config.get('input_directory')
        target_site = spatial_config.get('target_site')
        
        if input_directory and target_site:
            spatial_features = load_spatial_features(
                input_directory,
                target_site,
                target_column,
                df.index
            )
            
            # Merge spatial features (single concat to avoid fragmentation)
            if not spatial_features.empty:
                df = pd.concat([df, spatial_features], axis=1)
                all_features.extend(spatial_features.columns.tolist())

                # Apply linear interpolation to spatial features to reduce missingness
                spatial_cols = [col for col in spatial_features.columns if col.startswith('spatial_')]
                if spatial_cols:
                    logging.info("Applying linear interpolation to spatial features...")
                    try:
                        df[spatial_cols] = df[spatial_cols].interpolate(method='linear', limit_direction='both')
                    except Exception as e:
                        logging.warning("Interpolation failed for spatial features: %s", e)

            else:
                logging.info("No spatial features loaded to merge.")
        else:
            logging.warning("Spatial features requested but configuration incomplete")
    
    # Reset index to get DateTime back as a column
    df = df.reset_index()
    
    # Add temporal features
    if spatial_config.get('use_temporal', True):
        df = add_temporal_features(df, datetime_column='DateTime')
        temporal_features = ['Hour', 'Day', 'Month', 'DayOfWeek', 'DayOfYear', 'WeekOfYear',
                            ]
        # temporal_features = ['Hour', 'Day', 'Month', 'DayOfWeek', 'DayOfYear', 'WeekOfYear',
        #                     'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 
        #                     'DayOfWeek_sin', 'DayOfWeek_cos']
        all_features.extend(temporal_features)
    
    # Add lagged features
    if spatial_config.get('use_lagged', False):
        lag_values = spatial_config.get('lag_values', [1, 2, 3, 6, 12, 24])
        df = add_lagged_features(df, target_column, lags=lag_values)
        lagged_features = [f'{target_column}_lag_{lag}' for lag in lag_values]
        all_features.extend(lagged_features)
    
    # Add rolling features
    if spatial_config.get('use_rolling', False):
        rolling_windows = spatial_config.get('rolling_windows', [3, 6, 12, 24])
        df = add_rolling_features(df, target_column, windows=rolling_windows)
        rolling_features = []
        for window in rolling_windows:
            rolling_features.extend([
                f'{target_column}_rolling_mean_{window}',
                f'{target_column}_rolling_std_{window}',
                f'{target_column}_rolling_min_{window}',
                f'{target_column}_rolling_max_{window}'
            ])
        all_features.extend(rolling_features)
    
    # ------------------------------
    # Drop columns that are entirely NaN BEFORE imputation (prevents sklearn skipping columns)
    # ------------------------------
    cols_in_df = [c for c in all_features if c in df.columns and c != target_column]
    all_na_cols = [c for c in cols_in_df if df[c].dropna().empty]
    if all_na_cols:
        logging.debug(
            "Dropping %d feature columns that contain NO observed values (all-NaN). "
            "These would cause sklearn imputers to skip columns and change output shapes. "
            "Dropped columns (first 20 shown): %s",
            len(all_na_cols),
            all_na_cols[:20]
        )
        try:
            df.drop(columns=all_na_cols, inplace=True, errors='ignore')
        except Exception as e:
            logging.debug("Failed to drop all-NaN columns cleanly: %s", e)
        all_features = [c for c in all_features if c not in all_na_cols]
    # ------------------------------
    
    # Filter to only available features (excluding target column)
    available_features = [col for col in all_features if col in df.columns and col != target_column]
    
    logging.info(
        "Shared preprocessing prepared %d features; model-specific features "
        "may be added after this stage.",
        len(available_features),
    )
    
    return df, available_features

def get_available_sites(input_directory):
    """
    Get list of available sites from input directory.
    
    Args:
        input_directory (str): Directory containing CSV files
        
    Returns: 
        list: List of site names
    """
    sites = []
    for filename in os.listdir(input_directory):
        if filename.lower().endswith('.csv'):
            site_name = extract_site_name(filename)
            sites.append(site_name)
    
    return sorted(sites)


def calculate_spatial_distances(sites_metadata):
    """
    Calculate distances between sites (if coordinates are available).
    
    Args:
        sites_metadata (dict): Dictionary with site coordinates
            {
                'site_name': {'lat': float, 'lon': float},
                ...
            }
    
    Returns:
        pd. DataFrame: Distance matrix between sites
    """
    from math import radians, sin, cos, sqrt, atan2
    
    def haversine(lat1, lon1, lat2, lon2):
        """Calculate distance between two points on Earth (in km)"""
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    sites = list(sites_metadata.keys())
    n = len(sites)
    distances = np.zeros((n, n))
    
    for i, site1 in enumerate(sites):
        for j, site2 in enumerate(sites):
            if i != j:
                lat1, lon1 = sites_metadata[site1]['lat'], sites_metadata[site1]['lon']
                lat2, lon2 = sites_metadata[site2]['lat'], sites_metadata[site2]['lon']
                distances[i, j] = haversine(lat1, lon1, lat2, lon2)
    
    distance_df = pd.DataFrame(distances, index=sites, columns=sites)
    
    return distance_df


def weight_spatial_features_by_distance(spatial_features, target_site, site_coordinates, max_distance=100):
    """
    Weight spatial features by inverse distance. 
    
    Args:
        spatial_features (pd.DataFrame): DataFrame with spatial features
        target_site (str): Target site name
        site_coordinates (dict): Dictionary with site coordinates {'site':  {'lat': x, 'lon': y}}
        max_distance (float): Maximum distance to consider (km)
        
    Returns:  
        pd.DataFrame: Weighted spatial features
    """
    from math import radians, sin, cos, sqrt, atan2
    
    def haversine(lat1, lon1, lat2, lon2):
        """Calculate distance between two points on Earth (in km)"""
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c
    
    if target_site not in site_coordinates:
        logging.warning(f"Target site {target_site} not found in coordinates, skipping weighting")
        return spatial_features
    
    weighted_features = spatial_features.copy()
    target_lat = site_coordinates[target_site]['lat']
    target_lon = site_coordinates[target_site]['lon']
    
    for col in spatial_features.columns:
        if col.startswith('spatial_'):
            # Extract site name from column (format: spatial_<site>_<variable>)
            parts = col.split('_')
            if len(parts) >= 3:
                site_name = parts[1]
                
                if site_name in site_coordinates:
                    other_lat = site_coordinates[site_name]['lat']
                    other_lon = site_coordinates[site_name]['lon']
                    distance = haversine(target_lat, target_lon, other_lat, other_lon)
                    
                    # Apply inverse distance weighting
                    if 0 < distance <= max_distance: 
                        weight = 1 / distance
                        weighted_features[col] = weighted_features[col] * weight
                        logging.debug(f"Weighted {col} by distance {distance:.1f} km (weight={weight:.4f})")
                    elif distance > max_distance:
                        # Exclude sites beyond max_distance
                        weighted_features = weighted_features.drop(columns=[col])
                        logging.debug(f"Excluded {col} (distance {distance:.1f} km > max {max_distance} km)")
    
    return weighted_features


# Example usage and testing
if __name__ == "__main__":
    # Test the spatial module
    logging.info("Testing spatial module...")
    
    # Example configuration
    test_config = {
        'input_directory': '/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Imputation/Imputation_model/input/Processed_Data_NSW_ALL',
        'target_site': 'Chullora',
        'use_spatial': False,
        'use_temporal': True,
        'use_lagged': False,
        'use_rolling':  False
    }
    
    # Get available sites
    if os.path.exists(test_config['input_directory']):
        sites = get_available_sites(test_config['input_directory'])
        logging.info(f"Found {len(sites)} sites: {', '.join(sites[: 10])}" + 
                    (f" and {len(sites)-10} more" if len(sites) > 10 else ""))
    else:
        logging.warning(f"Input directory not found: {test_config['input_directory']}")
