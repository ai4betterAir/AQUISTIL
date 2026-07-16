"""
..  module:: Sparse_LSTM_forecast
    :platform: Unix
    :synopsis: Definition of the basic object class to output .

.. moduleauthor:: Xavier Barthelemy <xavier.barthelemy@environment.nsw.gov.au>
.. moduleauthor:: Hubert Nguyen <hubert.nguyen@environment.nsw.gov.au>
.. moduleauthor:: Sagthitharan Karalasingham <d9630120@umail.usq.edu.au>
   
"""

import sys
import os
import stat
import copy
import json
import glob
import zipfile
import datetime as dtime
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
import time
import logging
import importlib
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass
import Core_iHPC.Tools.InitLogging as IL
# from sklearn.preprocessing import MinMaxScaler


import Core_iHPC.Inputs.Input_manager as IM
import Core_iHPC.Inputs.Yaml_file_reader as YFR
import Core_iHPC.Outputs.Output_manager as OM
import Core_iHPC.Configuration.Configuration as CC
import Core_iHPC.Processing.Impute as Impute
import Core_iHPC.Processing.decomposition.decomp_joint_preparation as VJPrep
import Core_iHPC.Visualization.plot as TSPlots
import Core_iHPC.Visualization.Forecast_plot as ForecastPlots
import Core_iHPC.Configuration.DPE_region_stations as DPERT
import Core_iHPC.Evaluation_training as EV
import Core_iHPC.Models.model_loader as MODEL_LOADER

ALL_REGION_SENTINELS = {"ALL", "AUTO", "ALL_AVAILABLE", "*", "SLSYD"}
DEFAULT_TRAIN_MODEL = True #Set to True only when you want to train; forecast/dashboard runs should default to False.
PIPELINE_MODE = "imf_joint" # Valid values: "raw", "imf_joint", "imf_per_imf", "custom" 
#(if PIPELINE_MODE= imf_joint, select USE_ONE_MODEL_PER_IMF= no; if PIPELINE_MODE= imf_per_imf, select USE_ONE_MODEL_PER_IMF= yes; if PIPELINE_MODE= custom, select the two switches below directly).
# In "custom" mode, the two switches below are used directly.
USE_DECOMPOSE = "yes" #Use IMF decomposition in the data preprocessing step. Set to "no" to skip it and feed imputed data directly into the model.
USE_ONE_MODEL_PER_IMF = "no" #Use one model per IMF. Set to "no" to train one joint model on all IMFs together.
TYPE_OF_DECOMPOSITION = "VMD" # Valid values: "VMD", "EMD", ...
# Change this one line to switch the default region for all runs.
# Use all-region sentinels like "ALL", "AUTO", "ALL_AVAILABLE", "*", or "SLSYD" to run every region.
DEFAULT_SELECTED_REGION = "CE_Sydney" # Available region keys: "CC_Coast", "CE_Sydney", "CN_Table", "Lower_Hunter", "NW_Sydney", "Newcastle", "SR_Table", "SW_Sydney", "Sydney", "Upper_Hunter"; use the all-region sentinels above for every region.
DEFAULT_VAR_TO_PREDICT = ["O3"] #, "PM2.5", "PM10"] #["O3", "PM2.5", "PM10"]
DEFAULT_FORECAST_METHOD = "Sparse_LSTM_v1"
AVAILABLE_IMPUTATION_METHODS = ["NONE", "MICE", "KNN", "TemporalMICE", "AQUISTIL"]
IMPUTATION_METHOD = "AQUISTIL"
DEFAULT_IMPUTATION_METHOD = "AQUISTIL"
DEFAULT_PLOT = True

PIPELINE_MODE_PRESETS = {
    "raw": {"use_decompose": False, "use_one_model_per_imf": False},
    "imf_joint": {"use_decompose": True, "use_one_model_per_imf": False},
    "imf_per_imf": {"use_decompose": True, "use_one_model_per_imf": True},
}

PIPELINE_MODE_ALIASES = {
    "vmd_joint": "imf_joint",
    "vmd_per_imf": "imf_per_imf",
}

DECOMPOSITION_BACKEND_ALIASES = {
    "VMD": "VMD",
    "EMD": "EMD",
}

def resolve_decomposition_backend_name():
    backend = str(TYPE_OF_DECOMPOSITION or "").strip().upper()
    return DECOMPOSITION_BACKEND_ALIASES.get(backend, backend)

def load_decomposition_backend():
    backend_name = resolve_decomposition_backend_name()
    try:
        return importlib.import_module(f"Core_iHPC.Processing.decomposition.{backend_name}")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Decomposition backend {backend} is not available. "
            "Add Core_iHPC/Processing/decomposition/{backend}.py implementing the same interface.".format(
                backend=backend_name,
            )
        ) from exc

DECOMPOSITION_BACKEND = load_decomposition_backend()

# Edit model choices here. This is the single place that maps each target
# pollutant to the Python module to execute and the checkpoint folder name.
TARGET_MODEL_CONFIG = {
    "O3": {"module_name": "Sparse_LSTM_v1","weights_name": "sparse_lstm_o3_{lags}",},
    "PM2.5": {"module_name": "Sparse_LSTM_v2","weights_name": "sparse_lstm_pm25_{lags}",},
    "PM10": {"module_name": "Sparse_LSTM_v1","weights_name": "sparse_lstm_pm10_{lags}",},
}

BASE_DIR = "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Nowcasting/cnn_lstm_forecast"
DEFAULT_RUNTIME_BASE_DIR = BASE_DIR
AI_RUNS_DIR_NAME = "AI_Runs_decomposition"
DEFAULT_MODEL_BASE_PATH = os.path.join(AI_RUNS_DIR_NAME, "Model_weights")
API_INPUT_ROOT = os.path.join(BASE_DIR, "API_Input")
LOG_DIR = os.path.join(BASE_DIR, AI_RUNS_DIR_NAME, "logs")
RUNTIME_LOCATION_TEMPLATES = [
    os.path.join("API_Input", "Inputs"),
    os.path.join("API_Input", "model_data", "Imputed_data"),
    os.path.join("AI_dashboard_files_imf_joint"),
    os.path.join(AI_RUNS_DIR_NAME, "Forecast", "{region}", "{target}", "{model}", "{simulation}"),
    os.path.join(AI_RUNS_DIR_NAME, "Forecast", "{region}", "{target}", "{model}", "{simulation}", "Plots"),
    os.path.join(AI_RUNS_DIR_NAME, "Forecast", "{region}", "{target}", "{model}", "{simulation}", "status"),
    os.path.join(AI_RUNS_DIR_NAME, "Forecast", "{region}", "{target}", "{model}", "{simulation}", "Dashboard"),
    os.path.join(AI_RUNS_DIR_NAME, "Training", "{region}", "{target}", "{model}", "Plots"),
    os.path.join(AI_RUNS_DIR_NAME, "Training", "{region}", "{target}", "{model}", "Plots", "Histogram"),
    os.path.join(AI_RUNS_DIR_NAME, "Training", "{region}", "{target}", "{model}", "Plots", "Training_loss"),
]
RUNTIME_LOCATION_PATHS = [os.path.join(BASE_DIR, template) for template in RUNTIME_LOCATION_TEMPLATES]

def flatten_config_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]

    output = []
    for item in value:
        if isinstance(item, (list, tuple)):
            output.extend(flatten_config_list(item))
        else:
            output.append(item)
    return output

def resolve_master_toggle(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default

def resolve_pipeline_mode_settings():
    mode = PIPELINE_MODE_ALIASES.get(str(PIPELINE_MODE).strip().lower(), str(PIPELINE_MODE).strip().lower())
    if mode == "custom":
        return mode, resolve_master_toggle(USE_DECOMPOSE, False), resolve_master_toggle(USE_ONE_MODEL_PER_IMF, False)
    if mode not in PIPELINE_MODE_PRESETS:
        raise ValueError(
            "Unsupported PIPELINE_MODE={mode}. Choose one of: {choices}".format(
                mode=PIPELINE_MODE,
                choices=", ".join(sorted(list(PIPELINE_MODE_PRESETS.keys()) + ["custom"] + list(PIPELINE_MODE_ALIASES.keys()))),
            )
        )
    preset = PIPELINE_MODE_PRESETS[mode]
    return mode, preset["use_decompose"], preset["use_one_model_per_imf"]

PIPELINE_MODE_RESOLVED, MASTER_USE_DECOMPOSE, MASTER_USE_ONE_MODEL_PER_IMF = resolve_pipeline_mode_settings()

def pollutant_model_token(target):
    return {
        "O3": "o3",
        "PM2.5": "pm25",
        "PM10": "pm10",
    }.get(target, target.lower().replace(".", "").replace(" ", ""))

def log_target_token(target):
    token = str(target).strip().upper()
    if token in {"OZONE", "O3"}:
        return "O3"
    if token in {"PM2.5", "PM25", "PM2_5"}:
        return "PM2.5"
    if token == "PM10":
        return "PM10"
    return token

def infer_targets_from_config_name(config_name, yaml_file_dict=None):
    """
    Infer pollutant target(s) from a config filename when var_to_predict is
    omitted.

    Historical config files in Tools/Config_testing often encode the pollutant
    in the basename only. Falling back to the old global default would run
    O3/PM2.5/PM10 for those configs, which is incorrect.
    """
    explicit_targets = flatten_config_list((yaml_file_dict or {}).get("var_to_predict"))
    if explicit_targets:
        return [log_target_token(target) for target in explicit_targets]

    basename = os.path.basename(str(config_name))
    for suffix in (".yaml", ".yml"):
        if basename.lower().endswith(suffix):
            basename = basename[: -len(suffix)]
            break
    basename = basename.upper()
    if any(token in basename for token in ("PM2.5", "PM25", "PM2_5")):
        return ["PM2.5"]
    if "PM10" in basename:
        return ["PM10"]
    if any(token in basename for token in ("OZONE", "O3")):
        return ["O3"]

    forecast_method = str((yaml_file_dict or {}).get("model_parameters", {}).get("forecast_method", "")).upper()
    if any(token in forecast_method for token in ("PM2.5", "PM25")):
        return ["PM2.5"]
    if "PM10" in forecast_method:
        return ["PM10"]
    if any(token in forecast_method for token in ("OZONE", "O3")):
        return ["O3"]

    return []


def build_model_name(target):
    return TARGET_MODEL_CONFIG.get(target, {}).get("module_name", DEFAULT_FORECAST_METHOD)


def build_model_weights_name(target, lags):
    template = TARGET_MODEL_CONFIG.get(target, {}).get("weights_name")
    if template is None:
        template = "sparse_lstm_{target}_{lags}"
    return template.format(
        target=pollutant_model_token(target),
        lags=lags,
    )


def is_sparse_model_method(model_name):
    return str(model_name or "").strip().lower().startswith("sparse_lstm")


def resolve_forecast_method(yaml_file_dict, target=None):
    if target is not None:
        return build_model_name(target)
    explicit_method = (
        (yaml_file_dict or {}).get("forecast_method")
        or ((yaml_file_dict or {}).get("model_parameters") or {}).get("forecast_method")
    )
    if explicit_method:
        return str(explicit_method).strip()
    return DEFAULT_FORECAST_METHOD


def resolve_imputation_method(yaml_file_dict):
    method = (
        (yaml_file_dict or {}).get("imputation_method")
        or (yaml_file_dict or {}).get("imputation")
        or DEFAULT_IMPUTATION_METHOD
    )
    return str(method).strip()


def resolve_model_name(yaml_file_dict, target=None, forecast_method=None, lags=None):
    if target is not None:
        resolved_forecast_method = forecast_method or resolve_forecast_method(yaml_file_dict, target)
        if is_sparse_model_method(resolved_forecast_method):
            if lags is None:
                lags = (yaml_file_dict or {}).get("lags", (yaml_file_dict or {}).get("model_parameters", {}).get("n_outputs", 24))
            return build_model_weights_name(target, lags)
        return build_model_name(target)

    explicit_model_name = (yaml_file_dict or {}).get("model_name")
    if explicit_model_name:
        return str(explicit_model_name).strip()

    resolved_forecast_method = forecast_method or resolve_forecast_method(yaml_file_dict, target)
    if is_sparse_model_method(resolved_forecast_method):
        if lags is None:
            lags = (yaml_file_dict or {}).get("lags", (yaml_file_dict or {}).get("model_parameters", {}).get("n_outputs", 24))
        if target is not None:
            return build_model_weights_name(target, lags)
        return DEFAULT_FORECAST_METHOD

    if resolved_forecast_method:
        return str(resolved_forecast_method).strip()
    return DEFAULT_FORECAST_METHOD


def load_model_module_for_target(target, forecast_method=None, allow_fallback=True):
    model_name = forecast_method if forecast_method else build_model_name(target)
    module_path = f"Core_iHPC.Models.{model_name}"

    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError:
        if allow_fallback and target is not None:
            fallback_module = build_model_name(target)
            if fallback_module != model_name:
                try:
                    return importlib.import_module(f"Core_iHPC.Models.{fallback_module}")
                except ModuleNotFoundError:
                    pass
        return importlib.import_module("Core_iHPC.Models.Sparse_LSTM_v1")


def assert_target_model_consistency(target, forecast_method, model_name, enforce=True):
    if not enforce:
        return
    target_token = log_target_token(target)
    expected_module = build_model_name(target_token)

    normalized_forecast_method = str(forecast_method or "").strip()
    normalized_model_name = str(model_name or "").strip().lower()

    if normalized_forecast_method != expected_module:
        raise ValueError(
            "Resolved target/model mismatch: target={target} expects module={expected}, "
            "but runtime forecast_method={actual}".format(
                target=target_token,
                expected=expected_module,
                actual=normalized_forecast_method,
            )
        )


def _copy_dataframe(obj):
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return obj.copy(deep=True)
    return obj


def _copy_dataframe_dict(data):
    if data is None:
        return None
    return {key: value.copy(deep=True) for key, value in data.items()}


def apply_runtime_overrides(yaml_file_dict, target=None, region=None):
    yaml_for_run = copy.deepcopy(yaml_file_dict)

    if target is not None:
        yaml_for_run["var_to_predict"] = [target]
        existing_inputs = flatten_config_list(yaml_for_run.get("additional_var_to_select"))
        if target not in existing_inputs:
            existing_inputs = existing_inputs + [target]
        yaml_for_run["additional_var_to_select"] = existing_inputs

        lags = yaml_for_run.get("lags", yaml_for_run.get("model_parameters", {}).get("n_outputs", 24))
        yaml_for_run.setdefault("forecast_method", resolve_forecast_method(yaml_for_run, target))
        yaml_for_run.setdefault(
            "model_name",
            resolve_model_name(
                yaml_for_run,
                target=target,
                forecast_method=yaml_for_run.get("forecast_method"),
                lags=lags,
            ),
        )

    if region is not None:
        yaml_for_run["selected_region"] = region

    return yaml_for_run


def resolve_regions_for_target(selected_region, dpe_region):
    selected_regions = flatten_config_list(selected_region)
    if not selected_regions:
        selected_regions = ["ALL"]

    region_map = get_region_stations_dict(dpe_region)
    if isinstance(region_map, tuple):
        region_map = get_region_stations_dict(region_map)

    if any(str(region).upper() in ALL_REGION_SENTINELS for region in selected_regions):
        return list(region_map.keys())

    resolved_regions = []
    for region in selected_regions:
        resolved_region = resolve_region_name(dpe_region, region)
        if resolved_region not in region_map:
            available = ", ".join(region_map.keys())
            raise ValueError(
                "Region {region} is not available for target {target}. "
                "Available regions: {available}".format(
                    region=region,
                    target=getattr(dpe_region, "target", "unknown"),
                    available=available,
                )
            )
        resolved_regions.append(resolved_region)

    return resolved_regions


def get_region_stations_dict(dpe_region):
    if isinstance(dpe_region, tuple):
        for item in dpe_region:
            if isinstance(item, dict):
                return item
            if isinstance(item, tuple):
                try:
                    return get_region_stations_dict(item)
                except TypeError:
                    pass
            if hasattr(item, "DPE_region_stations_dict"):
                candidate = item.DPE_region_stations_dict
                if isinstance(candidate, dict):
                    return candidate
                if isinstance(candidate, tuple):
                    return get_region_stations_dict(candidate)
        raise TypeError(f"Unsupported tuple format for DPE region map: {dpe_region!r}")
    if isinstance(dpe_region, dict):
        return dpe_region
    if hasattr(dpe_region, "DPE_region_stations_dict"):
        candidate = dpe_region.DPE_region_stations_dict
        if isinstance(candidate, dict):
            return candidate
        if isinstance(candidate, tuple):
            return get_region_stations_dict(candidate)
    raise TypeError(f"Unsupported DPE region map type: {type(dpe_region)!r}")


def resolve_region_name(dpe_region, selected_region):
    if isinstance(dpe_region, tuple):
        for item in dpe_region:
            if hasattr(item, "resolve_region_name"):
                return item.resolve_region_name(selected_region)
        return selected_region
    if hasattr(dpe_region, "resolve_region_name"):
        return dpe_region.resolve_region_name(selected_region)
    return selected_region


_VMD_IMF_NAME_RE = re.compile(r"^imf[\s_-]*(\d+)$", re.IGNORECASE)


def normalize_vmd_station_frame(station_df):
    """
    Normalize VMD column labels loaded from cache or returned by preprocessing.

    This keeps the model-facing schema stable even if older cached files use
    `IMF1`/`IMF 1` style labels or mixed casing.
    """
    if not isinstance(station_df, pd.DataFrame):
        return station_df

    renamed = {}
    for column in station_df.columns:
        column_str = str(column).strip()
        match = _VMD_IMF_NAME_RE.match(column_str)
        if match:
            renamed[column] = f"IMF_{int(match.group(1))}"
            continue

        lower = column_str.lower()
        if lower == "residual":
            renamed[column] = "Residual"
        elif lower == "original":
            renamed[column] = "Original"
        elif lower == "reconstructed":
            renamed[column] = "Reconstructed"

    if renamed:
        station_df = station_df.rename(columns=renamed)

    return station_df


def normalize_vmd_station_dict(station_decomposed_dict):
    if not isinstance(station_decomposed_dict, dict):
        return station_decomposed_dict
    return {
        station: normalize_vmd_station_frame(station_df)
        for station, station_df in station_decomposed_dict.items()
    }


def has_vmd_imf_columns(station_df):
    if not isinstance(station_df, pd.DataFrame):
        return False
    return any(_VMD_IMF_NAME_RE.match(str(column).strip()) for column in station_df.columns)


def model_root_exists(yaml_file_dict):
    if yaml_file_dict.get("train_model", False):
        return True
    target_list = flatten_config_list(yaml_file_dict.get("var_to_predict"))
    target = target_list[0] if target_list else None
    forecast_method = resolve_forecast_method(yaml_file_dict, target)
    if not is_sparse_model_method(forecast_method):
        return True
    return os.path.isdir(model_root_path(yaml_file_dict))


def discover_config_files(config_root):
    config_files = []
    pattern = os.path.join(config_root, "**", "*.yaml")
    for full_path in glob.glob(pattern, recursive=True):
        if os.path.isdir(full_path):
            continue
        normalized_path = full_path.replace("\\", "/")
        if "/models/" in normalized_path or "/add_vars/" in normalized_path or "/Optima/" in normalized_path:
            continue
        base_name = os.path.basename(full_path)
        if base_name.startswith("main_"):
            continue
        rel_path = os.path.relpath(full_path, config_root)
        config_files.append(os.path.splitext(rel_path)[0].replace("\\", "/"))

    return sorted(dict.fromkeys(config_files))


def model_root_path(yaml_file_dict):
    model_base_path = yaml_file_dict.get("model_base_path", DEFAULT_MODEL_BASE_PATH)
    model_version = yaml_file_dict.get("model_version", "v1.0")
    target_list = flatten_config_list(yaml_file_dict.get("var_to_predict"))
    target = target_list[0] if target_list else None
    lags = yaml_file_dict.get("lags", yaml_file_dict.get("model_parameters", {}).get("n_outputs", 24))
    model_name = resolve_model_name(
        yaml_file_dict,
        target=target,
        forecast_method=resolve_forecast_method(yaml_file_dict, target),
        lags=lags,
    )
    return os.path.join(model_base_path, "{name}_{version}".format(
        name=model_name,
        version=model_version,
    ))


def resolve_metrics_csv_path(configuration, iteration=0):
    metrics_dir = configuration.Main_Model_training_evaluation_full_dir
    if not metrics_dir:
        return None

    metrics_filename = configuration.metrics_output_filename_template.format(
        region=configuration.selected_region,
        metrics='average_metric',
        var=configuration.input_var_dir,
        inputs=configuration.n_steps_in,
        outputs=configuration.n_steps_out,
        additional_vars=configuration.additional_var_dir,
        iteration=iteration,
        model=configuration.get_metrics_model_token(),
    )
    metrics_path = os.path.join(metrics_dir, metrics_filename)
    if os.path.exists(metrics_path):
        return metrics_path

    pattern = os.path.join(
        metrics_dir,
        f"{configuration.selected_region}_average_metric_metrics_"
        f"{configuration.input_var_dir}_{configuration.n_steps_in}_{configuration.n_steps_out}_"
        f"{configuration.additional_var_dir}_*_{configuration.get_metrics_model_token()}.csv",
    )
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return matches[0] if matches else None


def append_total_runtime_row(metrics_csv_path, total_run_time_seconds):
    if not metrics_csv_path or not os.path.exists(metrics_csv_path):
        return False

    metrics_pd = pd.read_csv(metrics_csv_path)
    if "Stations" not in metrics_pd.columns:
        return False

    metrics_pd = metrics_pd[
        metrics_pd["Stations"].astype(str) != "TOTAL_RUN_TIME_SECONDS"
    ].copy()

    runtime_row = {col: "" for col in metrics_pd.columns}
    runtime_row["Stations"] = "TOTAL_RUN_TIME_SECONDS"
    if "MAE" in metrics_pd.columns:
        runtime_row["MAE"] = round(float(total_run_time_seconds), 3)
    elif len(metrics_pd.columns) > 0:
        runtime_row[metrics_pd.columns[0]] = "TOTAL_RUN_TIME_SECONDS"

    metrics_pd = pd.concat([metrics_pd, pd.DataFrame([runtime_row])], ignore_index=True)
    metrics_pd.to_csv(metrics_csv_path, index=False)
    return True


def model_station_count_matches(yaml_file_dict, station_count):
    if yaml_file_dict.get("train_model", False):
        return True, None

    expected_station_count = model_output_units(yaml_file_dict)
    if expected_station_count is None:
        return True, None

    return expected_station_count == station_count, expected_station_count


def model_output_units(yaml_file_dict):
    model_path = os.path.join(
        model_root_path(yaml_file_dict),
        "models",
        "IMF_1_model.keras",
    )
    if not os.path.isfile(model_path):
        return None

    with zipfile.ZipFile(model_path) as zf:
        model_config = json.loads(zf.read("config.json"))

    for layer in model_config.get("config", {}).get("layers", []):
        if layer.get("class_name") == "Dense":
            return layer.get("config", {}).get("units")
    return None


def runtime_base_dir_for_run(yaml_file_dict=None):
    base_dir = (
        (yaml_file_dict or {}).get("base_dir")
        or (yaml_file_dict or {}).get("runtime_base_dir")
        or (yaml_file_dict or {}).get("run_base_dir")
    )
    if base_dir:
        return os.path.abspath(base_dir)
    return DEFAULT_RUNTIME_BASE_DIR


def runtime_runs_dir_for(runtime_base_dir):
    return os.path.join(runtime_base_dir, AI_RUNS_DIR_NAME)


def shared_data_dir_for_run(yaml_file_dict):
    return os.path.join(runtime_base_dir_for_run(yaml_file_dict), "API_Input", "model_data", "Shared_data")


def imputed_data_dir_for_run(yaml_file_dict):
    return os.path.join(
        runtime_base_dir_for_run(yaml_file_dict),
        "API_Input",
        "model_data",
        "Imputed_data",
    )


def resolve_imputation_cache_path(imputation_dir, imputation_method, selected_region, target, station):
    """
    Accept both the current method-qualified cache naming and the older legacy
    cache naming that omitted the imputation method token.
    """
    target_token = str(target)
    region_token = str(selected_region)
    station_token = str(station)

    candidate_paths = [
        os.path.join(
            imputation_dir,
            f"Imputed_{imputation_method}_{region_token}_{target_token}_{station_token}.csv",
        ),
        os.path.join(
            imputation_dir,
            f"Imputed_{region_token}_{target_token}_{station_token}.csv",
        ),
    ]

    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path

    glob_pattern = os.path.join(
        imputation_dir,
        f"Imputed*_{region_token}_{target_token}_{station_token}.csv",
    )
    glob_matches = sorted(glob.glob(glob_pattern))
    if glob_matches:
        return glob_matches[0]

    return candidate_paths[0]


def load_available_imputation_cache(imputation_dir, imputation_method, selected_region, target, stations):
    station_imputed_dict = {}
    missing_stations = []

    for station in stations:
        imputation_path = resolve_imputation_cache_path(
            imputation_dir,
            imputation_method,
            selected_region,
            target,
            station,
        )
        if not os.path.isfile(imputation_path):
            missing_stations.append(station)
            continue

        station_df = pd.read_csv(imputation_path, parse_dates=["datetime"], index_col="datetime")
        station_imputed_dict[station] = station_df

    return station_imputed_dict, missing_stations

############################################################################################################
class Sparse_LSTM_Class(object):
    """ 
    This class defines a Input_Class, that contains the capacity to manage the intputs.

    Attributes
    -----------
    logger : logging.logger
        instance of a logger to output messages.
    justif : int
        max message width to justify logger output.   

       
    """
    def __init__(self, logger, justif,
                yaml_config_filename,):

        self.logger = logger
        self.justif = justif

        self.logger.info(''.ljust(self.justif,'-'))
        self.logger.info('Sparse_LSTM Forecast'.center(self.justif,'|'))
        self.logger.info(''.ljust(self.justif,'-'))
        self.config_from_file = None
        
        
        # configuration files
        self.yaml_config_filename = yaml_config_filename
        


        return
###########################################################################################
    def _sync_available_output_stations(self, station_imputed_dict):
        if not station_imputed_dict:
            return self.dpie_output_station_list, self.Configuration.output_column_names

        available_station_list = [
            station for station in self.dpie_output_station_list
            if station in station_imputed_dict
        ]
        if not available_station_list:
            available_station_list = sorted(station_imputed_dict)

        missing_stations = [
            station for station in self.dpie_output_station_list
            if station not in station_imputed_dict
        ]
        if missing_stations:
            self.logger.warning(
                "Missing imputed cache for {n} station(s): {stations}".format(
                    n=len(missing_stations),
                    stations=", ".join(missing_stations),
                ).ljust(self.justif - 2, '.') + 'SKIP'
            )

        if available_station_list != self.dpie_output_station_list:
            self.dpie_output_station_list = available_station_list
            self.Configuration.dpie_output_station_list = available_station_list
            target_token = log_target_token(self.var_to_predict[0])
            output_column_names = sorted(
                f"{target_token}_{station}" for station in available_station_list
            )
            self.Configuration.output_column_names = output_column_names
            return available_station_list, output_column_names

        return self.dpie_output_station_list, self.Configuration.output_column_names
###########################################################################################
    def MakeDir(self, ddir):
        ''' This function makes the different working directories
        '''
        if not os.path.exists(ddir):
            os.makedirs(ddir)
            mod775 = stat.S_IRUSR |stat.S_IWUSR |stat.S_IXUSR |stat.S_IRGRP |stat.S_IWGRP  |stat.S_IXGRP |stat.S_IROTH |stat.S_IXOTH
            os.chmod(ddir,mod775)
            #os.chmod(self.RunDir,0775)
            self.logger.info('Directory = {msg}'.format(msg=ddir).ljust(self.justif-7,'.') + 'CREATED')
        return    

###########################################################################################
    def _visualization_output_dirs(self):
        runtime_base_dir = runtime_base_dir_for_run()
        viz_base_dir = os.path.join(runtime_runs_dir_for(runtime_base_dir), "Visualization")
        return {
            "time_series": os.path.join(viz_base_dir, "Time_series"),
            "time_series_week": os.path.join(viz_base_dir, "Time_series_by_week"),
            "forecast_plot": os.path.join(viz_base_dir, "Forecast_Plot"),
        }

###########################################################################################
    def _build_visualization_frames_from_forecast(self, history_pd, forecast_pd):
        if history_pd is None or forecast_pd is None:
            return None, None

        dashboard_var = self.Configuration.input_var_dir
        station_names = list(getattr(self.Configuration, "dpie_output_station_list", []) or [])

        history_frame = history_pd.copy()
        if "datetime" not in history_frame.columns:
            history_frame = history_frame.reset_index()
        if "datetime" not in history_frame.columns and "timestamp" in history_frame.columns:
            history_frame = history_frame.rename(columns={"timestamp": "datetime"})

        forecast_frame = forecast_pd.copy()
        if "datetime" not in forecast_frame.columns:
            if "timestamp" in forecast_frame.columns:
                forecast_frame = forecast_frame.rename(columns={"timestamp": "datetime"})
            else:
                forecast_frame = forecast_frame.reset_index()
        if "datetime" not in forecast_frame.columns and "timestamp" in forecast_frame.columns:
            forecast_frame = forecast_frame.rename(columns={"timestamp": "datetime"})

        if not station_names:
            station_names = []
            for column in history_frame.columns:
                if column in {"datetime", "forecast_hours"}:
                    continue
                if column.startswith(f"{dashboard_var}_"):
                    station_names.append(column.split("_", 1)[1])

        long_frames = []
        for frame, value_label in ((history_frame, "observed"), (forecast_frame, "predicted")):
            if frame is None or frame.empty:
                continue

            if "datetime" in frame.columns:
                timestamps = pd.to_datetime(frame["datetime"], errors="coerce")
            else:
                timestamps = pd.to_datetime(frame.index, errors="coerce")

            if "forecast_hours" in frame.columns:
                forecast_hours = pd.to_numeric(frame["forecast_hours"], errors="coerce")
            else:
                forecast_hours = pd.Series(range(len(frame)), index=frame.index)

            for station in station_names:
                station_column = f"{dashboard_var}_{station}"
                if station_column not in frame.columns:
                    continue

                block = pd.DataFrame({
                    "timestamp": timestamps.values,
                    "forecast_hours": forecast_hours.values,
                    "station": station,
                    "variable": dashboard_var,
                    "observed": np.nan,
                    "predicted": np.nan,
                })
                block[value_label] = pd.to_numeric(frame[station_column], errors="coerce").values
                long_frames.append(block)

        long_frame = pd.concat(long_frames, ignore_index=True) if long_frames else None
        combined_wide = pd.concat([history_frame, forecast_frame], axis=0, ignore_index=True, sort=False)
        return long_frame, combined_wide

###########################################################################################
    def _build_visualization_frames_from_training(self, training_plot_data):
        if not training_plot_data:
            return None, None

        dashboard_var = self.Configuration.input_var_dir
        station_names = list(
            training_plot_data.get("station_names")
            or getattr(self.Configuration, "dpie_output_station_list", [])
            or []
        )
        timestamps = training_plot_data.get("timestamps")
        predictions = training_plot_data.get("predictions") or {}
        actuals = training_plot_data.get("actuals") or {}

        if not station_names:
            station_names = sorted(set(predictions) | set(actuals))

        if timestamps is not None:
            timestamps = pd.Index(timestamps)

        long_frames = []
        wide_frame = pd.DataFrame()
        if timestamps is not None:
            wide_frame["datetime"] = timestamps
            wide_frame["forecast_hours"] = list(range(1, len(timestamps) + 1))

        for station in station_names:
            predicted = np.asarray(predictions.get(station, []))
            observed = np.asarray(actuals.get(station, []))
            if predicted.size == 0 and observed.size == 0:
                continue
            if predicted.size == 0:
                predicted = np.full_like(observed, np.nan, dtype=float)
            if observed.size == 0:
                observed = np.full_like(predicted, np.nan, dtype=float)

            if timestamps is None:
                timestamps = pd.RangeIndex(start=1, stop=len(predicted) + 1, step=1)
                wide_frame["datetime"] = timestamps
                wide_frame["forecast_hours"] = list(range(1, len(predicted) + 1))

            block = pd.DataFrame({
                "timestamp": pd.Index(timestamps).values,
                "forecast_hours": list(range(1, len(predicted) + 1)),
                "station": station,
                "variable": dashboard_var,
                "observed": observed,
                "predicted": predicted,
            })
            long_frames.append(block)
            wide_frame[f"{dashboard_var}_{station}"] = predicted

        long_frame = pd.concat(long_frames, ignore_index=True) if long_frames else None
        return long_frame, wide_frame

###########################################################################################
    def _export_visualization_plots(self, long_frame=None, wide_frame=None):
        if long_frame is None and wide_frame is None:
            self.logger.warning(
                'Visualization export'.ljust(self.justif - 2, '.') + 'SKIPPED - NO DATA'
            )
            return

        output_dirs = self._visualization_output_dirs()
        region_name = getattr(self.Configuration, "selected_region", "Region")
        generated = []

        if long_frame is not None and not long_frame.empty:
            try:
                ts_plotter = TSPlots.Plot_Class(self.logger, self.justif, self.Configuration)
                generated.extend(ts_plotter.plot_test_phase_timeseries(
                    long_frame,
                    output_dir=output_dirs["time_series"],
                    region_name=region_name,
                ))
                generated.extend(ts_plotter.plot_test_phase_timeseries(
                    long_frame,
                    output_dir=output_dirs["time_series_week"],
                    region_name=region_name,
                    split_mode="week",
                ))
            except Exception as exc:
                self.logger.warning(
                    f"Time-series visualization failed: {exc}".ljust(self.justif - 2, '.') + 'WARN'
                )

        if wide_frame is not None and not wide_frame.empty:
            try:
                forecast_plotter = ForecastPlots.Plot_Class(self.logger, self.justif, self.Configuration)
                generated.extend(forecast_plotter.plot_horizon_windows(
                    wide_frame,
                    output_dir=output_dirs["forecast_plot"],
                ))
            except Exception as exc:
                self.logger.warning(
                    f"Forecast-window visualization failed: {exc}".ljust(self.justif - 2, '.') + 'WARN'
                )

        if generated:
            self.logger.info(
                f'Visualization outputs generated: {len(generated)}'.ljust(self.justif - 2, '.') + 'OK'
            )
        return
###########################################################################################

    def run_all(self,
               shared_input_data_pd=None,
               shared_input_metadata=None,
               shared_imputed_dict=None,
               shared_decomposed_dict=None,
               shared_data_dir=None,
               override_target=None,
               override_region=None,
               preprocess_only=False,
               ):
        """
        shared_input_data_pd   : pre-downloaded+sliced DataFrame — skips API call & slice
        shared_input_metadata  : cached column/species metadata for the shared input
        shared_imputed_dict    : {station: DataFrame} — skips imputation
        shared_decomposed_dict : {station: DataFrame} — skips decomposition
        shared_data_dir        : horizon-agnostic path set once in __main__
        preprocess_only        : run the shared preprocessing stack and stop
        """

        # self.base_output_dir = "Result_Outputs"
        # self.base_run_dir = "Forecasts"
        # self.model_forecast_save_dir = "model_data/Sparse_LSTM"
        # # self.model_forecast_save_dir = "/mnt/scratch_lustre/scratch3/AI_Runs/Model_weights/model_data/Sparse_LSTM"
        self.input_data_dir = os.path.join(DEFAULT_RUNTIME_BASE_DIR, "data")
        
        # Main_training_dir = "/mnt/scratch_lustre/scratch3/AI_Runs/Training"
        # Main_output_dir = "/mnt/scratch_lustre/scratch3/AI_Runs/Forecast"a
        # Main_model_data_dir = "/mnt/scratch_lustre/scratch3/AI_Runs/Model_weights"
        # Code_configuration_main_dir = "/home/barthelemyx/Projects/Deep_learning/Sparse_LSTM_forecast/Core_iHPC/Tools/Config_testing"
        
        #########
        # New place
        runtime_base_dir = runtime_base_dir_for_run()
        runtime_runs_dir = runtime_runs_dir_for(runtime_base_dir)
        Main_training_dir = os.path.join(runtime_runs_dir, "Training")
        Main_output_dir = os.path.join(runtime_runs_dir, "Forecast")
        Main_model_data_dir = os.path.join(runtime_runs_dir, "Model_weights")
        Code_configuration_main_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "Core_iHPC",
            "Tools",
            "Config_testing",
        )

        Second_dashboard_viz_file_output_dir = (
            "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Nowcasting/cnn_lstm_forecast/AI_dashboard_files"
        )
        
        
        # Get the current date
        #current_date = dtime.date.today()
        #current_date = dtime.datetime.now()
        current_date = getattr(self, 'shared_timestamp', dtime.datetime.now())
        #current_date = dtime.datetime(2025,6,10,0)
        
        #### Start date of data for training
        #self.train_start_date_utc = dtime.datetime(2021,12,1,0)
        self.train_start_date_utc = dtime.datetime(2021,11,1,0)

        #### End date of data for training
        self.train_end_date_utc = current_date

        ###### NOTE: We cannot make forecast at anytime (only from 0AM last night) API limitation

        #### Start date of data for forecasting
        #self.start_date_utc = dtime.datetime(current_date.year, current_date.month,current_date.day,0)
        self.start_date_utc = current_date

        #### End date of getting data for forecasting
        self.end_date_utc = self.start_date_utc - dtime.timedelta(days=4) ### Get higher data from batch size !!!
        
        #print(self.end_date_utc)
        #print(current_date)
        #input("Press Enter to continue...") # Pauses and waits for user input
        
        #print(self.start_date_utc)
        #print(self.end_date_utc)
        #input("Press Enter to continue...") # Pauses and waits for user input        
        

        # self.timestamp_aedt = dtime.datetime.strptime('20231215', '%Y%m%d%H')
        self.timestamp_aedt = current_date


        self.Forecast_RunTimeHours = 1
        self.Forecast_NumberofCpu = 1
        self.Forecast_Partition = "toto"

        # self.var_to_predict = self.config.var_to_predict[0]
        # self.full_input_pd = None
    ###########################################################
        self.Configuration = CC.Configuration_Class(self.logger, self.justif,
            self.train_start_date_utc, self.train_end_date_utc, 
            self.start_date_utc, self.end_date_utc, self.timestamp_aedt, 
            self.input_data_dir,
            Main_training_dir, Main_output_dir, Main_model_data_dir, Code_configuration_main_dir, Second_dashboard_viz_file_output_dir,
            self.Forecast_RunTimeHours, self.Forecast_NumberofCpu, self.Forecast_Partition, 
            )
    ###########################################################
    # read the YAML configuration file        
        YFRC = YFR.Yaml_file_reader_Class(self.Configuration)
        yaml_file_dict = YFRC.read_yaml_config_file(self.yaml_config_filename)
        explicit_model_requested = any(
            flatten_config_list([
                yaml_file_dict.get("forecast_method"),
                yaml_file_dict.get("model_name"),
                (yaml_file_dict.get("model_parameters") or {}).get("forecast_method"),
            ])
        )
        yaml_file_dict = apply_runtime_overrides(
            yaml_file_dict,
            target=override_target,
            region=override_region,
        )
        yaml_file_dict["var_to_predict"] = flatten_config_list(yaml_file_dict.get("var_to_predict"))
        yaml_file_dict["additional_var_to_select"] = flatten_config_list(yaml_file_dict.get("additional_var_to_select"))
        runtime_base_dir = runtime_base_dir_for_run(yaml_file_dict)
        self.input_data_dir = os.path.join(runtime_base_dir, "data")
        self.Configuration.input_data_dir = self.input_data_dir
        runtime_runs_dir = runtime_runs_dir_for(runtime_base_dir)
        self.Configuration.Main_training_dir = os.path.join(runtime_runs_dir, "Training")
        self.Configuration.Main_output_dir = os.path.join(runtime_runs_dir, "Forecast")
        self.Configuration.Main_model_data_dir = os.path.join(runtime_runs_dir, "Model_weights")
        self.Configuration.Second_dashboard_viz_file_output_dir = os.path.join(
            "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Nowcasting/cnn_lstm_forecast",
            "AI_dashboard_files",
        )
        if not flatten_config_list(yaml_file_dict.get("var_to_predict")):
            inferred_targets = infer_targets_from_config_name(
                self.yaml_config_filename,
                yaml_file_dict,
            )
            if inferred_targets:
                yaml_file_dict["var_to_predict"] = inferred_targets
            else:
                raise ValueError(
                    f"Unable to determine var_to_predict for {self.yaml_config_filename}. "
                    "Add var_to_predict explicitly or rename the config file to include the target."
                )
        self.use_decomposition = MASTER_USE_DECOMPOSE
        self.use_vmd_decomposition = MASTER_USE_DECOMPOSE
        self.use_one_model_per_imf = MASTER_USE_ONE_MODEL_PER_IMF
        self.Configuration.set_vmd_pipeline_mode(
            self.use_decomposition,
            self.use_one_model_per_imf,
        )
        yaml_file_dict.setdefault("selected_region", DEFAULT_SELECTED_REGION)

        if len(flatten_config_list(yaml_file_dict["var_to_predict"])) != 1:
            raise ValueError(
                "main.py expands multi-target var_to_predict values before run_all(). "
                "Each Sparse_LSTM_Class.run_all() call must receive one target."
            )
        
        # Pass model management parameters into Configuration
        self.Configuration.model_base_path = yaml_file_dict.get("model_base_path", DEFAULT_MODEL_BASE_PATH)
        self.Configuration.model_name = None
        self.Configuration.model_version = yaml_file_dict.get("model_version", "v1.0")
        self.Configuration.time_steps = yaml_file_dict.get("time_steps", 12)
        self.Configuration.lags = yaml_file_dict.get("lags", 24)
        self.Configuration.strict_version_check = yaml_file_dict.get("strict_version_check", False)
        
        #####
        #setup the runs
        initial_region_map = DPERT.DPE_region_stations(yaml_file_dict["var_to_predict"])
        selected_regions_for_run = resolve_regions_for_target(
            yaml_file_dict["selected_region"],
            initial_region_map,
        )
        if len(selected_regions_for_run) != 1:
            raise ValueError(
                "run_all() now executes exactly one resolved region per call. "
                "__main__ must expand multi-region jobs before invoking it."
            )

        region = selected_regions_for_run[0]
        n_inputs = yaml_file_dict["model_parameters"]["n_inputs"]
        n_outputs = yaml_file_dict["model_parameters"]["n_outputs"]

        def _run_single_pass():

            #### Selected region
            self.selected_region = region

            self.n_steps_in = n_inputs
            self.n_steps_out = n_outputs         
            
            self.var_to_predict = yaml_file_dict.get("var_to_predict", list(DEFAULT_VAR_TO_PREDICT))
            self.full_input_pd = None

            self.batch_size = yaml_file_dict["run_parameters"]["batchsize"]
            self.Configuration.num_batches  = yaml_file_dict["run_parameters"]["num_batches"]
            
            self.Configuration.yaml_config_filename = self.yaml_config_filename
            
            ### Resolve the model module strictly from the target, not from
            ### stale YAML strings. This keeps PM2.5 on Sparse_LSTM_v2.
            self.forecast_method = resolve_forecast_method(yaml_file_dict, self.var_to_predict[0])

            self.n_epochs = yaml_file_dict["run_parameters"]["n_epoch"]
            self.additional_var_to_select = list(dict.fromkeys(flatten_config_list(yaml_file_dict.get("additional_var_to_select", []))))
                   
            self.Configuration.prob_samples = yaml_file_dict["prob_samples"]
            
            #### First configuration set must download new data
            self.Configuration.use_file_training = False
            
            self.DPE_region = DPERT.DPE_region_stations(self.var_to_predict)
            self.DPE_region_stations_dict = get_region_stations_dict(self.DPE_region)
            self.selected_region = resolve_region_name(self.DPE_region, region)
            yaml_file_dict["selected_region"] = self.selected_region
            yaml_file_dict.setdefault("train_model", DEFAULT_TRAIN_MODEL)
            self.Configuration.model_name = resolve_model_name(
                yaml_file_dict,
                target=self.var_to_predict[0],
                forecast_method=self.forecast_method,
                lags=yaml_file_dict["model_parameters"].get("n_outputs", 24),
            )
            assert_target_model_consistency(
                self.var_to_predict[0],
                self.forecast_method,
                self.Configuration.model_name,
                enforce=not explicit_model_requested,
            )

            # ── Resolve shared_data_dir ───────────────────────────────────────
            # All forecast horizons (hr_3/6/9/12) for the same region/var/model
            # share ONE directory — n_outputs is excluded from the path so every
            # horizon reads the same pre-computed imputed/VMD files.
            # __main__ computes this once and passes it in via shared_data_dir;
            # single-config runs fall back to computing it locally.
            run_shared_data_dir = shared_data_dir
            if run_shared_data_dir is None:
                run_shared_data_dir = shared_data_dir_for_run(yaml_file_dict)
            self.Configuration.configure_shared_data_dir(run_shared_data_dir)
            run_imputed_data_dir = imputed_data_dir_for_run(yaml_file_dict)
            self.Configuration.configure_imputed_data_dir(run_imputed_data_dir)
            self.logger.info(
                f'Shared data dir: {run_shared_data_dir}'.ljust(self.justif - 2, '.') + 'SET')
            self.logger.info(
                f'Imputed data dir: {run_imputed_data_dir}'.ljust(self.justif - 2, '.') + 'SET')
            # ─────────────────────────────────────────────────────────────────

            ### Input list of stations
            self.dpie_input_station_list = self.DPE_region_stations_dict[self.selected_region]
            self.lcs_input_station_list = []
            self.custom_input_station_list = []

            ### Output list of stations
            self.dpie_output_station_list = self.DPE_region_stations_dict[self.selected_region]
            self.lcs_output_station_list = []
            self.custom_output_station_list = []

            #### Defining the training model -> True, of skipping training and making the forecast with pretrained model --> False
            self.train_model = yaml_file_dict.get("train_model", DEFAULT_TRAIN_MODEL)

            #### parameters of models
            self.model_parameters_dict = yaml_file_dict["model_parameters"]

            #### additional internal vars to add, such as hour, month, etc...
            # ["hour", "month", "season_meteorological", "season_astronomical","day_of_year","day_of_week"]
            # Safely read from YAML, default to empty list when key is missing
            self.internal_var_to_add_list = yaml_file_dict.get("internal_var_to_add", [])
            if self.internal_var_to_add_list is None:
                self.internal_var_to_add_list = []
            elif isinstance(self.internal_var_to_add_list, str):
                self.internal_var_to_add_list = [self.internal_var_to_add_list]
            
            # save whole model weigths
            self.save_model = True


            print("*"*50)
            print("Selected region: ", self.selected_region)
            print("number Inputs: ", self.n_steps_in)
            print("number Outputs: ", self.n_steps_out)
            print("*"*50)
 
            
            self.Configuration.configure_model(self.forecast_method,
                                self.var_to_predict, self.additional_var_to_select,
                                self.DPE_region_stations_dict, self.selected_region,  
                                self.dpie_input_station_list, self.lcs_input_station_list, self.custom_input_station_list,
                                self.dpie_output_station_list, self.lcs_output_station_list, self.custom_output_station_list,
                                self.n_steps_in, self.n_steps_out, self.n_epochs, self.batch_size,
                                self.train_model, self.save_model, #self.selected_region, self.full_input_pd
                                self.model_parameters_dict, self.internal_var_to_add_list,
                                )

            ###############
            ### Using data collected from API
            self.Data_input_flow = ["DPE_station_api"]
            self.use_file = yaml_file_dict["use_file"]
            # self.use_file = False
            # self.use_file = True
            
            self.save_obs_processed_data = False
            self.save_obs_processed_data = True

            effective_use_file = self.use_file
            if shared_input_data_pd is not None:
                # Later horizons only need metadata from disk; never redownload.
                effective_use_file = True
            elif not self.Configuration.train_model:
                # First forecast horizon must always run the cache-aware refresh
                # path so the regional processed-input cache is updated with
                # any missing tail before imputation / VMD.
                effective_use_file = False
                self.logger.info(
                    'Input refresh policy'.ljust(self.justif - 2, '.') +
                    ('FORCED CACHE CHECK' if self.use_file else 'CACHE CHECK')
                )

            self.Configuration.configure_input(self.Data_input_flow,  
                                               effective_use_file, 
                                               self.save_obs_processed_data)
            
            ##### output configuration
            # self.result_file_subdir = "Result_files"
            # self.plot_subdir = "Plots"


            # self.Configuration.configure_output(self.result_file_subdir, self.plot_subdir)
            self.Configuration.configure_output()
            self.Configuration.ensure_all_output_directories()

            self.plot_results = resolve_master_toggle(
                yaml_file_dict.get("plot", yaml_file_dict.get("plots")),
                default=DEFAULT_PLOT,
            )
            if not self.Configuration.train_model:
                self.plot_results = False

            ##### imputation configuration
            yaml_imputation_method = (
                yaml_file_dict.get("IMPUTATION_METHOD")
                or yaml_file_dict.get("imputation_method")
            )
            if isinstance(yaml_imputation_method, str) and yaml_imputation_method.strip().upper() in {
                "MAIN",
                "DEFAULT",
                "USE_MAIN",
                "FROM_MAIN",
            }:
                yaml_imputation_method = None
            self.imputation_method = (
                yaml_imputation_method
                or IMPUTATION_METHOD
                or DEFAULT_IMPUTATION_METHOD
            )
            if self.imputation_method not in AVAILABLE_IMPUTATION_METHODS:
                raise ValueError(
                    "Unsupported imputation_method={method}. Choose one of: {choices}".format(
                        method=self.imputation_method,
                        choices=", ".join(AVAILABLE_IMPUTATION_METHODS),
                    )
                )

            self.save_imputed_data = yaml_file_dict.get("save_imputed_data", True)
            
            self.Configuration.configure_imputation(self.imputation_method, self.save_imputed_data)
            self.Configuration.configure_evaluation_metrics(yaml_file_dict["evaluation_metrics"])


            ##### End of configuration ########################################################
            self.IMC = IM.Input_manager_Class(self.Configuration)

            # ── Reuse shared download if available (2nd+ horizon in multi-run) ─
            if shared_input_data_pd is not None:
                self.logger.info('Input data'.ljust(self.justif - 2, '.') + 'REUSED FROM FIRST RUN')
                if shared_input_metadata is not None:
                    # Normalize shared_input_metadata if it's a legacy tuple/list
                    metadata = shared_input_metadata
                    if not isinstance(metadata, dict) and metadata is not None:
                        try:
                            if isinstance(metadata, (list, tuple)) and len(metadata) >= 4:
                                pd_val = metadata[0]
                                input_cols = metadata[1]
                                output_cols = metadata[2]
                                specie_props = metadata[3]
                                metadata = {
                                    "input_data": [pd_val] if not isinstance(pd_val, list) else pd_val,
                                    "input_column_names": input_cols,
                                    "output_column_names": output_cols,
                                    "specie_properties_dict": specie_props,
                                }
                        except Exception:
                            pass
                    input_data_dict = {
                        self.Data_input_flow[0]: metadata,
                    }
                else:
                    self.logger.warning(
                        'Shared input metadata missing; reloading input metadata from cache'.ljust(
                            self.justif - 2, '.'
                        ) + 'FALLBACK'
                    )
                    input_data_dict = self.IMC.load_inputs()
                input_data_pd_list = [_copy_dataframe(shared_input_data_pd)]
            else:
                input_data_dict = self.IMC.load_inputs()
                input_data_pd_list = input_data_dict[self.Data_input_flow[0]]["input_data"]
            # ──────────────────────────────────────────────────────────────────

            metadata = input_data_dict.get(self.Data_input_flow[0], {})
            if not isinstance(metadata, dict):
                metadata = {}

            input_frame = input_data_pd_list[0]
            if "input_column_names" not in metadata:
                metadata["input_column_names"] = list(getattr(input_frame, "columns", []))
            if "output_column_names" not in metadata:
                self.DPE_region = DPERT.DPE_region_stations(self.var_to_predict)
                self.DPE_region_stations_dict = get_region_stations_dict(self.DPE_region)
                metadata["output_column_names"] = sorted(
                    "_".join([self.var_to_predict[0], station])
                    for station in self.DPE_region_stations_dict[self.selected_region]
                )
            if "specie_properties_dict" not in metadata:
                metadata["specie_properties_dict"] = {}
            if "input_data" not in metadata:
                metadata["input_data"] = [input_frame]

            input_data_dict[self.Data_input_flow[0]] = metadata
            input_column_names = metadata["input_column_names"]
            output_column_names = metadata["output_column_names"]
            specie_properties_dict = metadata["specie_properties_dict"]
            if getattr(self.Configuration, "var_data_list_from_input_pd", None) in (None, []):
                target_token = log_target_token(self.var_to_predict[0])
                target_api_name = {
                    "O3": "OZONE",
                    "PM2.5": "PM2.5",
                    "PM10": "PM10",
                }.get(target_token, target_token)
                self.Configuration.var_data_list_from_input_pd = [target_api_name]

            def _input_frame_has_target_columns(frame, target):
                if frame is None or not hasattr(frame, "columns"):
                    return False
                normalized_target = "".join(ch for ch in str(target).upper() if ch.isalnum())
                alias_groups = {
                    "O3": ("O3", "OZONE"),
                    "OZONE": ("O3", "OZONE"),
                    "PM25": ("PM25", "PM2.5", "PM_25"),
                    "PM2.5": ("PM25", "PM2.5", "PM_25"),
                    "PM_25": ("PM25", "PM2.5", "PM_25"),
                    "PM10": ("PM10",),
                }
                target_aliases = alias_groups.get(normalized_target, (target,))
                return any(
                    str(col).startswith(f"{alias}_")
                    for alias in target_aliases
                    for col in frame.columns
                )

            if not _input_frame_has_target_columns(input_data_pd_list[0], self.var_to_predict[0]):
                self.logger.warning(
                    "Input cache mismatch for {target}; reloading target-specific input frame".format(
                        target=self.var_to_predict[0]
                    ).ljust(self.justif - 2, '.') + 'RELOAD'
                )
                processed_cache_dir = (
                    self.Configuration.Main_output_run_shared_dir
                    or self.Configuration.Main_output_run_full_dir
                )
                processed_cache_name = self.IMC.OBS_save_file_template.format(
                    raw_processed="processed",
                    input_stream=self.Data_input_flow[0],
                    region=self.selected_region,
                    var=self.Configuration.input_var_dir,
                    additional_vars=self.Configuration.additional_var_dir,
                )
                processed_cache_path = os.path.join(processed_cache_dir, processed_cache_name)
                if os.path.isfile(processed_cache_path):
                    cached_input_pd = pd.read_csv(processed_cache_path, index_col=0, parse_dates=True)
                    if _input_frame_has_target_columns(cached_input_pd, self.var_to_predict[0]):
                        input_data_pd_list = [cached_input_pd]
                        input_column_names = list(cached_input_pd.columns)
                    else:
                        self.logger.warning(
                            f"Target-specific processed cache still mismatched at {processed_cache_path}".ljust(
                                self.justif - 2, '.'
                            ) + 'SKIP'
                        )
                        input_data_dict = self.IMC.load_inputs()
                        input_data_pd_list = input_data_dict[self.Data_input_flow[0]]["input_data"]
                        input_column_names = input_data_dict[self.Data_input_flow[0]]["input_column_names"]
                        output_column_names = input_data_dict[self.Data_input_flow[0]]["output_column_names"]
                        specie_properties_dict = input_data_dict[self.Data_input_flow[0]]["specie_properties_dict"]
                else:
                    input_data_dict = self.IMC.load_inputs()
                    input_data_pd_list = input_data_dict[self.Data_input_flow[0]]["input_data"]
                    input_column_names = input_data_dict[self.Data_input_flow[0]]["input_column_names"]
                    output_column_names = input_data_dict[self.Data_input_flow[0]]["output_column_names"]
                    specie_properties_dict = input_data_dict[self.Data_input_flow[0]]["specie_properties_dict"]

            ### update Configure
            self.Configuration.input_column_names = input_column_names
            self.Configuration.output_column_names = output_column_names
            self.Configuration.specie_properties_dict = specie_properties_dict

            
            ########################################################
            ### Forecast mode: slice input_data to last N days only
            ### Skipped when shared_input_data_pd provided (already sliced).
            ########################################################
            forecast_window_days = yaml_file_dict.get("forecast_window_days", 4)
            if not self.Configuration.train_model and shared_input_data_pd is None:
                cutoff = input_data_pd_list[0].index.max() - pd.Timedelta(days=forecast_window_days)
                input_data_pd_list[0] = input_data_pd_list[0][input_data_pd_list[0].index >= cutoff]
                self.logger.info(
                    f'Forecast window: last {forecast_window_days} days ({len(input_data_pd_list[0])} rows)'.ljust(
                        self.justif - 2, '.') + 'OK')

            ########################################################
            ### Imputation
            ########################################################
            imputer_class_map = {
                "NONE": Impute.NoneImputation,
                "MICE": Impute.MICEImputation,
                "KNN": Impute.KNNImputation,
                "TemporalMICE": Impute.TemporalMICEImputation,
                "AQUISTIL": Impute.AQUISTILImputation,
            }
            imputer_cls = imputer_class_map.get(self.imputation_method)
            if imputer_cls is None:
                raise ValueError(
                    "Unsupported imputation method {method}".format(method=self.imputation_method)
                )
            self.Imputation = imputer_cls(self.Configuration)
            self.impute = yaml_file_dict.get("impute", True)

            # ── Resolve data directories ─────────────────────────────────────
            if self.Configuration.train_model:
                data_dir = self.Configuration.Main_Model_training_full_dir
            else:
                data_dir = (self.Configuration.Main_output_run_shared_dir
                            or self.Configuration.Main_Model_training_full_dir)
            imputation_dir = (self.Configuration.Main_imputed_data_shared_dir
                              or imputed_data_dir_for_run(yaml_file_dict))
            # ─────────────────────────────────────────────────────────────────

            if self.impute:
                # Training       : cache check — skip if files exist.
                # Forecast 1st   : always run on fresh N-day slice.
                # Forecast 2nd+  : shared_imputed_dict provided — skip entirely.
                if shared_imputed_dict is not None:
                    self.logger.info('Imputation'.ljust(self.justif - 2, '.') + 'REUSED FROM FIRST RUN')
                    station_imputed_dict = _copy_dataframe_dict(shared_imputed_dict)
                    imputed_data_pd = pd.concat(list(station_imputed_dict.values()), axis=1)
                else:
                    run_imputation = True
                    if self.Configuration.train_model:
                        imputation_files_exist = all(
                            os.path.exists(
                                resolve_imputation_cache_path(
                                    imputation_dir,
                                    self.imputation_method,
                                    self.selected_region,
                                    self.var_to_predict[0],
                                    station,
                                )
                            )
                            for station in self.dpie_output_station_list
                        )
                        if imputation_files_exist:
                            self.logger.info('Imputed files found for all stations'.ljust(self.justif - 2, '.') + 'LOADING')
                            station_imputed_dict, missing_stations = load_available_imputation_cache(
                                imputation_dir,
                                self.imputation_method,
                                self.selected_region,
                                self.var_to_predict[0],
                                self.dpie_output_station_list,
                            )
                            if missing_stations:
                                self.logger.warning(
                                    f"Cache load missed station(s): {', '.join(missing_stations)}; recomputing imputation".ljust(
                                        self.justif - 2, '.'
                                    ) + 'RELOAD'
                                )
                                station_imputed_dict = {}
                                run_imputation = True
                            else:
                                imputed_frames = list(station_imputed_dict.values())
                                imputed_data_pd = pd.concat(imputed_frames, axis=1)
                                self.logger.info('Imputed data'.ljust(self.justif - 2, '.') + 'LOADED FROM FILE')
                                run_imputation = False

                    if run_imputation:
                        self.logger.info('Imputation'.ljust(self.justif - 2, '.') + 'RUNNING')
                        # Pass save_data=False — imputation classes save to the
                        # timestamped Main_output_run_full_dir which changes each
                        # run.  We save the files ourselves to the stable
                        # Imputed_data/ cache so they can be found on the next
                        # run.
                        imputed_data_pd, station_imputed_dict = self.Imputation.impute(
                            _copy_dataframe(input_data_pd_list[0]),
                            save_data=False,
                        )
                        self.logger.info('Imputation'.ljust(self.justif - 2, '.') + 'DONE')
                        # ── Save per-station imputed files to stable data_dir ──
                        if self.Configuration.save_imputed_data:
                            os.makedirs(imputation_dir, exist_ok=True)
                            for station, station_df in station_imputed_dict.items():
                                save_path = os.path.join(imputation_dir,
                                    f"Imputed_{self.imputation_method}_{self.selected_region}_{self.var_to_predict[0]}_{station}.csv")
                                station_df.to_csv(save_path)
                                self.logger.info(
                                    f'Imputed data = Imputed_{self.imputation_method}_{self.selected_region}_{self.var_to_predict[0]}_{station}.csv'.ljust(
                                        self.justif - 2, '.') + 'WRITTEN')
                        # ──────────────────────────────────────────────────────
            else:
                # impute: false — load saved files directly
                self.logger.info('Imputation'.ljust(self.justif - 2, '.') + 'SKIPPED - LOADING FILES DIRECTLY')
                station_imputed_dict, missing_stations = load_available_imputation_cache(
                    imputation_dir,
                    self.imputation_method,
                    self.selected_region,
                    self.var_to_predict[0],
                    self.dpie_output_station_list,
                )
                if not station_imputed_dict:
                    missing_hint = ", ".join(missing_stations) if missing_stations else "no cache files matched"
                    raise FileNotFoundError(
                        "No imputed cache files found for region {region}, target {target}. Missing station(s): {stations}".format(
                            region=self.selected_region,
                            target=self.var_to_predict[0],
                            stations=missing_hint,
                        )
                    )
                imputed_frames = list(station_imputed_dict.values())
                imputed_data_pd = pd.concat(imputed_frames, axis=1)

            self.dpie_output_station_list, output_column_names = self._sync_available_output_stations(
                station_imputed_dict
            )
            self.Configuration.dpie_output_station_list = self.dpie_output_station_list
            self.Configuration.output_column_names = output_column_names

            ########################################################
            ### IMF Decomposition — skip if all station files exist
            ########################################################
            decomp_dir = os.path.join(data_dir, "decomposition")

            if self.use_decomposition:
                self.logger.info(''.ljust(self.justif, '-'))
                self.logger.info('IMF Decomposition'.center(self.justif, '|'))
                self.logger.info(''.ljust(self.justif, '-'))

                # Read VMD parameters regardless of decompose flag
                self.vmd_n_imfs    = yaml_file_dict.get("vmd_n_imfs",  20)
                self.vmd_alpha     = yaml_file_dict.get("vmd_alpha",  2000)
                self.vmd_tau       = yaml_file_dict.get("vmd_tau",     0.0)
                self.vmd_DC        = yaml_file_dict.get("vmd_DC",        0)
                self.vmd_init      = yaml_file_dict.get("vmd_init",      1)
                self.vmd_tol       = yaml_file_dict.get("vmd_tol",    1e-7)
                self.save_vmd_data = yaml_file_dict.get("save_vmd_data", True)

                self.Configuration.configure_vmd(
                    n_imfs     = self.vmd_n_imfs,
                    alpha      = self.vmd_alpha,
                    tau        = self.vmd_tau,
                    DC         = self.vmd_DC,
                    init       = self.vmd_init,
                    tol        = self.vmd_tol,
                    output_dir = yaml_file_dict.get("vmd_output_dir", "./outputs/vmd"),
                )

                # Resolve API parameter code for VMD filenames.
                # Input_manager maps var_to_predict codes to API names
                # (e.g. O3 → OZONE, PM2.5 → PM2.5) via dpe_var_selection_dict.
                # var_data_list_from_input_pd[0] holds the resolved API name.
                vmd_var_name = self.Configuration.var_data_list_from_input_pd[0]

                # Training       : cache check — skip if files exist.
                # Forecast 1st   : always run on fresh N-day imputed slice.
                # Forecast 2nd+  : shared_decomposed_dict provided — skip entirely.
                if shared_decomposed_dict is not None:
                    self.logger.info('IMF decomposition'.ljust(self.justif - 2, '.') + 'REUSED FROM FIRST RUN')
                    data_for_model = normalize_vmd_station_dict(_copy_dataframe_dict(shared_decomposed_dict))
                else:
                    run_vmd = True
                    if self.Configuration.train_model:
                        vmd_files_exist = all(
                            os.path.exists(os.path.join(decomp_dir,
                                f"IMF_{vmd_var_name}_{self.selected_region}_{station}.csv"))
                            for station in self.dpie_output_station_list
                        )
                        if vmd_files_exist:
                            self.logger.info('IMF files found for all stations'.ljust(self.justif - 2, '.') + 'LOADING')
                            station_decomposed_dict = {}
                            loaded_vmd_valid = True
                            for station in self.dpie_output_station_list:
                                vmd_path = os.path.join(decomp_dir,
                                    f"IMF_{vmd_var_name}_{self.selected_region}_{station}.csv")
                                station_df = pd.read_csv(
                                    vmd_path, parse_dates=['datetime'], index_col='datetime')
                                station_df = normalize_vmd_station_frame(station_df)
                                if not has_vmd_imf_columns(station_df):
                                    loaded_vmd_valid = False
                                    self.logger.warning(
                                        f"Cached IMF file {os.path.basename(vmd_path)} has no IMF columns; "
                                        "recomputing decomposition instead.".ljust(self.justif - 2, '.') + 'WARN'
                                    )
                                station_decomposed_dict[station] = station_df
                            if loaded_vmd_valid:
                                self.logger.info('IMF decomposition'.ljust(self.justif - 2, '.') + 'LOADED FROM FILE')
                                data_for_model = normalize_vmd_station_dict(station_decomposed_dict)
                                run_vmd = False
                            else:
                                station_decomposed_dict = {}
                                run_vmd = True

                    if run_vmd:
                        self.logger.info('IMF decomposition'.ljust(self.justif - 2, '.') + 'RUNNING')
                        if hasattr(DECOMPOSITION_BACKEND, "VMD_Class"):
                            backend_class = DECOMPOSITION_BACKEND.VMD_Class
                        elif hasattr(DECOMPOSITION_BACKEND, "EMD_Class"):
                            backend_class = DECOMPOSITION_BACKEND.EMD_Class
                        elif hasattr(DECOMPOSITION_BACKEND, "Decomposition_Class"):
                            backend_class = DECOMPOSITION_BACKEND.Decomposition_Class
                        else:
                            raise AttributeError(
                                f"Decomposition backend {resolve_decomposition_backend_name()} "
                                "does not expose a runnable class"
                            )
                        self.VMDC = backend_class(self.Configuration)
                        target_variable = self.Configuration.var_data_list_from_input_pd[0]
                        self.logger.info(
                            f'Target column: {self.var_to_predict[0]} -> {target_variable}'.ljust(
                                self.justif - 2, '.') + 'OK')
                        self.logger.info(f"Applying decomposition to {target_variable}: n_imfs={self.vmd_n_imfs}, alpha={self.vmd_alpha}")
                        # Pass save_data=False — the decomposition class saves to its own
                        # output_dir (timestamped or vmd_output_dir from YAML).
                        # We save ourselves to the stable decomposition dir under data_dir.
                        station_decomposed_dict = self.VMDC.decompose(
                            imputed_data_pd,
                            station_imputed_dict,
                            target_column = target_variable,
                            save_data     = False,
                        )
                        if len(station_decomposed_dict) == 0:
                            self.logger.warning("Decomposition failed for all stations".ljust(self.justif - 2, '.') + 'FALLBACK')
                            data_for_model = station_imputed_dict
                        else:
                            self.logger.info('IMF decomposition'.ljust(self.justif - 2, '.') + 'OK')
                            data_for_model = normalize_vmd_station_dict(station_decomposed_dict)
                            # ── Save per-station decomposition files to stable decomposition dir ──
                            if self.save_vmd_data:
                                os.makedirs(decomp_dir, exist_ok=True)
                                for station, station_df in data_for_model.items():
                                    vmd_save_path = os.path.join(decomp_dir,
                                        f"IMF_{vmd_var_name}_{self.selected_region}_{station}.csv")
                                    station_df.to_csv(vmd_save_path)
                                    self.logger.info(
                                        f'IMF data = IMF_{vmd_var_name}_{self.selected_region}_{station}.csv'.ljust(
                                            self.justif - 2, '.') + 'WRITTEN')
                                # ──────────────────────────────────────────────────
                if not self.use_one_model_per_imf and data_for_model is not None:
                    self.logger.info(''.ljust(self.justif, '-'))
                    self.logger.info('Joint IMF preprocessing'.center(self.justif, '|'))
                    self.logger.info(''.ljust(self.justif, '-'))
                    data_for_model = VJPrep.collapse_joint_components(
                        data_for_model,
                        logger=self.logger,
                        justif=self.justif,
                    )
                    self.Configuration.vmd_n_imfs = 0
                    self.vmd_n_imfs = 0
                    self.logger.info(
                        'All IMF components collapsed to a single residual series'.ljust(
                            self.justif - 2, '.') + 'OK')
                elif self.use_one_model_per_imf and data_for_model is not None:
                    missing_imf_stations = [
                        station for station, station_df in data_for_model.items()
                        if not has_vmd_imf_columns(station_df)
                    ]
                    if missing_imf_stations:
                        self.logger.warning(
                            "Per-IMF VMD requested, but these stations have no IMF_* "
                            "columns and will fall back to their available raw/original "
                            "series: {stations}".format(
                                stations=", ".join(missing_imf_stations),
                            ).ljust(self.justif - 2, '.') + 'WARN'
                        )
            else:
                if PIPELINE_MODE_RESOLVED in {"imf_joint", "imf_per_imf", "vmd_joint", "vmd_per_imf"}:
                    raise RuntimeError(
                        "PIPELINE_MODE={mode} requires decomposition, but the execution path reached "
                        "the non-VMD branch for target={target}, region={region}.".format(
                            mode=PIPELINE_MODE_RESOLVED,
                            target=self.var_to_predict[0],
                            region=self.selected_region,
                        )
                    )
                self.logger.info('IMF decomposition'.ljust(self.justif - 2, '.') + 'SKIPPED - USING ORIGINAL INPUTS')
                data_for_model = station_imputed_dict

            if preprocess_only:
                # The shared preprocessing bundle is horizon-agnostic.  Stop here
                # so __main__ can reuse the same imputed/VMD data for hr_3/6/9/12.
                self._last_input_data_pd = _copy_dataframe(input_data_pd_list[0])
                self._last_input_metadata = {
                    self.Data_input_flow[0]: {
                        "input_data": [_copy_dataframe(input_data_pd_list[0])],
                        "input_column_names": list(input_column_names),
                        "output_column_names": list(output_column_names),
                        "specie_properties_dict": copy.deepcopy(specie_properties_dict),
                    }
                }
                self._last_imputed_dict = _copy_dataframe_dict(station_imputed_dict)
                self._last_decomposed_dict = _copy_dataframe_dict(data_for_model)
                return
            
            ########################################################
            # Run the model
            ########################################################
            
            # with tf.device(tf.DeviceSpec(device_type="GPU", device_index= 0)):
            with tf.device(tf.DeviceSpec(device_type="GPU")):
                ########################################################
                ## Define number of rolling iterations for training, it will divide the validation set ny n_iteration to chunks of data 
                ## Each iteration, the new chunk will be added to training data and retrain again to evaluate the next chunk until the last one  
                # n_retrain = yaml_file_dict["n_iteration"]

                # for iterative in range(n_retrain):
                #     #### The first training (iterative = 0) use_file maybe False or True, from the 2nd training time, it uses the downloaded data
                #     if iterative >= 1:
                #         self.use_file = True
                iterative = 0
                ########################################################
                # Instance of the forecast model
                ########################################################
                SLSTM_Module = load_model_module_for_target(
                    self.var_to_predict[0],
                    self.forecast_method,
                    allow_fallback=not explicit_model_requested,
                )
                self.SLSTM = MODEL_LOADER.instantiate_model(SLSTM_Module, self.Configuration)
                list_output_pd, train_hist = self.SLSTM.run_all(data_for_model, iterative)
                self.OMC = OM.Output_manager_Class(self.Configuration,)

                # Save forecast output — only produced in inference mode.
                if list_output_pd is not None:
                    self.OMC.output_forecast_forward(list_output_pd, iterative)

                    if self.plot_results and not self.Configuration.train_model:
                        history_pd = self.OMC._combined_history_for_dashboard(station_imputed_dict)
                        forecast_pd = self.OMC._wide_forecast_for_dashboard(list_output_pd)
                        if history_pd is not None and forecast_pd is not None:
                            self._export_visualization_plots(history_pd, forecast_pd)
                        else:
                            self.logger.warning(
                                'Forecast plotting'.ljust(self.justif - 2, '.') + 'SKIPPED - NO COMBINED DATA'
                            )
                
                
                ########################################################
                # Training mode: per-station metrics written by
                # Sparse_LSTM_v1._generate_metrics_report().
                ########################################################
                if self.Configuration.train_model:
                    dashboard_metrics_pd = getattr(self.SLSTM, "latest_dashboard_metrics_pd", None)
                    if dashboard_metrics_pd is not None:
                        self.OMC.output_dashboard_metrics_file(dashboard_metrics_pd)
                        self.logger.info("Dashboard metrics".ljust(self.justif - 2, '.') + 'OK')
                    else:
                        self.logger.warning(
                            "Dashboard metrics".ljust(self.justif - 2, '.') + 'SKIPPED - NO METRICS DATA'
                        )
                    self.logger.info("Training metrics".ljust(self.justif - 8, '.') + 'SEE STATION REPORTS')

                    # After diagnostics, run one inference pass so the same
                    # trained model also emits the forecast-style dashboard.
                    forecast_configuration = copy.copy(self.Configuration)
                    forecast_configuration.train_model = False
                    forecast_configuration.save_model = False
                    forecast_configuration.plot_results = False
                    forecast_omc = OM.Output_manager_Class(forecast_configuration)
                    forecast_model = MODEL_LOADER.instantiate_model(SLSTM_Module, forecast_configuration)
                    forecast_list_output_pd, _ = forecast_model.run_all(data_for_model, iterative)
                    if forecast_list_output_pd is not None:
                        forecast_omc.output_forecast_forward(forecast_list_output_pd, iterative)
                        forecast_evaluation_pd = getattr(
                            forecast_model,
                            'latest_dashboard_metrics_pd',
                            None,
                        )
                        if forecast_evaluation_pd is None:
                            self.logger.warning(
                                "Forecast dashboard".ljust(self.justif - 2, '.') + 'SKIPPED - NO EVALUATION DATA'
                            )
                        else:
                            forecast_omc.output_combined_dashboard_viz_file(
                                forecast_list_output_pd,
                                station_history_dict=station_imputed_dict,
                                evaluation_pd=forecast_evaluation_pd,
                            )
                            self.logger.info("Forecast dashboard".ljust(self.justif - 2, '.') + 'OK')
                    else:
                        self.logger.warning(
                            "Forecast dashboard".ljust(self.justif - 2, '.') + 'SKIPPED - NO FORECAST OUTPUT'
                        )

                ########################################################
                # Forecast mode: build the dashboard file
                ########################################################
                else:
                    self.build_dashboard_file = True
                    if self.build_dashboard_file and list_output_pd is not None:
                        evaluation_pd, evaluation_file_exist = self.IMC.load_evaluation_file_for_dashboard(iterative)
                        if not evaluation_file_exist:
                            self.logger.warning("Evaluation metrics files".ljust(self.justif - 9, '.') + 'NOT FOUND')
                            evaluation_pd = getattr(self.SLSTM, 'latest_dashboard_metrics_pd', None)
                            if evaluation_pd is None:
                                self.logger.warning("Dashboard file will be saved without metrics".ljust(self.justif, '.'))
                            else:
                                self.logger.warning("Using recent hindcast metrics".ljust(self.justif, '.'))

                        self.OMC.output_combined_dashboard_viz_file(
                            list_output_pd,
                            station_history_dict=station_imputed_dict,
                            evaluation_pd=evaluation_pd,
                            )
                        self.logger.info("Dashboard file".ljust(self.justif - 2, '.') + 'OK')
                                

            # ── Expose preprocessing results for reuse by subsequent configs ──
            # __main__ reads these via getattr() after the first forecast run
            # and passes them back as shared_* args for all subsequent configs.
            if not self.Configuration.train_model:
                self._last_input_data_pd   = _copy_dataframe(input_data_pd_list[0])
                self._last_input_metadata  = {
                    self.Data_input_flow[0]: {
                        "input_data": [_copy_dataframe(input_data_pd_list[0])],
                        "input_column_names": list(input_column_names),
                        "output_column_names": list(output_column_names),
                        "specie_properties_dict": copy.deepcopy(specie_properties_dict),
                    }
                }
                self._last_imputed_dict    = _copy_dataframe_dict(station_imputed_dict)
                self._last_decomposed_dict = _copy_dataframe_dict(data_for_model)
            # ─────────────────────────────────────────────────────────────────

            # ### Plot the training_validating plots
            if self.plot_results and self.Configuration.train_model and train_hist is not None:
                training_plot_data = getattr(self.SLSTM, "latest_training_plot_data", None)
                if training_plot_data:
                    long_frame, wide_frame = self._build_visualization_frames_from_training(
                        training_plot_data
                    )
                    self._export_visualization_plots(long_frame, wide_frame)
                else:
                    self.logger.warning(
                        'Visualization export'.ljust(self.justif - 2, '.') + 'SKIPPED - NO TRAINING PLOT DATA'
                    )
            
            # # plot forecast 
            # if (not self.Configuration.train_model):
            #     Plot_results.plot_forecast(list_output_pd[0])
        _run_single_pass()

###########################################################################################
if __name__ == '__main__':
    # Shared timestamp — all horizons in this batch share the same run time
    # so output directories are consistent across hr_3/6/9/12.
    shared_timestamp = dtime.datetime.now()
    process_start_time = time.time()
    summary_logger = IL.Initialise_logging("summary", log_dir=LOG_DIR)
    total_preprocess_seconds = 0.0
    total_model_seconds = 0.0

    # ── Read config files once and expand multi-target / multi-region jobs ────
    # The model code is single-pollutant per run.  A config can now specify:
    #   var_to_predict: ["O3", "PM2.5", "PM10"]
    #   selected_region: "ALL"
    # Each target is expanded to only the regions available for that target in
    # DPE_region_stations.py.
    _tmp_logger = IL.Initialise_logging("_setup", log_dir=LOG_DIR)
    _tmp_cfg = CC.Configuration_Class(
        _tmp_logger, 102,
        dtime.datetime(2021,11,1,0), dtime.datetime(2026,2,23,0),
        shared_timestamp, shared_timestamp, shared_timestamp,
        "data", os.path.join(AI_RUNS_DIR_NAME, "Training"), os.path.join(AI_RUNS_DIR_NAME, "Forecast"),
        os.path.join(AI_RUNS_DIR_NAME, "Model_weights"), "Core_iHPC/Tools/Config_testing",
        (
            "/mnt/scratch_lustre/ar_ai4ba_scratch/Ai4BetterAir/AI_Nowcasting/cnn_lstm_forecast/AI_dashboard_files"
        ), 1, 1, "toto",
    )
    _tmp_yfrc = YFR.Yaml_file_reader_Class(_tmp_cfg)
    list_configs = discover_config_files(_tmp_cfg.Code_configuration_main_dir)

    run_plan = []
    default_targets = [log_target_token(target) for target in flatten_config_list(DEFAULT_VAR_TO_PREDICT)]
    for yaml_config_filename in list_configs:
        yaml_file_dict = _tmp_yfrc.read_yaml_config_file(yaml_config_filename)
        yaml_file_dict.setdefault("train_model", DEFAULT_TRAIN_MODEL)
        explicit_targets = [log_target_token(target) for target in flatten_config_list(yaml_file_dict.get("var_to_predict"))]
        candidate_targets = explicit_targets or [
            log_target_token(target)
            for target in infer_targets_from_config_name(yaml_config_filename, yaml_file_dict)
        ]
        if not candidate_targets:
            print(
                "Skipping {config}: unable to determine pollutant target. "
                "Add var_to_predict explicitly or set DEFAULT_VAR_TO_PREDICT.".format(
                    config=yaml_config_filename
                )
            )
            continue

        targets = [
            target for target in candidate_targets
            if not default_targets or target in default_targets
        ]
        if not targets:
            print(
                "Skipping {config}: target(s) {targets} are not enabled by DEFAULT_VAR_TO_PREDICT {default}".format(
                    config=yaml_config_filename,
                    targets=candidate_targets,
                    default=default_targets,
                )
            )
            continue

        for target in targets:
            target_region_map = get_region_stations_dict(DPERT.DPE_region_stations([target]))
            regions = resolve_regions_for_target(
                yaml_file_dict.get("selected_region", DEFAULT_SELECTED_REGION),
                target_region_map,
            )
            yaml_for_target = apply_runtime_overrides(
                yaml_file_dict,
                target=target,
            )
            target_model_root = model_root_path(yaml_for_target)

            if not model_root_exists(yaml_for_target):
                print(
                    "Skipping {config}: no model weights for {target} "
                    "({model}_{version})".format(
                        config=yaml_config_filename,
                        target=target,
                        model=yaml_for_target["model_name"],
                        version=yaml_for_target.get("model_version", "v1.0"),
                    )
                )
                continue

            for region in regions:
                yaml_for_run = apply_runtime_overrides(
                    yaml_file_dict,
                    target=target,
                    region=region,
                )
                station_count = len(target_region_map[region])

                station_count_ok, expected_station_count = model_station_count_matches(
                    yaml_for_run,
                    station_count,
                )
                print(
                    "Plan candidate: {config} / {region} / {target} | "
                    "model_root={model_root} | stations={actual} | "
                    "expected_outputs={expected} | include={include}".format(
                        config=yaml_config_filename,
                        region=region,
                        target=target,
                        model_root=target_model_root,
                        actual=station_count,
                        expected=expected_station_count,
                        include=station_count_ok,
                    )
                )
                if not station_count_ok:
                    print(
                        "Skipping {config}: {region} / {target} has {actual} stations, "
                        "but model {model}_{version} expects {expected}".format(
                            config=yaml_config_filename,
                            region=region,
                            target=target,
                            actual=station_count,
                            model=yaml_for_run["model_name"],
                            version=yaml_for_run.get("model_version", "v1.0"),
                            expected=expected_station_count,
                        )
                    )
                    continue

                resolved_forecast_method = resolve_forecast_method(yaml_for_run, target)
                resolved_model_name = resolve_model_name(
                    yaml_for_run,
                    target=target,
                    forecast_method=resolved_forecast_method,
                    lags=yaml_for_run.get("lags", yaml_for_run.get("model_parameters", {}).get("n_outputs", 24)),
                )
                resolved_imputation_method = resolve_imputation_method(yaml_for_run)
                resolved_additional_inputs = tuple(flatten_config_list(yaml_for_run.get("additional_var_to_select")))

                run_plan.append({
                    "yaml_config_filename": yaml_config_filename,
                    "target": target,
                    "region": region,
                    "shared_data_dir": shared_data_dir_for_run(yaml_for_run),
                    "cache_key": (
                        region,
                        target,
                        resolved_forecast_method,
                        resolved_model_name,
                        resolved_imputation_method,
                        resolved_additional_inputs,
                        PIPELINE_MODE_RESOLVED,
                        MASTER_USE_DECOMPOSE,
                        MASTER_USE_ONE_MODEL_PER_IMF,
                    ),
                })

    if len(run_plan) == 0:
        raise RuntimeError("No runnable forecast jobs were resolved from list_configs.")

    def _horizon_sort_key(job):
        horizon_token = str(job["yaml_config_filename"]).split("_hr_")[-1] if "_hr_" in job["yaml_config_filename"] else "na"
        try:
            horizon_value = int(str(horizon_token).split(".")[0])
        except Exception:
            horizon_value = 10**9
        return (
            job["region"],
            job["target"],
            job["cache_key"],
            horizon_value,
            job["yaml_config_filename"],
        )

    run_plan = sorted(run_plan, key=_horizon_sort_key)

    print("Resolved forecast jobs:")
    for job in run_plan:
        print("  - {config}: {region} / {target}".format(
            config=job["yaml_config_filename"],
            region=job["region"],
            target=job["target"],
        ))
    print("Resolved job count: {}".format(len(run_plan)))
    # ─────────────────────────────────────────────────────────────────────────

    # ── In-memory preprocessing cache ────────────────────────────────────────
    # Populated per (region, target, model family, pipeline mode). This keeps
    # hr_3/6/9/12 reuse while preventing cross-pollutant contamination.
    _shared_preprocessing_cache = {}
    # ─────────────────────────────────────────────────────────────────────────

    # ── Precompute the shared preprocessing stage once per cache key ────────
    # The impute/VMD stack does not depend on forecast horizon. We run it once
    # here, then each horizon job only consumes the prepared data.
    seed_jobs = {}
    for job in run_plan:
        seed_jobs.setdefault(job["cache_key"], job)

    for cache_key, seed_job in seed_jobs.items():
        loggername = "{pollutant}_{region}_{model}_{imputation}_{mode}".format(
            pollutant=log_target_token(seed_job["target"]),
            region=seed_job["region"],
            model=resolve_forecast_method(seed_job, seed_job["target"]),
            imputation=resolve_imputation_method(seed_job),
            mode=PIPELINE_MODE_RESOLVED,
        )
        logger = IL.Initialise_logging(loggername, log_dir=LOG_DIR)
        preprocess_start = time.time()
        preprocessor = Sparse_LSTM_Class(logger, 102, seed_job["yaml_config_filename"])
        preprocessor.shared_timestamp = shared_timestamp
        logger.info(
            "Shared preprocessing start for {config} / {region} / {target}".format(
                config=seed_job["yaml_config_filename"],
                region=seed_job["region"],
                target=seed_job["target"],
            ).ljust(102 - 2, '.') + 'RUN'
        )
        preprocessor.run_all(
            shared_data_dir=seed_job["shared_data_dir"],
            override_target=seed_job["target"],
            override_region=seed_job["region"],
            preprocess_only=True,
        )
        preprocess_elapsed = time.time() - preprocess_start
        total_preprocess_seconds += preprocess_elapsed
        _shared_preprocessing_cache[cache_key] = {
            "input_data_pd": _copy_dataframe(getattr(preprocessor, '_last_input_data_pd', None)),
            "input_metadata": copy.deepcopy(getattr(preprocessor, '_last_input_metadata', None)),
            "imputed_dict": _copy_dataframe_dict(getattr(preprocessor, '_last_imputed_dict', None)),
            "decomposed_dict": _copy_dataframe_dict(getattr(preprocessor, '_last_decomposed_dict', None)),
        }
        if _shared_preprocessing_cache[cache_key]["input_data_pd"] is None:
            raise RuntimeError(
                "Preprocessing cache was not populated for {config} / {region} / {target}".format(
                    config=seed_job["yaml_config_filename"],
                    region=seed_job["region"],
                    target=seed_job["target"],
                )
            )
        logger.info(
            "Shared preprocessing captured once for cache_key={cache_key}".format(
                cache_key=cache_key
            ).ljust(102 - 2, '.') + 'OK'
        )
        summary_logger.info(
            "Preprocess {config} / {region} / {target}: {seconds:.2f}s".format(
                config=seed_job["yaml_config_filename"],
                region=seed_job["region"],
                target=seed_job["target"],
                seconds=preprocess_elapsed,
            )
        )
    # ────────────────────────────────────────────────────────────────────────

    for job in run_plan:
        ################ logger init #################
        justif = 102
        yaml_config_filename = job["yaml_config_filename"]
        target_token = log_target_token(job["target"])
        loggername = "{pollutant}_{region}_{model}_{imputation}_{mode}".format(
            pollutant=target_token,
            region=job["region"],
            model=resolve_forecast_method(job, job["target"]),
            imputation=resolve_imputation_method(job),
            mode=PIPELINE_MODE_RESOLVED,
        )
        logger = IL.Initialise_logging(loggername, log_dir=LOG_DIR)

        cache_entry = _shared_preprocessing_cache.get(job["cache_key"], {})

        ### Define object of the class
        Sparse_LSTM = Sparse_LSTM_Class(logger, justif, yaml_config_filename)
        Sparse_LSTM.shared_timestamp = shared_timestamp

        ### Running main program — pass shared preprocessing results if available
        job_start_time = time.time()
        logger.info(
            "Model run start for {config} / {region} / {target}".format(
                config=yaml_config_filename,
                region=job["region"],
                target=job["target"],
            ).ljust(justif - 2, '.') + 'RUN'
        )
        Sparse_LSTM.run_all(
            shared_input_data_pd   = cache_entry.get("input_data_pd"),
            shared_input_metadata  = cache_entry.get("input_metadata"),
            shared_imputed_dict    = cache_entry.get("imputed_dict"),
            shared_decomposed_dict = cache_entry.get("decomposed_dict"),
            shared_data_dir        = job["shared_data_dir"],
            override_target        = job["target"],
            override_region        = job["region"],
        )
        job_elapsed = time.time() - job_start_time
        total_model_seconds += job_elapsed
        summary_logger.info(
            "Model run {config} / {region} / {target}: {seconds:.2f}s".format(
                config=yaml_config_filename,
                region=job["region"],
                target=job["target"],
                seconds=job_elapsed,
            )
        )

        metrics_csv_path = resolve_metrics_csv_path(Sparse_LSTM.Configuration)
        if append_total_runtime_row(metrics_csv_path, job_elapsed):
            logger.info(
                f'  {os.path.basename(metrics_csv_path)}'.ljust(justif - 2, '.') + 'RUNTIME APPENDED'
            )

    total_process_seconds = time.time() - process_start_time
    summary_logger.info("Total preprocess time: {seconds:.2f}s".format(seconds=total_preprocess_seconds))
    summary_logger.info("Total model run time: {seconds:.2f}s".format(seconds=total_model_seconds))
    summary_logger.info("Total process run time: {seconds:.2f}s".format(seconds=total_process_seconds))

    print("-" * 110)
    print("|||||||||||||||||||||||||||||||||||||||||!!!!Finished!!!!|||||||||||||||||||||||||||||||||||||||||||||")
    print("-" * 110)
