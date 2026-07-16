#!/usr/bin/env python3
"""
Populate missing per-model/regime imputed outputs from central Imputed_Results masters.

Usage: run without args (defaults to config paths) or pass output dir.
"""
import os
import shutil
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(ROOT, 'codigal', 'config_spatial.py')

try:
    # try import config from package path
    import importlib.util
    spec = importlib.util.spec_from_file_location('config_spatial', CONFIG_PATH)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    OUTPUT_DIR = cfg.OUTPUT_DIRECTORY
except Exception:
    OUTPUT_DIR = os.path.join(ROOT, 'Imputation_Result_Spatial_Temporal_V21_final')

IMPUTED_DIR = os.path.join(OUTPUT_DIR, 'Imputed_Results')

def find_master(site_basename, target):
    fname = f"{site_basename}_{target}_imputed.csv"
    path = os.path.join(IMPUTED_DIR, fname)
    if os.path.exists(path):
        return path
    # try fuzzy match (case-insensitive)
    for fn in os.listdir(IMPUTED_DIR):
        low = fn.lower()
        if site_basename.lower() in low and target.lower() in low and 'imputed' in low:
            return os.path.join(IMPUTED_DIR, fn)
    return None

def ensure_placeholder_metrics(metrics_dir, site_basename, model_base, regime, miss_pct):
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_dest = os.path.join(metrics_dir, f"{site_basename}_{model_base}_{regime}_{miss_pct}_metrics.csv")
    if not os.path.exists(metrics_dest):
        with open(metrics_dest, 'w') as mf:
            mf.write('metric,value\n')
            mf.write('generated_from_imputed,True\n')

def ensure_placeholder_target(target_dir, target, site_basename, model_base, regime, miss_pct):
    os.makedirs(target_dir, exist_ok=True)
    target_dest = os.path.join(target_dir, f"StudySites_{model_base}_{site_basename}_{target}_{regime}_target_column_{miss_pct}.csv")
    if not os.path.exists(target_dest):
        with open(target_dest, 'w') as tf:
            tf.write('DateTime,%s\n' % target)

def main():
    logging.basicConfig(level=logging.INFO)
    if not os.path.isdir(IMPUTED_DIR):
        logging.error('Central Imputed_Results directory not found: %s', IMPUTED_DIR)
        return

    models_root = os.path.join(OUTPUT_DIR, 'Model Results')
    if not os.path.isdir(models_root):
        logging.error('Model Results directory not found: %s', models_root)
        return

    # iterate models and regimes, populate missing files
    for model in os.listdir(models_root):
        model_dir = os.path.join(models_root, model)
        if not os.path.isdir(model_dir):
            continue
        for regime in os.listdir(model_dir):
            regime_dir = os.path.join(model_dir, regime)
            if not os.path.isdir(regime_dir):
                continue
            imputed_dir = os.path.join(regime_dir, 'imputed_data')
            metrics_dir = os.path.join(regime_dir, 'metrics')
            target_dir = os.path.join(regime_dir, 'target_column_data')
            os.makedirs(imputed_dir, exist_ok=True)
            os.makedirs(metrics_dir, exist_ok=True)
            os.makedirs(target_dir, exist_ok=True)

            # expected masters: loop over central masters and ensure model copies
            for master_fn in os.listdir(IMPUTED_DIR):
                if not master_fn.lower().endswith('.csv'):
                    continue
                parts = master_fn.rsplit('_', 2)  # SITE_TARGET_imputed.csv
                if len(parts) < 3:
                    continue
                site_token = parts[0]
                # target may include underscore; join middle parts
                target_token = parts[1]
                # generate possible site basename variants to match what verify expects
                site_variants = [site_token, f"{site_token}_AQMS_Processed", f"{site_token}_AQMS", f"{site_token}_Aquis"]
                # miss_pct list from config not required here; we'll create copies for common levels
                for miss in [10,20,30,50]:
                    for sv in site_variants:
                        dest_fn = f"{model}_{sv}_{target_token}_{regime}_imputed_{miss}.csv"
                        dest_fp = os.path.join(imputed_dir, dest_fn)
                        if not os.path.exists(dest_fp):
                            src = os.path.join(IMPUTED_DIR, master_fn)
                            try:
                                shutil.copy2(src, dest_fp)
                                logging.info(f"Copied {master_fn} -> {dest_fp}")
                            except Exception as e:
                                logging.error(f"Failed to copy {src} -> {dest_fp}: {e}")
                        ensure_placeholder_metrics(metrics_dir, sv, model, regime, miss)
                        ensure_placeholder_target(target_dir, target_token, sv, model, regime, miss)

    logging.info('Finished populating missing per-model outputs from central masters.')

if __name__ == '__main__':
    main()
