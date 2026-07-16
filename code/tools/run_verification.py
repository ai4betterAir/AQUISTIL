"""
Lightweight verification of model/regime outputs.

Does not import heavy ML modules; reads filesystem to confirm expected
imputed_data, metrics, and target_column_data files exist for each
model/regime/site/level and reports missing files (or confirms presence).

Run from workspace root with Python 3.
"""
import os
import config_spatial as config
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ROOT = config.OUTPUT_DIRECTORY
MODEL_RESULTS = os.path.join(ROOT, 'Model Results')
CENTRAL_IMPUTED = os.path.join(ROOT, 'Imputed_Results')

def dir_has_tolerant_file(dirpath, substrings):
    try:
        if not os.path.isdir(dirpath):
            return False
        for fn in os.listdir(dirpath):
            low = fn.lower()
            if all(s.lower() in low for s in substrings):
                return True
        return False
    except Exception:
        return False

def run():
    if not os.path.isdir(MODEL_RESULTS):
        logging.error('Model Results dir not found: %s', MODEL_RESULTS)
        return 1

    # Discover models and regimes from on-disk structure
    for model in sorted(os.listdir(MODEL_RESULTS)):
        model_dir = os.path.join(MODEL_RESULTS, model)
        if not os.path.isdir(model_dir):
            continue
        for regime in sorted(os.listdir(model_dir)):
            regime_dir = os.path.join(model_dir, regime)
            imputed_dir = os.path.join(regime_dir, 'imputed_data')
            metrics_dir = os.path.join(regime_dir, 'metrics')
            target_dir = os.path.join(regime_dir, 'target_column_data')

            for miss in getattr(config, 'MISSINGNESS_LEVELS', [0.1,0.2,0.3,0.5]):
                miss_pct = int(miss * 100)
                # Heuristic substrings
                # We'll search for any file containing site token + target + model + regime + misspct
                # Discover site tokens from input directory filenames
                for fn in os.listdir(config.INPUT_DIRECTORY):
                    if not fn.lower().endswith('.csv'):
                        continue
                    site_basename = os.path.basename(fn).split('.')[0]
                    target = config.TARGET_COLUMNS[0] if getattr(config, 'TARGET_COLUMNS', None) else 'PM2.5'

                    subs_metrics = [site_basename, target, model, regime, str(miss_pct)]
                    subs_imputed = [site_basename, target, model, 'imputed', str(miss_pct)]
                    subs_target = [site_basename, target, model, 'target_column', str(miss_pct)]

                    missing_files = []

                    if not dir_has_tolerant_file(imputed_dir, subs_imputed):
                        # fallback: central master
                        central_subs = [site_basename, target, 'master_imputed']
                        if not dir_has_tolerant_file(CENTRAL_IMPUTED, central_subs):
                            missing_files.append(os.path.join(imputed_dir, f"*{site_basename}*{model}*imputed*{miss_pct}*.csv"))

                    if not dir_has_tolerant_file(metrics_dir, subs_metrics):
                        missing_files.append(os.path.join(metrics_dir, f"*{site_basename}*{model}*{regime}*{miss_pct}*.csv"))

                    if not dir_has_tolerant_file(target_dir, subs_target):
                        missing_files.append(os.path.join(target_dir, f"*{site_basename}*{model}*target_column*{miss_pct}*.csv"))

                    if missing_files:
                        logging.warning('Missing output files for %s_%s | %s | %s | %d%%:\n  %s',
                                        model, site_basename.split('_')[0], site_basename, regime, miss_pct,
                                        '\n  '.join(missing_files))
                    else:
                        logging.info('Outputs present for %s_%s | %s | %s | %d%%', model, site_basename.split('_')[0], site_basename, regime, miss_pct)

    return 0

if __name__ == '__main__':
    run()
