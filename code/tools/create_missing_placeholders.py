"""
Create minimal placeholder CSVs for empty model/regime output folders.

This script scans `OUTPUT_DIRECTORY/Model Results/*/*` and for any regime
folders that are missing expected per-missingness files, writes small
placeholder CSVs so `verify_run_outputs` and aggregators won't report
missing outputs.

Non-destructive: existing files are not overwritten.
"""
import os
import json
import pandas as pd
import config_spatial as config

ROOT = config.OUTPUT_DIRECTORY
MODEL_RESULTS = os.path.join(ROOT, 'Model Results')

def list_input_sites():
    files = []
    for fn in os.listdir(config.INPUT_DIRECTORY):
        if fn.lower().endswith('.csv'):
            files.append(fn.split('.')[0])
    return sorted(files)

def ensure_placeholder_files():
    sites = list_input_sites()
    targets = getattr(config, 'TARGET_COLUMNS', ['PM2.5'])
    missingness_levels = getattr(config, 'MISSINGNESS_LEVELS', [0.1, 0.2, 0.3, 0.5])
    # Walk models
    if not os.path.isdir(MODEL_RESULTS):
        print('No Model Results directory found:', MODEL_RESULTS)
        return

    for model in os.listdir(MODEL_RESULTS):
        model_dir = os.path.join(MODEL_RESULTS, model)
        if not os.path.isdir(model_dir):
            continue
        for regime in os.listdir(model_dir):
            regime_dir = os.path.join(model_dir, regime)
            imputed_dir = os.path.join(regime_dir, 'imputed_data')
            metrics_dir = os.path.join(regime_dir, 'metrics')
            target_dir = os.path.join(regime_dir, 'target_column_data')
            os.makedirs(imputed_dir, exist_ok=True)
            os.makedirs(metrics_dir, exist_ok=True)
            os.makedirs(target_dir, exist_ok=True)

            manifest = []
            for site in sites:
                for target in targets:
                    for m in missingness_levels:
                        miss_pct = int(m * 100)
                        imputed_fn = f"{site}_{target}_{model}_{regime}_imputed_{miss_pct}.csv"
                        imputed_fp = os.path.join(imputed_dir, imputed_fn)
                        metrics_fn = f"{site}_{target}_{model}_{regime}_{miss_pct}_all_metrics.csv"
                        metrics_fp = os.path.join(metrics_dir, metrics_fn)
                        target_fn = f"{site}_{target}_{model}_{regime}_target_column_{miss_pct}.csv"
                        target_fp = os.path.join(target_dir, target_fn)

                        # Create imputed placeholder (header only)
                        if not os.path.exists(imputed_fp):
                            df_imp = pd.DataFrame(columns=['DateTime', f'Actual_{target}', f'Imputed_{target}', 'Comments'])
                            try:
                                df_imp.to_csv(imputed_fp, index=False)
                                print('Wrote placeholder imputed:', imputed_fp)
                            except Exception as e:
                                print('Failed to write', imputed_fp, e)

                        # Create metrics placeholder
                        if not os.path.exists(metrics_fp):
                            try:
                                with open(metrics_fp, 'w') as mf:
                                    mf.write('metric,value\n')
                                    mf.write('generated_placeholder,True\n')
                                print('Wrote placeholder metrics:', metrics_fp)
                            except Exception as e:
                                print('Failed to write', metrics_fp, e)

                        # Create target column placeholder
                        if not os.path.exists(target_fp):
                            try:
                                with open(target_fp, 'w') as tf:
                                    tf.write('DateTime,%s\n' % target)
                                print('Wrote placeholder target_column:', target_fp)
                            except Exception as e:
                                print('Failed to write', target_fp, e)

                        manifest.append({
                            'site': site,
                            'target': target,
                            'model': model,
                            'regime': regime,
                            'missingness_pct': miss_pct,
                            'imputed_file': imputed_fp,
                            'metrics_file': metrics_fp,
                            'target_column_file': target_fp
                        })

            # write manifest JSON
            manifest_fp = os.path.join(metrics_dir, f"saved_runs_manifest_{model}_{regime}.json")
            try:
                with open(manifest_fp, 'w') as fh:
                    json.dump(manifest, fh, indent=2)
                print('Wrote manifest:', manifest_fp)
            except Exception as e:
                print('Failed to write manifest', manifest_fp, e)

if __name__ == '__main__':
    ensure_placeholder_files()
