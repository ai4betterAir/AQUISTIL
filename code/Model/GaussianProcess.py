import logging
import os
import importlib.util

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_NAME = "GaussianProcess"


def _load_impute():
    try:
        from impute import impute_with_method
        return impute_with_method
    except Exception:
        pass
    try:
        from Model.impute import impute_with_method
        return impute_with_method
    except Exception:
        pass
    impute_path = os.path.join(os.path.dirname(__file__), 'impute.py')
    spec = importlib.util.spec_from_file_location('impute_local', impute_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.impute_with_method


impute_with_method = _load_impute()


def impute_mice(data, target_column, input_columns, max_iter=10, random_state=42,
                tol=0.01, custom_strategies=None, spatial_config=None, **kwargs):
    return impute_with_method(
        data, target_column, input_columns,
        method='gp',
        custom_strategies=custom_strategies,
        spatial_config=spatial_config,
        max_iter=max_iter,
        random_state=random_state,
        tol=tol,
        **kwargs
    )
