# gating_surrogate.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def build_case_features(
    df: pd.DataFrame,
    dt_col: str,
    target_col: str,
    mask_kind: str,
    block_hours: Optional[int] = None,
) -> pd.DataFrame:
    """
    Creates a *row-level* feature table aligned to df rows.
    Gate learns to pick the best imputer for masked rows.
    """
    out = pd.DataFrame(index=df.index)

    # time features
    dt = pd.to_datetime(df[dt_col])
    out["hour"] = dt.dt.hour
    out["dow"] = dt.dt.dayofweek
    out["month"] = dt.dt.month

    # sinusoidal (optional)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)

    # target context (using available values; masked rows will be NaN, so use rolling on original df if you want)
    # Here we compute rolling stats from current df values (works if you compute before masking).
    y = pd.to_numeric(df[target_col], errors="coerce")
    out["y_roll_mean_24"] = y.rolling(24, min_periods=6).mean()
    out["y_roll_std_24"] = y.rolling(24, min_periods=6).std()
    out["y_roll_mean_72"] = y.rolling(72, min_periods=12).mean()
    out["y_roll_std_72"] = y.rolling(72, min_periods=12).std()

    out["mask_kind"] = 0 if mask_kind == "temporal_block" else 1
    out["block_hours"] = float(block_hours) if block_hours is not None else 0.0

    # fill
    return out.ffill().bfill().fillna(0.0)


def train_gate_classifier(
    X: pd.DataFrame,
    y_best_model: np.ndarray,
    method: str = "lightgbm",
    seed: int = 42,
) -> Dict:
    """
    y_best_model is integer-encoded model id for each training case.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_best_model, test_size=0.2, random_state=seed, stratify=y_best_model
    )

    if method.lower() == "lightgbm":
        if lgb is None:
            raise RuntimeError("lightgbm not installed, cannot train gate with LightGBM.")
        clf = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
        )
    elif method.lower() == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost not installed, cannot train gate with XGBoost.")
        clf = XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            tree_method="hist",
            eval_metric="mlogloss",
        )
    else:
        raise ValueError("method must be 'lightgbm' or 'xgboost'")

    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)
    acc = float(accuracy_score(y_val, pred))

    return {"model": clf, "val_accuracy": acc, "feature_names": list(X.columns)}


def predict_gate(
    gate_obj: Dict,
    X_cases: pd.DataFrame,
) -> np.ndarray:
    clf = gate_obj["model"]
    # ensure column alignment
    cols = gate_obj["feature_names"]
    Xc = X_cases[cols].copy()
    return clf.predict(Xc)
