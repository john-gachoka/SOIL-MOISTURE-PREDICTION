# src/models.py

"""
ML models for Objective 2: training, cross-validation, SHAP, and prediction.

Responsibilities:
    - Factory functions for RandomForest and XGBoost.
    - Spatial block cross-validation.
    - Temporal cross-validation.
    - Final model fitting on the training window.
    - SHAP computation on a bounded sample.
    - Prediction onto the full (time, y, x) grid.
    - A single orchestrator (`run_training`) driven by a config dict.

All results are written to disk with a run-id tag so that repeated runs
are reproducible and distinguishable.
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from src.config import (
    MODELS_DIR, ML_DIR, RUN_LOG_DIR, PRED_DIR,
    TARGET_NAME, DEFAULT_FEATURES,
    TRAIN_YEARS, VALID_YEARS,
    DEFAULT_SEED, DEFAULT_SPATIAL_K, DEFAULT_SHAP_SAMPLE,
    DEFAULT_TEST_FRAC, SEASONS,
)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_rf(params: Dict[str, Any]) -> RandomForestRegressor:
    """Build a RandomForestRegressor from a config dict."""
    defaults = dict(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        max_features='sqrt',
        n_jobs=-1,
        random_state=DEFAULT_SEED,
    )
    defaults.update(params or {})
    return RandomForestRegressor(**defaults)


def make_xgb(params: Dict[str, Any]) -> XGBRegressor:
    """Build an XGBRegressor from a config dict."""
    defaults = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        tree_method='hist',
        n_jobs=-1,
        random_state=DEFAULT_SEED,
    )
    defaults.update(params or {})
    return XGBRegressor(**defaults)


MODEL_FACTORY = {
    'rf':  make_rf,
    'xgb': make_xgb,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics. Returns a dict so it can be appended to a
    DataFrame row directly.
    """
    y_true = np.asarray(y_true, dtype='float64')
    y_pred = np.asarray(y_pred, dtype='float64')
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return dict(n=0, r2=np.nan, rmse=np.nan, mae=np.nan,
                    bias=np.nan, slope=np.nan)
    r2 = r2_score(y_true, y_pred) if y_true.size > 1 else np.nan
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    # Slope of predicted vs observed (via least squares through origin-free
    # linear fit). y_true is x, y_pred is y.
    if y_true.size > 1 and np.std(y_true) > 0:
        slope = float(np.polyfit(y_true, y_pred, 1)[0])
    else:
        slope = np.nan
    return dict(
        n=int(y_true.size),
        r2=float(r2),
        rmse=rmse,
        mae=mae,
        bias=bias,
        slope=slope,
    )


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def spatial_block_cv(df: pd.DataFrame,
                     model_factory,
                     params: Dict[str, Any],
                     feature_cols: Sequence[str],
                     target_col: str = TARGET_NAME,
                     fold_col: str = 'spatial_fold') -> pd.DataFrame:
    """
    Leave-one-block-out spatial CV. Returns a DataFrame with one row per
    fold and the metrics computed on the held-out block.
    """
    if fold_col not in df.columns:
        raise KeyError(
            f"Column '{fold_col}' missing. Call src.features.spatial_blocks "
            f"before this function."
        )
    folds = sorted(df[fold_col].unique())
    rows = []
    X_all = df[list(feature_cols)].values.astype('float32')
    y_all = df[target_col].values.astype('float32')
    fold_all = df[fold_col].values

    for f in folds:
        train_mask = fold_all != f
        test_mask  = fold_all == f
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        model = model_factory(params)
        model.fit(X_all[train_mask], y_all[train_mask])
        pred = model.predict(X_all[test_mask])
        m = regression_metrics(y_all[test_mask], pred)
        m['fold'] = int(f)
        m['scheme'] = 'spatial'
        rows.append(m)
    return pd.DataFrame(rows)


def temporal_cv(df: pd.DataFrame,
                model_factory,
                params: Dict[str, Any],
                feature_cols: Sequence[str],
                target_col: str = TARGET_NAME,
                fold_col: str = 'temporal_fold') -> pd.DataFrame:
    """
    Leave-one-year-block-out temporal CV. Returns a DataFrame with one
    row per fold and the metrics computed on the held-out years.
    """
    if fold_col not in df.columns:
        raise KeyError(
            f"Column '{fold_col}' missing. Call src.features.temporal_folds "
            f"before this function."
        )
    folds = sorted(df[fold_col].unique())
    rows = []
    X_all = df[list(feature_cols)].values.astype('float32')
    y_all = df[target_col].values.astype('float32')
    fold_all = df[fold_col].values

    for f in folds:
        train_mask = fold_all != f
        test_mask  = fold_all == f
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        model = model_factory(params)
        model.fit(X_all[train_mask], y_all[train_mask])
        pred = model.predict(X_all[test_mask])
        m = regression_metrics(y_all[test_mask], pred)
        m['fold'] = int(f)
        m['scheme'] = 'temporal'
        # Record which years were held out so the metrics table is readable.
        held_out_years = sorted(df.loc[test_mask, 'year'].unique().tolist())
        m['held_out_years'] = ','.join(str(y) for y in held_out_years)
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Final fit
# ---------------------------------------------------------------------------

def train_final(df: pd.DataFrame,
                model_factory,
                params: Dict[str, Any],
                feature_cols: Sequence[str],
                target_col: str = TARGET_NAME):
    """
    Fit the model on all rows of df. Returns the fitted estimator.
    """
    X = df[list(feature_cols)].values.astype('float32')
    y = df[target_col].values.astype('float32')
    model = model_factory(params)
    print(f"Fitting final model on {len(df):,} rows, "
          f"{len(feature_cols)} features...")
    t0 = time.time()
    model.fit(X, y)
    print(f"  fit completed in {time.time() - t0:.1f}s")
    return model


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def compute_shap(model,
                 X_background: np.ndarray,
                 X_explain: np.ndarray,
                 feature_names: Sequence[str]) -> Dict[str, Any]:
    """
    Compute SHAP values on a bounded sample.

    Uses TreeExplainer for tree-based models (RF, XGB). Returns a dict with:
        values:      (n_samples, n_features) array of SHAP values
        base_value:  scalar or vector
        feature_names: list of column names
        importance:  DataFrame sorted by mean |SHAP|
    """
    if not HAS_SHAP:
        raise RuntimeError(
            "shap package is not installed. Run: pip install shap"
        )
    print(f"Computing SHAP on {len(X_explain):,} rows, "
          f"background {len(X_background):,} rows...")
    explainer = shap.TreeExplainer(
        model,
        data=X_background,
        feature_perturbation='interventional',
    )
    sv = explainer.shap_values(X_explain, check_additivity=False)
    sv = np.asarray(sv)
    if sv.ndim == 3:
        # Some XGBoost/SHAP combinations return (n, features, n_targets).
        # For a single-output regressor, squeeze the trailing dimension.
        sv = sv[..., 0]
    mean_abs = np.mean(np.abs(sv), axis=0)
    importance = (
        pd.DataFrame({
            'feature': list(feature_names),
            'mean_abs_shap': mean_abs,
        })
        .sort_values('mean_abs_shap', ascending=False)
        .reset_index(drop=True)
    )
    return dict(
        values=sv,
        base_value=explainer.expected_value,
        feature_names=list(feature_names),
        importance=importance,
    )


def _sample_for_shap(df: pd.DataFrame,
                     feature_cols: Sequence[str],
                     n: int,
                     seed: int = DEFAULT_SEED
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample rows for SHAP background and explanation sets.
    Returns (X_background, X_explain), each of shape (n_bg or n_exp, n_feat).
    The background is half the size of the explanation set.
    """
    n = min(n, len(df))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n, replace=False)
    sample = df.iloc[idx]
    n_bg = max(100, n // 4)
    bg_idx = rng.choice(n, size=n_bg, replace=False)
    X_explain = sample[list(feature_cols)].values.astype('float32')
    X_bg = X_explain[bg_idx]
    return X_bg, X_explain


# ---------------------------------------------------------------------------
# Prediction onto the full grid
# ---------------------------------------------------------------------------

def predict_map(model,
                ds: xr.Dataset,
                feature_cols: Sequence[str]) -> xr.DataArray:
    """
    Apply the model to every (time, y, x) in `ds`.

    Returns an xr.DataArray named 'SM_PRED' with the same coords as ds.
    NaNs in the feature stack propagate as NaNs in the prediction.
    """
    nt = ds.sizes['time']
    ny = ds.sizes['y']
    nx = ds.sizes['x']

    # Build the (nt*ny*nx, n_features) matrix
    X = np.empty((nt * ny * nx, len(feature_cols)), dtype='float32')
    for i, f in enumerate(feature_cols):
        X[:, i] = ds[f].values.ravel()

    # Valid rows: no NaN in any feature
    valid = np.isfinite(X).all(axis=1)
    print(f"Predicting on {valid.sum():,} of {valid.size:,} pixels "
          f"({100.0 * valid.sum() / valid.size:.2f}% valid).")

    y_pred = np.full(X.shape[0], np.nan, dtype='float32')
    if valid.any():
        y_pred[valid] = model.predict(X[valid]).astype('float32')

    pred = xr.DataArray(
        y_pred.reshape(nt, ny, nx),
        dims=('time', 'y', 'x'),
        coords={
            'time':   ds.time,
            'year':   ('time', ds.year.values),
            'season': ('time', ds.season.values),
            'y':      ds.y,
            'x':      ds.x,
        },
        name='SM_PRED',
        attrs={'units': 'm3/m3', 'crs': str(ds.rio.crs)},
    )
    pred = pred.rio.write_crs(ds.rio.crs, inplace=False)
    return pred


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_training(cfg: Dict[str, Any],
                 df_train: pd.DataFrame,
                 df_valid: pd.DataFrame,
                 ds_full: xr.Dataset) -> Dict[str, Any]:
    """
    Full training pipeline for one config.

    Parameters
    ----------
    cfg : dict
        Must contain 'model_type' ('rf' or 'xgb'), optionally 'params',
        'feature_cols', 'run_id', 'spatial_k', 'temporal_folds',
        'shap_sample'.
    df_train : DataFrame
        Rows for fitting and CV.
    df_valid : DataFrame
        Rows for the temporal holdout (2015-2025 by default). Used only for
        a final validation metric, not for CV.
    ds_full : xr.Dataset
        The aligned (time, y, x) Dataset, used for producing the
        prediction stack.

    Returns
    -------
    dict with keys: run_id, model_path, cv_metrics_path, shap_path,
    pred_path, run_log_path, metrics_summary.
    """
    from src.features import spatial_blocks, temporal_folds

    model_type = cfg['model_type']
    if model_type not in MODEL_FACTORY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from {list(MODEL_FACTORY)}."
        )
    factory = MODEL_FACTORY[model_type]
    params  = cfg.get('params', {})
    feature_cols = cfg.get('feature_cols', DEFAULT_FEATURES)
    run_id = cfg.get('run_id') or (
        f"{model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )
    print(f"\n=== run_training: {run_id} ===")
    print(f"  model_type:   {model_type}")
    print(f"  features:     {len(feature_cols)}")
    print(f"  train rows:   {len(df_train):,}")
    print(f"  valid rows:   {len(df_valid):,}")

    # Fold assignment
    k = int(cfg.get('spatial_k', DEFAULT_SPATIAL_K))
    n_temporal = int(cfg.get('temporal_folds', 5))
    df_train = spatial_blocks(df_train, k=k, by='y_idx')
    df_train = temporal_folds(df_train, n_folds=n_temporal)

    # CV
    print("\nRunning spatial block CV...")
    spatial_cv = spatial_block_cv(df_train, factory, params,
                                  feature_cols=feature_cols)
    print("Running temporal CV...")
    temporal_cv_df = temporal_cv(df_train, factory, params,
                                 feature_cols=feature_cols)

    cv_all = pd.concat([spatial_cv, temporal_cv_df], ignore_index=True)
    cv_path = ML_DIR / f'{run_id}_cv.csv'
    cv_all.to_csv(cv_path, index=False)
    print(f"CV metrics -> {cv_path}")

    # Pooled CV summary per scheme
    summary = (
        cv_all.groupby('scheme')
              .agg(n=('n', 'sum'),
                   r2_mean=('r2', 'mean'),
                   rmse_mean=('rmse', 'mean'),
                   mae_mean=('mae', 'mean'),
                   bias_mean=('bias', 'mean'))
              .reset_index()
    )
    print("\nCV summary:")
    print(summary.to_string(index=False))

    # Final fit on all training rows
    print("\nFitting final model on all training rows...")
    model = train_final(df_train, factory, params,
                        feature_cols=feature_cols)
    model_path = MODELS_DIR / f'{run_id}_model.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved -> {model_path}")

    # SHAP on a bounded sample
    shap_path = None
    shap_importance_df = None
    if cfg.get('compute_shap', True) and HAS_SHAP:
        n_shap = int(cfg.get('shap_sample', DEFAULT_SHAP_SAMPLE))
        X_bg, X_ex = _sample_for_shap(df_train, feature_cols, n=n_shap)
        shap_result = compute_shap(model, X_bg, X_ex, feature_cols)
        shap_path = ML_DIR / f'{run_id}_shap.npz'
        np.savez_compressed(
            shap_path,
            values=shap_result['values'],
            base_value=shap_result['base_value'],
            feature_names=np.array(shap_result['feature_names']),
        )
        shap_importance_df = shap_result['importance']
        imp_path = ML_DIR / f'{run_id}_shap_importance.csv'
        shap_importance_df.to_csv(imp_path, index=False)
        print(f"SHAP values      -> {shap_path}")
        print(f"SHAP importance  -> {imp_path}")
        print("\nTop 10 features by mean |SHAP|:")
        print(shap_importance_df.head(10).to_string(index=False))

    # Validation metrics on the temporal holdout (2015-2025)
    valid_metrics = None
    if len(df_valid) > 0:
        Xv = df_valid[list(feature_cols)].values.astype('float32')
        yv = df_valid[TARGET_NAME].values.astype('float32')
        yhat = model.predict(Xv).astype('float32')
        valid_metrics = regression_metrics(yv, yhat)
        valid_metrics['scheme'] = 'temporal_holdout'
        valid_metrics['fold'] = '2015-2025'
        print("\nTemporal holdout metrics (2015-2025):")
        print(pd.DataFrame([valid_metrics]).to_string(index=False))

    # Prediction on full grid
    print("\nPredicting on full grid...")
    pred = predict_map(model, ds_full, feature_cols=feature_cols)
    pred_path = PRED_DIR / f'{run_id}_SM_PRED.nc'
    pred.to_netcdf(pred_path, engine='netcdf4')
    print(f"Predictions -> {pred_path}")

    # Run log
    run_log = {
        'run_id':         run_id,
        'model_type':     model_type,
        'params':         params,
        'feature_cols':   list(feature_cols),
        'n_train_rows':   int(len(df_train)),
        'n_valid_rows':   int(len(df_valid)),
        'cv_summary':     summary.to_dict(orient='records'),
        'valid_metrics':  valid_metrics,
        'artifacts': {
            'model':           str(model_path),
            'cv_metrics':      str(cv_path),
            'shap_values':     str(shap_path) if shap_path else None,
            'shap_importance': str(imp_path) if shap_importance_df is not None else None,
            'predictions':     str(pred_path),
        },
        'started_at':     datetime.utcnow().isoformat() + 'Z',
    }
    run_log_path = RUN_LOG_DIR / f'{run_id}.json'
    with open(run_log_path, 'w') as f:
        json.dump(run_log, f, indent=2, default=str)
    print(f"Run log -> {run_log_path}")

    return dict(
        run_id=run_id,
        model_path=model_path,
        cv_metrics_path=cv_path,
        shap_path=shap_path,
        pred_path=pred_path,
        run_log_path=run_log_path,
        metrics_summary=summary,
        valid_metrics=valid_metrics,
    )