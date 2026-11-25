# src/features.py

"""
Feature engineering for the soil moisture ML model (Objective 2).

Responsibilities:
    - Load every predictor stack and the three ERA5 target layers.
    - Build the root-zone target as a thickness-weighted mean of
      SM_L2, SM_L3, SM_L4 (weights from src.config.TARGET_WEIGHTS).
    - Align all stacks onto the target grid and time axis.
    - Flatten (time, y, x) into a tidy pandas DataFrame.
    - Apply QA: drop rows where the target is NaN, drop rows where any
      feature is NaN, drop rows outside the target's valid footprint.
    - Produce a coverage report before any model sees the data.
    - Produce spatial-block and temporal split assignments.

Public API:
    build_feature_table(...)   -> (DataFrame, Dataset)
    split_train_valid(...)     -> (train_df, valid_df)
    spatial_blocks(df, k)      -> df with a 'spatial_fold' column
    temporal_folds(df, ...)    -> df with a 'temporal_fold' column
    report_coverage(df, path)  -> writes CSV, returns DataFrame
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from src.config import (
    PROCESSED_DIR, DERIVED_DIR, RAW_DIR,
    TARGET_LAYERS, TARGET_WEIGHTS, TARGET_NAME,
    DEFAULT_FEATURES, TRAIN_YEARS, VALID_YEARS,
    SEASONS, CRS, ML_DIR,
)
from src.io import load_stack


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------

def build_rootzone_target() -> xr.DataArray:
    """
    Return SM_ROOTZONE as a thickness-weighted mean of SM_L2, SM_L3, SM_L4.

    The three input layers come from data/processed/ via src.io.load_stack,
    which returns (time, y, x) DataArrays with 'year' and 'season' coords.
    The output DataArray has the same coords and dims.
    """
    accum  = None
    weight = None
    for name in TARGET_LAYERS:
        da = load_stack(name, source='processed')
        w  = TARGET_WEIGHTS[name]
        term = da * w
        accum = term if accum is None else accum + term
        weight = w if weight is None else weight + w

    # weight should be 1.0 by construction; guard anyway
    if not np.isclose(weight, 1.0, atol=1e-9):
        raise RuntimeError(
            f"TARGET_WEIGHTS sum to {weight}, expected 1.0. "
            f"Check src.config.TARGET_WEIGHTS."
        )

    accum = accum.rename(TARGET_NAME)
    accum.attrs = {
        'units':    'm3/m3',
        'method':   'thickness-weighted mean of SM_L2, SM_L3, SM_L4',
        'weights':  json.dumps(TARGET_WEIGHTS),
        'crs':      CRS,
    }
    accum = accum.rio.write_crs(CRS, inplace=False)
    return accum


# ---------------------------------------------------------------------------
# Feature loading and alignment
# ---------------------------------------------------------------------------

def _align_to_target(da: xr.DataArray, target: xr.DataArray,
                     name: str) -> xr.DataArray:
    """
    Align one feature DataArray to the target on (time, y, x).

    Uses left-join on target's time axis. If the feature has fewer time
    steps than the target, the missing steps become NaN and are dropped
    downstream by the NaN filter. If a feature is static (2D), broadcast
    it across time.
    """
    if 'time' not in da.dims:
        # Static feature (e.g. SRTM_ELEVATION). Broadcast to target time.
        da = da.expand_dims(time=target.time)
        da = da.assign_coords(year=('time', target.year.values),
                              season=('time', target.season.values))

    # Align spatial dims: use target's x, y. Interpolate if the feature
    # grid differs. Bilinear is safe for continuous features and only
    # used if the grids actually differ; if identical, this is a no-op.
    if not (np.array_equal(da.x.values, target.x.values)
            and np.array_equal(da.y.values, target.y.values)):
        da = da.interp(x=target.x, y=target.y, method='linear')

    # Align time: preserve target's time order, take matching indices.
    da = da.reindex(time=target.time)

    # Reattach year / season from target so downstream code has one truth.
    da = da.assign_coords(year=('time', target.year.values),
                          season=('time', target.season.values))

    da = da.rename(name)
    return da


def load_feature_stacks(features: Sequence[str],
                        target: xr.DataArray) -> xr.Dataset:
    """
    Load every feature in `features`, align to `target`, return Dataset.
    The target is included in the Dataset under TARGET_NAME.
    """
    data_vars = {TARGET_NAME: target}
    for name in features:
        # Most features come from data/processed/ or data/derived/.
        # load_stack tries processed first, then derived, then raises.
        try:
            da = load_stack(name, source='processed')
        except FileNotFoundError:
            da = load_stack(name, source='derived')
        da = _align_to_target(da, target, name)
        data_vars[name] = da
    return xr.Dataset(data_vars)


# ---------------------------------------------------------------------------
# Flattening to a tidy table
# ---------------------------------------------------------------------------

def flatten_to_dataframe(ds: xr.Dataset,
                         stride: int = 1) -> pd.DataFrame:
    """
    Convert the (time, y, x) Dataset into a tidy DataFrame with one row per
    (time, y, x) sample. `stride` subsamples the grid (stride=2 keeps every
    second pixel in x and y) for memory-bounded experimentation; stride=1
    keeps everything.

    Columns:
        time_idx, year, season, y_idx, x_idx, x_coord, y_coord,
        SM_ROOTZONE, <every feature>
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    # Subset the grid if stride > 1
    if stride > 1:
        ds = ds.isel(y=slice(None, None, stride),
                     x=slice(None, None, stride))

    nt = ds.sizes['time']
    ny = ds.sizes['y']
    nx = ds.sizes['x']

    # time / year / season arrays (length nt)
    year   = ds['year'].values
    season = ds['season'].values
    # Broadcast time metadata to (nt, ny, nx)
    year_grid   = np.broadcast_to(year  [:, None, None], (nt, ny, nx))
    season_grid = np.broadcast_to(season[:, None, None], (nt, ny, nx))
    time_grid   = np.broadcast_to(np.arange(nt)[:, None, None], (nt, ny, nx))

    # Pixel index grids
    y_idx = np.broadcast_to(np.arange(ny)[None, :, None], (nt, ny, nx))
    x_idx = np.broadcast_to(np.arange(nx)[None, None, :], (nt, ny, nx))
    y_coord = np.broadcast_to(ds.y.values[None, :, None], (nt, ny, nx))
    x_coord = np.broadcast_to(ds.x.values[None, None, :], (nt, ny, nx))

    # Base frame
    df = pd.DataFrame({
        'time_idx': time_grid.ravel(),
        'year':     year_grid.ravel(),
        'season':   season_grid.ravel(),
        'y_idx':    y_idx.ravel(),
        'x_idx':    x_idx.ravel(),
        'x_coord':  x_coord.ravel(),
        'y_coord':  y_coord.ravel(),
    })

    # Every data variable becomes a column
    for name in ds.data_vars:
        df[name] = ds[name].values.ravel()

    return df


# ---------------------------------------------------------------------------
# QA and validation-row construction
# ---------------------------------------------------------------------------

def apply_qa(df: pd.DataFrame,
             feature_cols: Sequence[str],
             target_col: str = TARGET_NAME,
             dropna_target: bool = True,
             dropna_features: bool = True) -> pd.DataFrame:
    """
    Drop rows that are not usable for training.

    - Rows with NaN in the target are dropped if dropna_target.
    - Rows with NaN in any feature column are dropped if dropna_features.
    """
    before = len(df)
    if dropna_target:
        df = df[df[target_col].notna()]
    if dropna_features:
        mask = df[list(feature_cols)].notna().all(axis=1)
        df = df[mask]
    df = df.reset_index(drop=True)
    after = len(df)
    print(f"apply_qa: {before:,} -> {after:,} rows "
          f"({100.0 * after / max(before, 1):.2f}% retained)")
    return df


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def split_train_valid(df: pd.DataFrame,
                      train_years: Sequence[int] = TRAIN_YEARS,
                      valid_years: Sequence[int] = VALID_YEARS
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal split: rows whose `year` is in train_years go to train,
    rows in valid_years go to valid. Rows outside both are discarded.
    """
    train_mask = df['year'].isin(train_years)
    valid_mask = df['year'].isin(valid_years)
    train_df = df[train_mask].reset_index(drop=True)
    valid_df = df[valid_mask].reset_index(drop=True)
    print(f"split_train_valid: train={len(train_df):,} rows, "
          f"valid={len(valid_df):,} rows")
    print(f"  train years: {min(train_years)}-{max(train_years)}")
    print(f"  valid years: {min(valid_years)}-{max(valid_years)}")
    return train_df, valid_df


# ---------------------------------------------------------------------------
# Spatial block CV folds
# ---------------------------------------------------------------------------

def spatial_blocks(df: pd.DataFrame, k: int = 5,
                   by: str = 'y_idx') -> pd.DataFrame:
    """
    Assign each pixel to one of k spatial blocks.

    Default: horizontal bands along y_idx (north-south strips). That is
    conservative for an equatorial mountain AOI because it preserves
    the elevational gradient across folds rather than putting the
    summit and the foothills in the same fold.

    For a more general partition, set by='x_idx' for vertical strips,
    or write a custom block function and pass the assignment in.
    """
    if by not in ('x_idx', 'y_idx'):
        raise ValueError(f"by must be 'x_idx' or 'y_idx', got {by}")
    n = df[by].max() + 1
    edges = np.linspace(0, n, k + 1)
    # np.digitize returns 1..k, then -1 to make it 0..k-1
    fold = np.clip(np.digitize(df[by].values, edges[1:-1]), 0, k - 1)
    df = df.copy()
    df['spatial_fold'] = fold.astype('int16')
    return df


def spatial_folds_for_pixels(ny: int, k: int = 5,
                             by: str = 'y_idx') -> np.ndarray:
    """
    Return an (ny, nx) int16 array of fold ids, useful for plotting the
    fold layout. Does not depend on df; only uses grid shape.
    """
    raise NotImplementedError(
        "Pixel-grid fold layout: call spatial_blocks on the dataframe "
        "and pivot back if needed."
    )


# ---------------------------------------------------------------------------
# Temporal CV folds
# ---------------------------------------------------------------------------

def temporal_folds(df: pd.DataFrame,
                   n_folds: int = 5) -> pd.DataFrame:
    """
    Assign each year to one of n_folds temporal folds, contiguous in time.
    A row's 'temporal_fold' is the fold that contains its year.
    """
    years = np.sort(df['year'].unique())
    if len(years) < n_folds:
        raise ValueError(
            f"Cannot make {n_folds} temporal folds from {len(years)} years"
        )
    # Contiguous chunks, as even as possible
    year_fold = {}
    for i, fold_years in enumerate(np.array_split(years, n_folds)):
        for y in fold_years:
            year_fold[int(y)] = i
    df = df.copy()
    df['temporal_fold'] = df['year'].map(year_fold).astype('int16')
    return df


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def report_coverage(df: pd.DataFrame,
                    feature_cols: Sequence[str],
                    target_col: str = TARGET_NAME,
                    out_dir: Path = ML_DIR,
                    tag: str = '') -> pd.DataFrame:
    """
    Write a coverage report of the post-QA training table.

    Returns a DataFrame with per-(year, season) row counts and per-feature
    non-null counts. Also writes the same content to a CSV under out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per (year, season)
    per_ts = (
        df.groupby(['year', 'season'], observed=True)
          .size()
          .reset_index(name='n_rows')
    )
    per_ts = per_ts.sort_values(['year', 'season']).reset_index(drop=True)

    # Per feature: coverage fraction
    per_feat = pd.DataFrame({
        'feature':        list(feature_cols) + [target_col],
        'non_null_rows':  [int(df[c].notna().sum()) for c in feature_cols]
                          + [int(df[target_col].notna().sum())],
        'total_rows':     len(df),
    })
    per_feat['coverage_pct'] = (
        100.0 * per_feat['non_null_rows'] / per_feat['total_rows']
    ).round(2)

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    suffix = f'_{tag}' if tag else ''
    per_ts_path   = out_dir / f'coverage_by_timeslice{suffix}_{timestamp}.csv'
    per_feat_path = out_dir / f'coverage_by_feature{suffix}_{timestamp}.csv'
    per_ts.to_csv(per_ts_path, index=False)
    per_feat.to_csv(per_feat_path, index=False)

    print(f"\ncoverage written:")
    print(f"  {per_ts_path}")
    print(f"  {per_feat_path}")
    print("\nPer-feature coverage:")
    print(per_feat.to_string(index=False))

    return per_feat


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def build_feature_table(features: Optional[Sequence[str]] = None,
                        stride: int = 1,
                        write_coverage: bool = True,
                        coverage_tag: str = '') -> Tuple[pd.DataFrame, xr.Dataset]:
    """
    Build the tidy training table.

    Steps:
        1. Build SM_ROOTZONE from SM_L2, SM_L3, SM_L4.
        2. Load every feature, align to target grid and time.
        3. Flatten to a tidy DataFrame.
        4. Apply QA: drop rows with NaN target, then NaN features.
        5. Optionally write a coverage report.

    Returns:
        df  - post-QA DataFrame with one row per usable sample
        ds  - the aligned (time, y, x) Dataset (pre-flatten), for
              downstream mapping of predictions.
    """
    if features is None:
        features = DEFAULT_FEATURES

    print("Building root-zone target...")
    target = build_rootzone_target()

    print(f"Loading and aligning {len(features)} features...")
    ds = load_feature_stacks(features, target)

    print(f"Flattening to a tidy table (stride={stride})...")
    df = flatten_to_dataframe(ds, stride=stride)
    print(f"  raw rows: {len(df):,}")

    df = apply_qa(df, feature_cols=list(features), target_col=TARGET_NAME)

    if write_coverage:
        report_coverage(df, feature_cols=list(features),
                        target_col=TARGET_NAME, tag=coverage_tag)

    return df, ds