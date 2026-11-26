# src/train.py

"""
Batch entry point for Objective 2 model training.

Reads a YAML config, builds the feature table, splits train/validation,
runs cross-validation, fits the final model, computes SHAP, writes
predictions, and logs a run record.

Usage:
    python -m src.train --config src/configs/rf_baseline.yaml
    python -m src.train --config src/configs/xgb_tuned.yaml --run-id my_test
    python -m src.train --config src/configs/rf_baseline.yaml --dry-run

Designed to be called from models.sh via Slurm, but runs fine interactively.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.config import (
    DEFAULT_FEATURES, TRAIN_YEARS, VALID_YEARS,
    DEFAULT_SHAP_SAMPLE, DEFAULT_SPATIAL_K,
    RUN_LOG_DIR, SEASONS,
)
from src.features import (
    build_feature_table, split_train_valid,
    spatial_blocks, temporal_folds, report_coverage,
)
from src.models import run_training


def load_config(path: Path) -> dict:
    """Load a YAML config file, return dict."""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} did not parse to a dict.")
    return cfg


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an ML model for soil moisture prediction (Obj 2)."
    )
    parser.add_argument('--config', required=True, type=Path,
                        help="Path to YAML config.")
    parser.add_argument('--run-id', default=None, type=str,
                        help="Run identifier. Defaults to "
                             "<model>_<YYYYMMDD_HHMMSS>.")
    parser.add_argument('--stride', default=1, type=int,
                        help="Grid subsampling stride for the feature "
                             "table. stride=1 uses every pixel. "
                             "stride=2 uses every second pixel in x and y "
                             "(4x fewer rows).")
    parser.add_argument('--dry-run', action='store_true',
                        help="Build the table, print coverage, do not fit.")
    parser.add_argument('--no-shap', action='store_true',
                        help="Skip SHAP computation.")
    parser.add_argument('--override', action='append', default=[],
                        metavar='KEY=VALUE',
                        help="Override a config key, e.g. "
                             "--override spatial_k=3. "
                             "Values are parsed with yaml.safe_load.")
    return parser.parse_args(argv)


def apply_overrides(cfg: dict, overrides: list) -> dict:
    """Apply --override KEY=VALUE pairs to cfg in place."""
    for item in overrides:
        if '=' not in item:
            raise ValueError(f"--override expects KEY=VALUE, got {item!r}")
        k, v = item.split('=', 1)
        parsed = yaml.safe_load(v)
        cfg[k.strip()] = parsed
        print(f"Override: {k.strip()} = {parsed!r}")
    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)
    print(f"[{datetime.utcnow().isoformat()}Z] src.train starting")
    print(f"  config:  {args.config}")
    print(f"  cwd:     {os.getcwd()}")
    print(f"  python:  {sys.version.split()[0]}")

    if not args.config.exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 2
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    # --- Merge CLI onto cfg ---
    if args.run_id:
        cfg['run_id'] = args.run_id
    if args.no_shap:
        cfg['compute_shap'] = False
    cfg.setdefault('feature_cols', DEFAULT_FEATURES)
    cfg.setdefault('spatial_k', DEFAULT_SPATIAL_K)
    cfg.setdefault('shap_sample', DEFAULT_SHAP_SAMPLE)

    # --- Build feature table ---
    print(f"\nBuilding feature table (stride={args.stride})...")
    df, ds = build_feature_table(
        features=cfg['feature_cols'],
        stride=args.stride,
        write_coverage=True,
        coverage_tag=cfg.get('run_id', 'pre_run'),
    )
    print(f"  total usable rows: {len(df):,}")
    if len(df) == 0:
        print("ERROR: no usable rows after QA. Check coverage report.",
              file=sys.stderr)
        return 3

    if args.dry_run:
        print("\nDry run: coverage complete, exiting without fitting.")
        return 0

    # --- Split train / valid by year ---
    train_years = cfg.get('train_years', TRAIN_YEARS)
    valid_years = cfg.get('valid_years', VALID_YEARS)
    df_train, df_valid = split_train_valid(
        df, train_years=train_years, valid_years=valid_years,
    )

    if len(df_train) == 0:
        print("ERROR: zero training rows after temporal split.",
              file=sys.stderr)
        return 4

    # --- Fit + CV + SHAP + predict ---
    try:
        result = run_training(
            cfg=cfg,
            df_train=df_train,
            df_valid=df_valid,
            ds_full=ds,
        )
    except Exception:
        print("\nERROR during training:", file=sys.stderr)
        traceback.print_exc()
        # Write a failure marker so the run is traceable in the logs
        run_id = cfg.get('run_id') or 'unknown'
        fail_path = RUN_LOG_DIR / f'{run_id}_FAILED.json'
        with open(fail_path, 'w') as f:
            f.write(f'{{"error": "training failed", '
                    f'"run_id": "{run_id}", '
                    f'"timestamp": "{datetime.utcnow().isoformat()}Z"}}')
        print(f"Failure marker: {fail_path}", file=sys.stderr)
        return 5

    print(f"\n[{datetime.utcnow().isoformat()}Z] src.train finished OK")
    print(f"  run_id: {result['run_id']}")
    print(f"  model:  {result['model_path']}")
    print(f"  cv:     {result['cv_metrics_path']}")
    print(f"  preds:  {result['pred_path']}")
    if result.get('shap_path'):
        print(f"  shap:   {result['shap_path']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())