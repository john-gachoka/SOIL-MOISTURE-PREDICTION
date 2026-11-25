# src/plots.py

"""
Reusable plotting functions for maps, time series, SHAP summaries, and
diagnostic plots.

All figures save to src.config.FIG_DIR unless a path is given.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import shap

from src.config import FIG_DIR, CRS


def _save(fig, name: str, out_dir: Optional[Path] = None,
          dpi: int = 150, tight: bool = True) -> Path:
    """Internal helper: save and return the path."""
    out_dir = Path(out_dir) if out_dir is not None else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    return path


def plot_map(da: xr.DataArray,
             cmap: str = 'viridis',
             vmin: Optional[float] = None,
             vmax: Optional[float] = None,
             title: str = '',
             filename: str = 'map.png',
             aoi=None,
             colorbar_label: str = '') -> Path:
    """
    Plot a 2D DataArray as a map with optional AOI boundary overlay.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = da.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                 add_colorbar=False)
    if aoi is not None:
        aoi.boundary.plot(ax=ax, color='black', linewidth=1)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    path = _save(fig, filename)
    plt.close(fig)
    return path


def plot_timeseries(years: np.ndarray,
                    values: np.ndarray,
                    slope: Optional[float] = None,
                    intercept: Optional[float] = None,
                    title: str = '',
                    ylabel: str = '',
                    filename: str = 'ts.png',
                    color: str = '#1f77b4') -> Path:
    """
    Plot a time series with optional Sen's slope trend line.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(years, values, '-o', color=color, lw=1.5, markersize=4,
            label='Observed')
    if slope is not None and intercept is not None:
        trend = slope * years + intercept
        ax.plot(years, trend, '--', color='black', lw=2,
                label=f'Trend: {slope:.3g}/yr')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Year')
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    path = _save(fig, filename)
    plt.close(fig)
    return path


def plot_heatmap(pivot: pd.DataFrame,
                 cmap: str = 'RdBu_r',
                 vmin: Optional[float] = None,
                 vmax: Optional[float] = None,
                 annotate: bool = False,
                 title: str = '',
                 filename: str = 'heatmap.png') -> Path:
    """
    Plot a pivot table (e.g. variable x year) as a heatmap.
    """
    fig, ax = plt.subplots(figsize=(12, max(4, 0.4 * len(pivot))))
    im = ax.imshow(pivot.values, aspect='auto', cmap=cmap,
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    if annotate:
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=7)
    ax.set_title(title, fontsize=13)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = _save(fig, filename)
    plt.close(fig)
    return path


def plot_panel(list_of_da: Sequence[xr.DataArray],
               nrows: int,
               ncols: int,
               titles: Optional[Sequence[str]] = None,
               cmap: str = 'viridis',
               filename: str = 'panel.png') -> Path:
    """
    Plot a grid of maps (one per DataArray) in a single figure.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, da in enumerate(list_of_da):
        if i >= len(axes):
            break
        da.plot(ax=axes[i], cmap=cmap, add_colorbar=False)
        if titles is not None and i < len(titles):
            axes[i].set_title(titles[i], fontsize=11)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_aspect('equal')
    for ax in axes[len(list_of_da):]:
        ax.axis('off')
    path = _save(fig, filename)
    plt.close(fig)
    return path


def plot_shap_summary(shap_values: np.ndarray,
                      X: np.ndarray,
                      feature_names: Sequence[str],
                      max_display: int = 20,
                      filename: str = 'shap_summary.png') -> Path:
    """
    Beeswarm-style SHAP summary plot.
    """
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=list(feature_names),
                      max_display=max_display, show=False)
    path = _save(fig, filename)
    plt.close(fig)
    return path


def plot_residual_map(pred: xr.DataArray,
                      obs: xr.DataArray,
                      title: str = 'Residuals (pred - obs)',
                      filename: str = 'residual_map.png') -> Path:
    """
    Plot residuals on the target grid.
    """
    residual = (pred - obs).mean(dim='time')
    return plot_map(residual, cmap='RdBu_r', title=title,
                    filename=filename, colorbar_label='m3/m3')