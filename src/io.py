"""
I/O helpers: load NetCDF stacks, GeoTIFFs, pickles.

All paths are resolved through src.config so the same code works
on any machine.
"""

import pickle
from pathlib import Path
from typing import Union, Optional

import xarray as xr
import rioxarray as rxr

from src.config import PROCESSED_DIR, DERIVED_DIR, CRS, NODATA


# ---------------- NetCDF stack loading ----------------

def load_stack(name: str,
               source: str = 'processed',
               var: Optional[str] = None) -> xr.DataArray:
    """
    Load a NetCDF stack as a DataArray.

    Parameters
    ----------
    name : str
        Base name, e.g. 'NDVI', 'FVC', 'SM_L2'.
    source : {'processed', 'derived'}
        Which directory to look in.
    var : str, optional
        Data variable inside the file. If None, picks the first
        non-metadata data variable automatically.

    Returns
    -------
    xr.DataArray with dims (time, y, x) or (y, x) for static layers.
    """
    base = PROCESSED_DIR if source == 'processed' else DERIVED_DIR

    # Try QA-masked version first, fall back to plain
    candidates = [
        base / f'{name}_stack_qa.nc',
        base / f'{name}_stack.nc',
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"No stack found for '{name}' in {base}. "
            f"Tried: {[p.name for p in candidates]}"
        )

    ds = xr.open_dataset(path)

    # If var not specified, pick the first non-metadata data variable
    if var is None:
        meta = {'spatial_ref', 'band', 'crs'}
        data_vars = [k for k in ds.data_vars if k not in meta]
        if not data_vars:
            raise RuntimeError(f"No data variables in {path.name}")
        var = data_vars[0]

    if var not in ds.data_vars:
        raise KeyError(
            f"Variable '{var}' not in {path.name}. "
            f"Available: {list(ds.data_vars)}"
        )

    return ds[var]


# ---------------- GeoTIFF loading ----------------

def load_tif(path: Union[str, Path]) -> xr.DataArray:
    """
    Load a GeoTIFF as a 2D DataArray (squeezes singleton dims).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    da = rxr.open_rasterio(path, masked=True).squeeze()
    return da


# ---------------- Pickle helpers ----------------

def save_pickle(obj, path: Union[str, Path]) -> None:
    """Save any Python object to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Union[str, Path]):
    """Load a pickled Python object."""
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------- Raster writing ----------------

def write_tif(da: xr.DataArray,
              path: Union[str, Path],
              nodata: float = NODATA,
              dtype: str = 'float32') -> None:
    """
    Write a 2D DataArray to a Cloud-Optimized GeoTIFF.
    Strips conflicting attrs before writing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Clean attrs / encoding that conflict with rioxarray
    da = da.copy()
    for k in ('_FillValue', 'missing_value'):
        da.attrs.pop(k, None)
        da.encoding.pop(k, None)

    da = da.rio.write_crs(CRS, inplace=False)
    da = da.rio.write_nodata(nodata, inplace=False)

    da.rio.to_raster(
        path,
        dtype=dtype,
        compress='LZW',
        tiled=True,
        nodata=nodata,
    )