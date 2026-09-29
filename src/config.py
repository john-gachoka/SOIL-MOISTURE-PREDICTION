"""
Project-wide configuration: paths, constants, variable metadata.

Auto-detects REPO_ROOT from this file's location, so the same code works
on both local machine and HPC without edits.
"""

from pathlib import Path

# ---------------- Paths (auto-detected) ----------------
REPO_ROOT     = Path(__file__).resolve().parent.parent

DATA_DIR      = REPO_ROOT / 'data'
RAW_DIR       = DATA_DIR / 'obj1'
PROCESSED_DIR = DATA_DIR / 'processed'
DERIVED_DIR   = DATA_DIR / 'derived'
TIFS_DIR      = DATA_DIR / 'derived_tifs'
TRENDS_DIR    = DATA_DIR / 'trends'

MODELS_DIR    = REPO_ROOT / 'models'
OUTPUTS_DIR   = REPO_ROOT / 'outputs'
FIG_DIR       = OUTPUTS_DIR / 'figures'
LOG_DIR       = REPO_ROOT / 'notebooks' / 'logs'
CONFIGS_DIR   = REPO_ROOT / 'src' / 'configs'

# Ensure output directories exist
for _p in [MODELS_DIR, OUTPUTS_DIR, FIG_DIR, LOG_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

# ---------------- Temporal constants ----------------
SEASONS        = ['JF', 'MAM', 'JJAS', 'OND']
YEARS          = list(range(1995, 2026))
SNAPSHOT_YEARS = [1995, 2001, 2007, 2013, 2019, 2025]
SEASON_MONTHS  = {'JF': 2, 'MAM': 3, 'JJAS': 4, 'OND': 3}
TOTAL_MONTHS   = sum(SEASON_MONTHS.values())

# ---------------- Spatial constants ----------------
CRS    = 'EPSG:21037'
NODATA = -9999

# ---------------- QA / analysis thresholds ----------------
NOBS_MIN  = 2       # minimum clear Landsat observations per pixel-season
ALPHA     = 0.05    # significance threshold for MK and Granger
N_MIN_MK  = 20      # minimum valid seasons required to run Mann-Kendall

# ---------------- Variable metadata ----------------
VARIABLES = {
    'NDVI':           {'units': 'unitless', 'display': 'NDVI',            'source': 'Landsat'},
    'LST':            {'units': 'K',        'display': 'LST',             'source': 'Landsat'},
    'FVC':            {'units': '0-1',      'display': 'FVC',             'source': 'derived'},
    'FOREST_DENSITY': {'units': '0-1',      'display': 'Forest density',  'source': 'derived'},
    'SM_L1':          {'units': 'm³/m³',    'display': 'SM (0–7 cm)',     'source': 'ERA5'},
    'SM_L2':          {'units': 'm³/m³',    'display': 'SM (7–28 cm)',    'source': 'ERA5'},
    'SM_L3':          {'units': 'm³/m³',    'display': 'SM (28–100 cm)',  'source': 'ERA5'},
    'SM_L4':          {'units': 'm³/m³',    'display': 'SM (100–289 cm)', 'source': 'ERA5'},
    'T2M':            {'units': '°C',       'display': 'T2M',             'source': 'ERA5'},
    'T2M_MAX':        {'units': '°C',       'display': 'T2M max',         'source': 'ERA5'},
    'T2M_MIN':        {'units': '°C',       'display': 'T2M min',         'source': 'ERA5'},
    'PRECIP':         {'units': 'mm',       'display': 'Precipitation',   'source': 'CHIRPS'},
    'PET':            {'units': 'mm',       'display': 'PET',             'source': 'derived'},
    'D':              {'units': 'mm',       'display': 'Water balance',   'source': 'derived'},
    'SPEI_3':         {'units': 'std',      'display': 'SPEI-3',          'source': 'derived'},
    'SPEI_6':         {'units': 'std',      'display': 'SPEI-6',          'source': 'derived'},
    'SPEI_12':        {'units': 'std',      'display': 'SPEI-12',         'source': 'derived'},
    'FIRE_DAYS':      {'units': 'days',     'display': 'Fire days',       'source': 'FIRMS'},
    'SRTM_ELEVATION': {'units': 'm',        'display': 'Elevation',       'source': 'SRTM'},
}

# ---------------- ML defaults (overridable via YAML) ----------------
DEFAULT_TARGET    = 'SM_L2'
DEFAULT_FEATURES  = [
    'NDVI', 'LST', 'FVC', 'FOREST_DENSITY',
    'SM_L1', 'SM_L3', 'SM_L4',
    'PRECIP', 'PET', 'D',
    'SPEI_3', 'SPEI_6', 'SPEI_12',
    'FIRE_DAYS', 'SRTM_ELEVATION',
]