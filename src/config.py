# src/config.py

"""
Project-wide configuration: paths, constants, variable metadata.

Auto-detects REPO_ROOT from this file's location, so the same code works
on both local machine and HPC without edits.

Obj-2 revision:
    - Adds ML output directories (ML_DIR, RUN_LOG_DIR, PRED_DIR).
    - Adds the root-zone target definition (TARGET_LAYERS, TARGET_WEIGHTS,
      TARGET_NAME) used by src.features and src.models.
    - Replaces DEFAULT_FEATURES with the leakage-free set agreed for
      Objective 2: climate + vegetation + terrain only. No SM layers.
    - Drops DEFAULT_TARGET, which is superseded by TARGET_NAME.
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
PRED_DIR      = DATA_DIR / 'predictions'

MODELS_DIR    = REPO_ROOT / 'models'
OUTPUTS_DIR   = REPO_ROOT / 'outputs'
FIG_DIR       = OUTPUTS_DIR / 'figures'
ML_DIR        = OUTPUTS_DIR / 'metrics'
RUN_LOG_DIR   = OUTPUTS_DIR / 'logs'
LOG_DIR       = REPO_ROOT / 'notebooks' / 'logs'
CONFIGS_DIR   = REPO_ROOT / 'src' / 'configs'

# Ensure output directories exist on import.
for _p in [MODELS_DIR, OUTPUTS_DIR, FIG_DIR, ML_DIR, RUN_LOG_DIR,
           PRED_DIR, LOG_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

# ---------------- Temporal constants ----------------
SEASONS        = ['JF', 'MAM', 'JJAS', 'OND']
YEARS          = list(range(1995, 2026))
SNAPSHOT_YEARS = [1995, 2001, 2007, 2013, 2019, 2025]
SEASON_MONTHS  = {'JF': 2, 'MAM': 3, 'JJAS': 4, 'OND': 3}
TOTAL_MONTHS   = sum(SEASON_MONTHS.values())

# Train / validate split for Objective 2 (hindcast design).
TRAIN_YEARS = list(range(1995, 2015))   # 1995-2014 inclusive
VALID_YEARS = list(range(2015, 2026))   # 2015-2025 inclusive

# ---------------- Spatial constants ----------------
CRS    = 'EPSG:21037'
NODATA = -9999

# ---------------- QA / analysis thresholds ----------------
NOBS_MIN  = 2       # minimum clear Landsat observations per pixel-season
ALPHA     = 0.05    # significance threshold for MK and Granger
N_MIN_MK  = 20      # minimum valid seasons required to run Mann-Kendall

# ---------------- Root-zone target definition ----------------
# ERA5-Land volumetric soil water layer thicknesses (cm):
#   layer 1: 0-7 cm      thickness 7
#   layer 2: 7-28 cm     thickness 21
#   layer 3: 28-100 cm   thickness 72
#   layer 4: 100-289 cm  thickness 189
# Layer 1 is surface, not subsurface; exclude it from the target.
TARGET_LAYERS  = ['SM_L2', 'SM_L3', 'SM_L4']
TARGET_WEIGHTS = {'SM_L2': 21.0 / 282.0,
                  'SM_L3': 72.0 / 282.0,
                  'SM_L4': 189.0 / 282.0}
TARGET_NAME    = 'SM_ROOTZONE'

# ---------------- ML feature set (Objective 2) ----------------
# Climate + vegetation + terrain only. No SM layers (leakage).
DEFAULT_FEATURES = [
    'NDVI',
    'LST',
    'FVC',
    'FOREST_DENSITY',
    'PRECIP',
    'PET',
    'D',
    'SPEI_3',
    'SPEI_6',
    'SPEI_12',
    'T2M',
    'T2M_MAX',
    'T2M_MIN',
    'FIRE_DAYS',
    'SRTM_ELEVATION',
]

# ---------------- ML defaults (overridable via YAML) ----------------
DEFAULT_SEED        = 42
DEFAULT_SPATIAL_K   = 5
DEFAULT_TEST_FRAC   = 0.30   # only used by the random-split baseline
DEFAULT_SHAP_SAMPLE = 20000  # rows sampled for SHAP to keep memory bounded

# ---------------- Variable metadata ----------------
VARIABLES = {
    'NDVI':           {'units': 'unitless', 'display': 'NDVI',            'source': 'Landsat'},
    'LST':            {'units': 'K',        'display': 'LST',             'source': 'Landsat'},
    'FVC':            {'units': '0-1',      'display': 'FVC',             'source': 'derived'},
    'FOREST_DENSITY': {'units': '0-1',      'display': 'Forest density',  'source': 'derived'},
    'SM_L1':          {'units': 'm3/m3',    'display': 'SM (0-7 cm)',     'source': 'ERA5'},
    'SM_L2':          {'units': 'm3/m3',    'display': 'SM (7-28 cm)',    'source': 'ERA5'},
    'SM_L3':          {'units': 'm3/m3',    'display': 'SM (28-100 cm)',  'source': 'ERA5'},
    'SM_L4':          {'units': 'm3/m3',    'display': 'SM (100-289 cm)', 'source': 'ERA5'},
    'T2M':            {'units': 'degC',     'display': 'T2M',             'source': 'ERA5'},
    'T2M_MAX':        {'units': 'degC',     'display': 'T2M max',         'source': 'ERA5'},
    'T2M_MIN':        {'units': 'degC',     'display': 'T2M min',         'source': 'ERA5'},
    'PRECIP':         {'units': 'mm',       'display': 'Precipitation',   'source': 'CHIRPS'},
    'PET':            {'units': 'mm',       'display': 'PET',             'source': 'derived'},
    'D':              {'units': 'mm',       'display': 'Water balance',   'source': 'derived'},
    'SPEI_3':         {'units': 'std',      'display': 'SPEI-3',          'source': 'derived'},
    'SPEI_6':         {'units': 'std',      'display': 'SPEI-6',          'source': 'derived'},
    'SPEI_12':        {'units': 'std',      'display': 'SPEI-12',         'source': 'derived'},
    'FIRE_DAYS':      {'units': 'days',     'display': 'Fire days',       'source': 'FIRMS'},
    'SRTM_ELEVATION': {'units': 'm',        'display': 'Elevation',       'source': 'SRTM'},
    'SM_ROOTZONE':    {'units': 'm3/m3',    'display': 'Root-zone SM',    'source': 'derived'},
}