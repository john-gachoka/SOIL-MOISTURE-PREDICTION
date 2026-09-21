# Integrated Assessment of Subsurface Soil Moisture Response to Forest Density and Climate Elasticity

**Case study:** Mount Kenya Forest Ecosystem
**Author:** Gachoka John 
**Institution:** Institute of Geomatics, GIS and Remote Sensing
**Scope:** 1995–2025

---

## Overview

This project quantifies how subsurface soil moisture responds to changes in
forest density and climate variability across the Mount Kenya ecosystem,
using Earth Observation data and machine learning. It produces
high-resolution (30 m) seasonal soil moisture information to support
catchment management and forest restoration planning.

Mount Kenya is a critical water tower: it supplies over 75% of Kenya's
freshwater and supports more than 2 million people through the Upper Tana
and Ewaso Ng'iro river basins. Forest degradation from agriculture, logging,
and land-use change is altering infiltration, runoff, and soil moisture
dynamics, threatening long-term water security. Coarse-resolution soil
moisture products (e.g. ERA5-Land at ~11 km) fail to capture the fine-scale
spatial variability needed for local decision-making.

---

## Objectives

**Overall:** Assess the response of subsurface soil moisture to forest
density and climate elasticity using Earth Observation and machine learning.

**Specific:**

1. Quantify and characterize spatio-temporal changes in forest density using
   multi-temporal Earth Observation imagery (Mann-Kendall + Sen's slope).
2. Model and validate a machine learning model for predicting subsurface
   soil moisture from EO-derived forest density and climate elasticity
   variables.
3. Map subsurface soil moisture vulnerability hotspots to identify zones
   highly susceptible to dry-season water deficit.

---

## Study Area

Mount Kenya ecosystem, spanning the Upper Tana and Ewaso Ng'iro catchments.
AOI asset: `projects/rbfn1-506017/assets/mkfe`.

The AOI straddles the equator (roughly 0.1°N to 0.3°S, 37.1°E to 37.7°E)
and is processed in **EPSG:21037** (Arc 1960 / UTM zone 37N) at 30 m.

---

## Data Sources

| Variable | Source | Native resolution | Temporal coverage |
| :--- | :--- | :--- | :--- |
| NDVI, LST | Landsat 5/7/8/9 (C02 T1 L2) | 30 m | 1984–present |
| Soil moisture (0–289 cm, 4 layers) | ERA5-Land Monthly Aggregated | ~11 km | 1950–present |
| Air temperature (T2M, TMAX, TMIN) | ERA5-Land Daily Aggregated | ~11 km | 1950–present |
| Precipitation | CHIRPS Daily | ~5.5 km | 1981–present |
| Active fire detections | NASA FIRMS | 1 km | 2000-11–present |
| Elevation | SRTM GL1 | 30 m | static (2000) |

**Temporal aggregation:** JF (Jan–Feb), MAM (Mar–May), JJAS (Jun–Sep),
OND (Oct–Dec). This matches the Kenya Meteorological Department seasonal
convention and separates the two dry seasons (JF, JJAS) from the two wet
seasons (MAM, OND).

**Gap handling:** seasons with zero source scenes are not exported.
Per-pixel observation counts are exported as `NOBS_NDVI` / `NOBS_LST` so
thin seasons can be filtered downstream.

---

## Methodology

### 1. Data acquisition (GEE)

All source data is exported from Google Earth Engine at native resolution.
Exports are seasonal composites:

- **Landsat:** per-pixel median of cloud/shadow/saturation-masked scenes.
  Landsat 8/9 OLI bands are harmonized to ETM+ equivalents using
  Roy et al. (2016) coefficients.
- **ERA5-Land:** seasonal mean of monthly means (soil moisture, T2M).
- **CHIRPS:** seasonal sum of daily precipitation (mm).
- **FIRMS:** count of fire-detection days per pixel per season.
- **SRTM:** static DEM export with 10 km buffer (for TWI flow accumulation).

All exports are float32 with nodata `-9999`, in EPSG:21037, COG-compressed.

### 2. Preprocessing (Notebook 01)

- Align all sources to a common 30 m master grid (snapped to Landsat).
- Resample coarse sources (ERA5, CHIRPS, FIRMS) to master grid:
  - Bilinear for continuous variables (soil moisture, precipitation, temperature)
  - Nearest-neighbour for count data (fire days)
- Apply QA: `NOBS_NDVI >= 2`, `NOBS_LST >= 2`.
- Convert `-9999` to `NaN`.
- Save per-variable NetCDF stacks with dimensions `(time, y, x)`.

### 3. Derived products (Notebook 02)

- **PET:** Hargreaves equation
  `PET = 0.0023 × Ra × (Tmean + 17.8) × √(Tmax − Tmin) × n_days`
- **SPEI:** at 3, 6, and 12-month scales, fitted with the log-logistic
  distribution (Vicente-Serrano et al. 2010), on the P − PET water balance.
- **FVC:** dimidiate pixel model with per-season NDVI endmembers
  (5th and 95th percentiles).
- **Forest mask:** ESA WorldCover 2021, tree-cover class, resampled to 30 m.
- **Forest density:** FVC restricted to the forest mask.

### 4. Trend and causality analysis (Notebook 03)

- **Mann-Kendall + Sen's slope:** per-pixel, per-season trends in NDVI,
  LST, FVC, forest density, SPEI, and soil moisture.
- **Zonal aggregation:** sub-catchments and elevation bands.
- **Granger causality:** on zonal seasonal series, testing whether forest
  density Granger-causes soil moisture, and vice versa.

### 5. Machine learning (Notebook 04 — planned)

- **Target:** SMAP L4 root-zone soil moisture (2015–2025).
- **Predictors:** NDVI, LST, FVC, forest density, PET, SPEI (3/6/12),
  precipitation, TWI, soil texture (SoilGrids), fire frequency.
- **Model:** Random Forest / XGBoost with spatial and temporal
  cross-validation.
- **Interpretation:** SHAP feature importance.
- **Output:** hindcast soil moisture 1995–2014 and vulnerability
  hotspot map.

---

