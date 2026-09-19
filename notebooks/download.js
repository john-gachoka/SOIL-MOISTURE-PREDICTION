// ======================================================================
// Landsat (NDVI, LST), ERA-5 (Soil Moisture), SRTM (TWI),
// CHIRPS (Precip), FIRMS (Fire, 2000+)
//
// Seasons: JF, MAM, JJAS, OND
// Naming : <Variable>_<Season>_<Year>   e.g. NDVI_JF_2016, PRECIP_JF_2016
// CRS    : EPSG:21037 (Arc 1960 / UTM zone 37N)
// Nodata : -9999.0 (float32)
// Rule   : If a season has ZERO source scenes, NO file is exported.
// =========================================================================

// ------------------------------ CONFIG -----------------------------------
var startYear = 1995;
var endYear   = 2002;
var FOLDER    = 'Soil_Moisture';
var NODATA    = -9999;

var CRS = 'EPSG:21037';

var LANDSAT_SCALE = 30;
var ERA5_SCALE    = 11000;
var CHIRPS_SCALE  = 5500;
var FIRMS_SCALE   = 1000;
var SRTM_SCALE    = 30;

var QA_MASK_BITS = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4);

var HARMONIZE_OLI = true;
var OLI_NIR_SLOPE = 0.8473, OLI_NIR_INT = 0.0415;
var OLI_RED_SLOPE = 0.9047, OLI_RED_INT = 0.0061;

var MIN_FIRE_CONFIDENCE = 0;
var DEM_BUFFER_M        = 10000;
var COARSE_BUFFER_M     = 15000;

var aoiGeom      = aoi.geometry();
var coarseRegion = aoiGeom.buffer(COARSE_BUFFER_M).bounds();
var demRegion    = aoiGeom.buffer(DEM_BUFFER_M).bounds();

var SEASONS = {
  JF:   {m0: 1,  n: 2},
  MAM:  {m0: 3,  n: 3},
  JJAS: {m0: 6,  n: 4},
  OND:  {m0: 10, n: 3}
};
var SEASON_KEYS = ['JF', 'MAM', 'JJAS', 'OND'];

// ------------------------------ HELPERS ----------------------------------
function windowOf(y, s) {
  var start = ee.Date.fromYMD(y, s.m0, 1);
  return {start: start, end: start.advance(s.n, 'month')};
}

function finalize(img, clipTo) {
  var out = clipTo ? img.clip(clipTo) : img;
  return out.toFloat().unmask(ee.Image.constant(NODATA).toFloat());
}

function exportImage(image, name, region, gridOpts) {
  var params = {
    image: image,
    description: name,
    folder: FOLDER,
    fileNamePrefix: name,
    region: region,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: NODATA}
  };
  for (var k in gridOpts) { params[k] = gridOpts[k]; }
  Export.image.toDrive(params);
}

// =========================================================================
// 1. LANDSAT — NDVI, LST
// =========================================================================
var LANDSAT = [
  {id:'LANDSAT/LT05/C02/T1_L2', nir:'SR_B4', red:'SR_B3', thermal:'ST_B6',  oli:false},
  {id:'LANDSAT/LE07/C02/T1_L2', nir:'SR_B4', red:'SR_B3', thermal:'ST_B6',  oli:false},
  {id:'LANDSAT/LC08/C02/T1_L2', nir:'SR_B5', red:'SR_B4', thermal:'ST_B10', oli:true},
  {id:'LANDSAT/LC09/C02/T1_L2', nir:'SR_B5', red:'SR_B4', thermal:'ST_B10', oli:true}
];

function makeLandsatPrep(cfg) {
  return function(img) {
    var clear = img.select('QA_PIXEL').bitwiseAnd(QA_MASK_BITS).eq(0)
                  .and(img.select('QA_RADSAT').eq(0));

    var nir = img.select(cfg.nir).multiply(0.0000275).add(-0.2);
    var red = img.select(cfg.red).multiply(0.0000275).add(-0.2);
    if (cfg.oli && HARMONIZE_OLI) {
      nir = nir.multiply(OLI_NIR_SLOPE).add(OLI_NIR_INT);
      red = red.multiply(OLI_RED_SLOPE).add(OLI_RED_INT);
    }
    var reflOk = nir.gt(0).and(nir.lt(1)).and(red.gt(0)).and(red.lt(1));
    var ndvi = nir.subtract(red).divide(nir.add(red))
                  .rename('NDVI').updateMask(clear.and(reflOk));

    var lst = img.select(cfg.thermal).multiply(0.00341802).add(149.0);
    var lstOk = lst.gt(250).and(lst.lt(340));
    lst = lst.rename('LST').updateMask(clear.and(lstOk));

    return ndvi.addBands(lst).copyProperties(img, ['system:time_start']);
  };
}

function landsatSeason(y, s) {
  var w = windowOf(y, s);
  var merged = null;
  LANDSAT.forEach(function(cfg) {
    var c = ee.ImageCollection(cfg.id)
      .filterBounds(aoi).filterDate(w.start, w.end)
      .map(makeLandsatPrep(cfg));
    merged = (merged === null) ? c : merged.merge(c);
  });

  var ndvi = merged.select('NDVI').median().rename('NDVI');
  var lst  = merged.select('LST').median().rename('LST');
  var nobsNdvi = merged.select('NDVI').count().rename('NOBS_NDVI');
  var nobsLst  = merged.select('LST').count().rename('NOBS_LST');

  return ndvi.addBands(lst).addBands(nobsNdvi).addBands(nobsLst);
}

// =========================================================================
// 2. ERA-5, 3. CHIRPS, 4. FIRMS — collection handles
// =========================================================================
var era    = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR');
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY');
var firms  = ee.ImageCollection('FIRMS');

function era5Season(y, s) {
  var w = windowOf(y, s);
  var c = era.filterDate(w.start, w.end);
  return c.select([
    'volumetric_soil_water_layer_1',
    'volumetric_soil_water_layer_2',
    'volumetric_soil_water_layer_3',
    'volumetric_soil_water_layer_4'
  ]).mean().rename(['SM_L1','SM_L2','SM_L3','SM_L4']);
}

function chirpsSeason(y, s) {
  var w = windowOf(y, s);
  return chirps.filterDate(w.start, w.end).select('precipitation')
    .sum().rename('PRECIP');
}

function firmsSeason(y, s) {
  var w = windowOf(y, s);
  var c = firms.filterDate(w.start, w.end).map(function(img) {
    return img.select('T21')
      .updateMask(img.select('confidence').gte(MIN_FIRE_CONFIDENCE));
  });
  return c.count().rename('FIRE_DAYS');
}

// =========================================================================
// STAGE 1 — COUNT SCENES PER SEASON, BATCHED BY YEAR (31 getInfo calls)
// =========================================================================
var counts = {};   // plain client-side dict

for (var cy = startYear; cy <= endYear; cy++) {
  var yearJobs = [];

  SEASON_KEYS.forEach(function(sk) {
    var w = windowOf(cy, SEASONS[sk]);

    // Landsat: sum across all four sensors
    var lsTotal = ee.Number(0);
    LANDSAT.forEach(function(cfg) {
      lsTotal = lsTotal.add(
        ee.ImageCollection(cfg.id)
          .filterBounds(aoi).filterDate(w.start, w.end).size()
      );
    });
    yearJobs.push(['landsat|' + sk, lsTotal]);

    // ERA5, CHIRPS
    yearJobs.push(['era5|'   + sk, era.filterDate(w.start, w.end).size()]);
    yearJobs.push(['chirps|' + sk, chirps.filterDate(w.start, w.end).size()]);

    // FIRMS only from 2001 onwards
    if (cy >= 2001) {
      yearJobs.push(['firms|' + sk, firms.filterDate(w.start, w.end).size()]);
    }
  });

  var yearCounts = ee.Dictionary(ee.List(yearJobs).flatten()).getInfo();
  Object.keys(yearCounts).forEach(function(k) {
    counts[k + '|' + cy] = yearCounts[k];
  });

  print('Counted ' + cy + ' (' + Object.keys(yearCounts).length + ' entries)');
}

// =========================================================================
// STAGE 2 — QUEUE EXPORTS ONLY WHERE COUNT > 0
// =========================================================================
SEASON_KEYS.forEach(function(sk) {
  for (var y = startYear; y <= endYear; y++) {

    // ---- Landsat ----
    if (counts['landsat|' + sk + '|' + y] > 0) {
      var limg = finalize(landsatSeason(y, SEASONS[sk]), aoiGeom);
      ['NDVI','LST','NOBS_NDVI','NOBS_LST'].forEach(function(v) {
        exportImage(
          limg.select(v).rename(v),
          v + '_' + sk + '_' + y,
          aoiGeom,
          {crs: CRS, scale: LANDSAT_SCALE}
        );
      });
    } else {
      print('SKIP Landsat ' + sk + '_' + y);
    }

    // ---- ERA5 ----
    if (counts['era5|' + sk + '|' + y] > 0) {
      var eimg = finalize(era5Season(y, SEASONS[sk]), null);
      ['SM_L1','SM_L2','SM_L3','SM_L4'].forEach(function(v) {
        exportImage(
          eimg.select(v).rename(v),
          v + '_' + sk + '_' + y,
          coarseRegion,
          {crs: CRS, scale: ERA5_SCALE}
        );
      });
    } else {
      print('SKIP ERA5 ' + sk + '_' + y);
    }

    // ---- CHIRPS ----
    if (counts['chirps|' + sk + '|' + y] > 0) {
      exportImage(
        finalize(chirpsSeason(y, SEASONS[sk]), null),
        'PRECIP_' + sk + '_' + y,
        coarseRegion,
        {crs: CRS, scale: CHIRPS_SCALE}
      );
    } else {
      print('SKIP CHIRPS ' + sk + '_' + y);
    }

    // ---- FIRMS ----
    var firmsKey = 'firms|' + sk + '|' + y;
    if (counts[firmsKey] > 0) {
      exportImage(
        finalize(firmsSeason(y, SEASONS[sk]), null),
        'FIRE_DAYS_' + sk + '_' + y,
        coarseRegion,
        {crs: CRS, scale: FIRMS_SCALE}
      );
    } else if (y >= 2001) {
      print('SKIP FIRMS ' + sk + '_' + y);
    }
  }
});

// =========================================================================
// STAGE 3 — STATIC SRTM DEM
// =========================================================================
exportImage(
  ee.Image('USGS/SRTMGL1_003').select('elevation').toFloat(),
  'SRTM_ELEVATION',
  demRegion,
  {crs: CRS, scale: SRTM_SCALE}
);

print('Done. Check Tasks tab. Only seasons with source data were queued.');