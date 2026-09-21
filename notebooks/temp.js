// =========================================================================
// ERA-5 — TEMPERATURE (T2M, T2M_MAX, T2M_MIN) from DAILY_AGGR
// Runs from 1995 to 2025. Saves in the same 'Soil_Moisture' folder.
// Naming : <Variable>_<Season>_<Year>   e.g. T2M_JF_2016
// CRS    : EPSG:21037 (Arc 1960 / UTM zone 37N)
// Nodata : -9999.0 (float32)
// Rule   : If a season has ZERO source scenes, NO file is exported.
// =========================================================================

// ------------------------------ CONFIG -----------------------------------
var startYear = 1995; 
var endYear   = 2025;
var FOLDER    = 'Soil_Moisture'; // Kept exact same folder
var NODATA    = -9999;

var CRS = 'EPSG:21037';
var ERA5_SCALE = 11000;
var COARSE_BUFFER_M = 15000;

var aoiGeom      = aoi.geometry();
var coarseRegion = aoiGeom.buffer(COARSE_BUFFER_M).bounds();

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
// DATA HANDLING
// =========================================================================
var eraDaily = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR');

function era5TempSeason(y, s) {
  var w = windowOf(y, s);
  var c = eraDaily.filterDate(w.start, w.end).select([
    'temperature_2m',
    'temperature_2m_max',
    'temperature_2m_min'
  ]);

  // Seasonal mean of daily means, daily maxes, daily mins.
  var meanK = c.select('temperature_2m').mean().rename('T2M');
  var maxK  = c.select('temperature_2m_max').mean().rename('T2M_MAX');
  var minK  = c.select('temperature_2m_min').mean().rename('T2M_MIN');

  // Convert Kelvin → Celsius in the export (cleaner for Python).
  var toC = function(img) { return img.subtract(273.15); };

  return toC(meanK).addBands(toC(maxK)).addBands(toC(minK));
}

// =========================================================================
// PIPELINE EXECUTION
// =========================================================================
SEASON_KEYS.forEach(function(sk) {
  for (var y = startYear; y <= endYear; y++) {
    var w = windowOf(y, SEASONS[sk]);
    var n = eraDaily.filterDate(w.start, w.end).size().getInfo();

    if (n === 0) {
      print('SKIP ERA5-TEMP ' + sk + '_' + y + ' — no days');
      continue;
    }

    var img = finalize(era5TempSeason(y, SEASONS[sk]), null);
    ['T2M', 'T2M_MAX', 'T2M_MIN'].forEach(function(v) {
      exportImage(
        img.select(v).rename(v),
        v + '_' + sk + '_' + y,
        coarseRegion,
        {crs: CRS, scale: ERA5_SCALE}
      );
    });
  }
});

print('Done. Check Tasks tab for ERA5 daily temperature exports.');
