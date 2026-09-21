// =========================================================================
// 2b. ERA-5 — TEMPERATURE (T2M, T2M_MAX, T2M_MIN) from DAILY_AGGR
// Needed for Hargreaves PET in Python. DAILY_AGGR is used because
// MONTHLY_AGGR does not expose Tmax/Tmin.
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

// Reuse the same counting mechanism as ERA5 soil moisture
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
        {crs: CRS, scale: 11000}
      );
    });
  }
});