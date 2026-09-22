// Load ESA WorldCover v200 (2021)
var worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select('Map');

// Extract Tree Cover (class 10), mask everything else
var forestMask = worldcover.eq(10).rename('FOREST_MASK');

// Export to Drive (single file, 30 m to match your master grid)
Export.image.toDrive({
  image: forestMask.toFloat(),
  description: 'FOREST_MASK_WorldCover',
  folder: 'Soil_Moisture',
  fileNamePrefix: 'FOREST_MASK',
  region: aoi.geometry().bounds(),
  scale: 30,
  crs: 'EPSG:21037',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF',
  formatOptions: {
    cloudOptimized: true,
    noData: -9999
  }
});