let map = null;
let currentTileLayer = null;

let stormMarker = null;
let trackPolyline = null;
let forecastMarkers = [];

const TILE_LIGHT =
  "https://tile.openstreetmap.org/{z}/{x}/{y}.png";

const OSM_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">' +
  "OpenStreetMap contributors</a>";

// ============================================================
// SINGLE SOURCE OF TRUTH
// ============================================================

let currentLat = 13.5;
let currentLon = 87.5;


// ============================================================
// INITIALIZE MAP
// ============================================================

function initLeafletMap() {

  console.log("Initializing CycloneAI Leaflet map...");

  const mapElement = document.getElementById("map");

  if (!mapElement) {
    console.error("#map element not found");
    return false;
  }

  // Prevent duplicate map initialization
  if (map) {
    console.warn("Leaflet map already initialized");
    return true;
  }

  map = L.map("map", {
    center: [currentLat, currentLon],
    zoom: 8,

    zoomControl: true,
    worldCopyJump: false,

    minZoom: 4,
    maxZoom: 19
  });

  currentTileLayer = L.tileLayer(TILE_LIGHT, {
    minZoom: 2,
    maxZoom: 19,
    attribution: OSM_ATTRIBUTION
  });

  currentTileLayer.addTo(map);

  console.log(
    `MAP INITIALIZED → ${currentLat.toFixed(2)}°N, ${currentLon.toFixed(2)}°E`
  );

  // Fix Leaflet rendering if map is inside a panel/sidebar
  setTimeout(() => {
    if (map) {
      map.invalidateSize();
    }
  }, 200);

  return true;
}


// ============================================================
// CENTER MAP
// ============================================================

function centerMapOnTarget(lat, lon, zoom = 8) {

  lat = Number(lat);
  lon = Number(lon);

  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {

    console.error(
      "Invalid coordinates:",
      lat,
      lon
    );

    return;
  }

  // Keep longitude/latitude inside valid geographic limits
  if (lat < -90 || lat > 90) {

    console.error("Invalid latitude:", lat);

    return;
  }

  if (lon < -180 || lon > 180) {

    console.error("Invalid longitude:", lon);

    return;
  }

  currentLat = lat;
  currentLon = lon;

  if (!map) {

    console.warn(
      "Map not initialized yet."
    );

    return;
  }

  map.setView(
    [lat, lon],
    zoom,
    {
      animate: true,
      duration: 0.8
    }
  );

  console.log(
    `MAP TARGET → ${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E`
  );
}


// ============================================================
// ADD / UPDATE STORM MARKER
// ============================================================

function updateStormMarker(lat, lon) {

  lat = Number(lat);
  lon = Number(lon);

  if (!map) {
    console.warn("Map is not initialized.");
    return;
  }

  if (
    !Number.isFinite(lat) ||
    !Number.isFinite(lon)
  ) {
    console.error(
      "Invalid storm coordinates:",
      lat,
      lon
    );
    return;
  }

  const stormIcon = L.divIcon({
    className: "cyclone-storm-marker",
    html: `
      <div class="storm-marker-ring">
        <div class="storm-marker-center"></div>
      </div>
    `,
    iconSize: [32, 32],
    iconAnchor: [16, 16]
  });

  if (stormMarker) {

    stormMarker.setLatLng([lat, lon]);

  } else {

    stormMarker = L.marker(
      [lat, lon],
      {
        icon: stormIcon,
        zIndexOffset: 1000
      }
    ).addTo(map);

  }

  console.log(
    `STORM MARKER → ${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E`
  );
}


// ============================================================
// CLEAR FORECAST
// ============================================================

function clearForecastTrack() {

  if (trackPolyline && map) {
    map.removeLayer(trackPolyline);
    trackPolyline = null;
  }

  forecastMarkers.forEach(marker => {

    if (map) {
      map.removeLayer(marker);
    }

  });

  forecastMarkers = [];
}


// ============================================================
// DRAW FORECAST TRACK
// ============================================================

function drawForecastTrack(points) {

  if (!map) {
    console.warn("Map is not initialized.");
    return;
  }

  if (!Array.isArray(points) || points.length === 0) {

    console.warn(
      "No forecast trajectory points available."
    );

    return;
  }

  clearForecastTrack();

  const coordinates = [];

  points.forEach((point, index) => {

    const lat = Number(
      point.latitude ??
      point.lat
    );

    const lon = Number(
      point.longitude ??
      point.lon
    );

    if (
      !Number.isFinite(lat) ||
      !Number.isFinite(lon)
    ) {
      return;
    }

    coordinates.push([lat, lon]);

    // Optional forecast point
    const marker = L.circleMarker(
      [lat, lon],
      {
        radius: 4,
        weight: 1,
        fillOpacity: 0.9
      }
    ).addTo(map);

    if (point.forecast_hours !== undefined) {

      marker.bindTooltip(
        `+${point.forecast_hours}h`,
        {
          direction: "top"
        }
      );

    }

    forecastMarkers.push(marker);
  });

  if (coordinates.length < 2) {

    console.warn(
      "Not enough valid points to draw forecast track."
    );

    return;
  }

  trackPolyline = L.polyline(
    coordinates,
    {
      weight: 4,
      opacity: 0.95
    }
  ).addTo(map);

  console.log(
    `FORECAST TRACK → ${coordinates.length} points`
  );
}


// ============================================================
// START
// ============================================================

document.addEventListener(
  "DOMContentLoaded",
  () => {

    console.log(
      "======================================"
    );

    console.log(
      "       CYCLONE AI STARTING"
    );

    console.log(
      "======================================"
    );

    // Theme
    if (typeof initThemeToggle === "function") {
      initThemeToggle();
    }

    // Map
    const mapReady = initLeafletMap();

    if (!mapReady) {
      console.error(
        "Cyclone AI startup stopped: map failed."
      );
      return;
    }

    // Satellite controls
    if (typeof initSatelliteControls === "function") {
      initSatelliteControls();
    }

    // Playback
    if (typeof initPlaybackControls === "function") {
      initPlaybackControls();
    }

    // DO NOT START OLD GOES CANVAS
    // initSatelliteCanvas(-68.5);

    // Events
    if (typeof bindEvents === "function") {
      bindEvents();
    }

    // Initial Bay of Bengal target
    if (typeof fetchCycloneData === "function") {

      fetchCycloneData(
        currentLat,
        currentLon
      );

    } else {

      console.warn(
        "fetchCycloneData() is not defined yet."
      );

    }

  }
);