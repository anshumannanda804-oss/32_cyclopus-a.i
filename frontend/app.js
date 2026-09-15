/* ============================================================
   CYCLONE AI — FRONTEND APP.JS
   Wired directly to the IDs/classes in index.html.
   ============================================================ */

/* ------------------------------------------------------------
   0. CONFIG
   ------------------------------------------------------------ */

const API_BASE = window.API_BASE || "http://127.0.0.1:8000";
const API = {
  cyclone: `${API_BASE}/api/v1/cyclone`,
  prediction: `${API_BASE}/api/v1/predict`
};

// Basemap / satellite weather imagery sources for the map layer switcher
const BASEMAPS = {
  osm: () =>
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors"
    }),

  // Real satellite-imagery basemap (Esri World Imagery)
  satellite: () =>
    L.tileLayer(
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      {
        maxZoom: 19,
        attribution: "Tiles &copy; Esri — Source: Esri, Maxar, Earthstar Geographics"
      }
    ),

  // Near-real-time NASA GIBS satellite weather composite (true-color, daily)
  nasa: () => {
    const d = new Date();
    d.setUTCDate(d.getUTCDate() - 1); // yesterday: today's composite may not be published yet
    const time = d.toISOString().slice(0, 10);
    return L.tileLayer.wms(
      "https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi",
      {
        layers: "VIIRS_SNPP_CorrectedReflectance_TrueColor",
        format: "image/jpeg",
        transparent: false,
        version: "1.1.1",
        time,
        attribution: "Imagery &copy; NASA GIBS / EOSDIS"
      }
    );
  }
};

/* ------------------------------------------------------------
   1. STATE
   ------------------------------------------------------------ */

const state = {
  map: null,
  baseLayer: null,
  stormMarker: null,
  trackLine: null,
  uncertaintyCone: null,
  forecastMarkers: [],
  activeStorm: "BOB-01",
  activeChannels: new Set(["ir"]), // multi-select curve set
  overlays: { eye: true, crosshair: true, cdo: true },
  lastMetrics: null,
  animPhase: 0,
  playing: false,
  playTimer: null
};

const STORMS = {
  "BOB-01": { lat: 15.0, lon: 90.0, name: "BOB-01 (Bay of Bengal)" },
  TEJ: { lat: 15.0, lon: 65.0, name: "TEJ (Arabian Sea)" }
};

/* ------------------------------------------------------------
   2. SMALL UTILITIES
   ------------------------------------------------------------ */

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function seedFromLatLon(lat, lon) {
  return Math.floor((lat * 1000 + lon * 1000 + 90000) % 2147483647);
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* ------------------------------------------------------------
   3. THEME TOGGLE
   ------------------------------------------------------------ */

function initThemeToggle() {
  const btn = document.getElementById("themeToggleBtn");
  const icon = document.getElementById("themeIcon");
  const text = document.getElementById("themeText");
  const root = document.documentElement;

  let saved = "dark";
  try {
    saved = localStorage.getItem("cyclone-ai-theme") || "dark";
  } catch (e) {}
  root.setAttribute("data-theme", saved);
  syncThemeLabel();

  function syncThemeLabel() {
    const dark = root.getAttribute("data-theme") === "dark";
    if (icon) icon.textContent = dark ? "☀️" : "🌙";
    if (text) text.textContent = dark ? "Light Mode" : "Dark Mode";
  }

  if (btn) {
    btn.addEventListener("click", () => {
      const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try {
        localStorage.setItem("cyclone-ai-theme", next);
      } catch (e) {}
      syncThemeLabel();
      drawSatellitePanel(); // repaint canvas legend colors etc.
    });
  }
}

/* ------------------------------------------------------------
   4. LEAFLET MAP + SATELLITE IMAGERY LAYER SWITCHER
   ------------------------------------------------------------ */

function initMap() {
  const el = document.getElementById("map");
  if (!el) return;

  state.map = L.map("map", { zoomControl: true, worldCopyJump: false, minZoom: 3, maxZoom: 19 })
    .setView([14.5, 86.2], 5);

  setBasemap("satellite"); // default to real satellite imagery, matches dropdown default

  setTimeout(() => state.map && state.map.invalidateSize(), 150);
}

function setBasemap(key) {
  if (!state.map || !BASEMAPS[key]) return;
  if (state.baseLayer) state.map.removeLayer(state.baseLayer);
  state.baseLayer = BASEMAPS[key]();
  state.baseLayer.addTo(state.map);

  const badge = document.getElementById("executionModeBadge");
  if (badge) {
    const label =
      key === "nasa" ? "STACK: NASA GIBS Satellite" : key === "satellite" ? "STACK: Esri Satellite Imagery" : "STACK: OpenStreetMap";
    badge.textContent = label;
  }
}

function updateStormMarker(lat, lon) {
  if (!state.map) return;
  const icon = L.divIcon({
    className: "cyclone-storm-marker",
    html: `<div style="width:16px;height:16px;border-radius:50%;background:${cssVar(
      "--accent-red"
    )};box-shadow:0 0 0 6px rgba(220,38,38,0.25),0 0 14px rgba(220,38,38,0.8);"></div>`,
    iconSize: [16, 16],
    iconAnchor: [8, 8]
  });
  if (state.stormMarker) {
    state.stormMarker.setLatLng([lat, lon]);
  } else {
    state.stormMarker = L.marker([lat, lon], { icon, zIndexOffset: 1000 }).addTo(state.map);
  }
}

function drawForecastTrack(origin, points) {
  if (!state.map) return;
  [state.trackLine, state.uncertaintyCone, ...state.forecastMarkers].forEach(
    (l) => l && state.map.removeLayer(l)
  );
  state.forecastMarkers = [];

  const coords = [[origin.lat, origin.lon], ...points.map((p) => [p.lat, p.lon])];
  state.trackLine = L.polyline(coords, {
    color: cssVar("--accent-red") || "#dc2626",
    weight: 3,
    opacity: 0.9,
    dashArray: "6 4"
  }).addTo(state.map);

  const coneCoords = points.map((p, i) => {
    const r = 0.35 + i * 0.35; // widening uncertainty cone, degrees
    return L.circle([p.lat, p.lon], { radius: r * 111000, weight: 0 }).getLatLng
      ? null
      : null;
  });
  // Simple widening cone polygon (upper + lower offset track)
  const upper = points.map((p, i) => [p.lat + (0.25 + i * 0.25), p.lon]);
  const lower = points.map((p, i) => [p.lat - (0.25 + i * 0.25), p.lon]).reverse();
  state.uncertaintyCone = L.polygon([[origin.lat, origin.lon], ...upper, ...lower], {
    color: "transparent",
    fillColor: cssVar("--accent-cyan") || "#0284c7",
    fillOpacity: 0.12
  }).addTo(state.map);

  points.forEach((p) => {
    const m = L.circleMarker([p.lat, p.lon], {
      radius: 4,
      weight: 1,
      color: "#fff",
      fillColor: cssVar("--accent-cyan") || "#0284c7",
      fillOpacity: 0.95
    })
      .addTo(state.map)
      .bindTooltip(`+${p.hours}h · ${p.windKnots}kt`, { direction: "top" });
    state.forecastMarkers.push(m);
  });
}

/* ------------------------------------------------------------
   5. CYCLONE PHYSICS (client-side port of detection_model.py)
   ------------------------------------------------------------ */

function computeMetrics(lat, lon) {
  const rand = mulberry32(seedFromLatLon(lat, lon));

  const sst = clamp(29.5 - Math.abs(lat - 12) * 0.35 + (rand() - 0.5) * 2, 24, 31.5);
  const shear = clamp(8 + Math.abs(lon - 88) * 0.6 + (rand() - 0.5) * 8, 3, 35);
  const rh = clamp(85 - Math.abs(lon - 88) * 0.5 + (rand() - 0.5) * 15, 40, 95);
  const wind = clamp(65 - shear * 1.1 + (sst - 27) * 6 + (rand() - 0.5) * 10, 15, 140);
  const mslp = clamp(1004 - wind * 0.42, 900, 1010);
  const ctt = clamp(-40 - wind * 0.5 + (rand() - 0.5) * 6, -90, -35);

  // Genesis Potential Index (Emanuel & Nolan style, simplified)
  const potentialIntensity = sst < 25 ? 0 : Math.max(0, (sst - 25) * 8.5);
  const vortTerm = Math.pow(Math.abs(1e5 * 3.5e-5), 1.5);
  const rhTerm = Math.pow(Math.max(0, rh) / 50, 3);
  const piTerm = Math.pow(potentialIntensity / 70, 3);
  const shearMs = shear * 0.5144;
  const shearTerm = Math.pow(1 + 0.1 * shearMs, -2);
  const gpi = Number((vortTerm * rhTerm * piTerm * shearTerm).toFixed(2));

  let gpiCategory = "LOW";
  if (gpi >= 15) gpiCategory = "VERY HIGH / EXTREME";
  else if (gpi >= 5) gpiCategory = "HIGH";
  else if (gpi >= 1) gpiCategory = "MODERATE";

  // Dvorak T-number / eye
  const eyeDetected = ctt <= -65;
  let tBase;
  if (ctt <= -75) tBase = 5.5;
  else if (ctt <= -65) tBase = 4.5;
  else if (ctt <= -55) tBase = 3.5;
  else if (ctt <= -45) tBase = 2.5;
  else tBase = 1.5;
  const windBasedT = 1 + wind / 20;
  const tNumber = Number(clamp(tBase * 0.6 + windBasedT * 0.4, 1, 8).toFixed(1));

  const imdCategory = getImdCategory(wind);
  const riRisk = clamp(tNumber * 12 + (sst - 27) * 8 - shear * 1.5, 5, 92);

  return {
    lat,
    lon,
    sst: Number(sst.toFixed(1)),
    shear: Number(shear.toFixed(1)),
    rh: Number(rh.toFixed(1)),
    wind: Number(wind.toFixed(1)),
    mslp: Number(mslp.toFixed(1)),
    ctt: Number(ctt.toFixed(1)),
    gpi,
    gpiCategory,
    ohc: Number((potentialIntensity * 1.05 + rand() * 10).toFixed(1)),
    eyeDetected,
    eyeRadiusKm: eyeDetected ? Number((10 + rand() * 12).toFixed(0)) : 0,
    tNumber,
    imdCategory,
    riRisk: Number(riRisk.toFixed(1))
  };
}

function getImdCategory(wind) {
  if (wind < 17) return "Low Pressure Area";
  if (wind < 28) return "Depression";
  if (wind < 34) return "Deep Depression";
  if (wind < 48) return "Cyclonic Storm";
  if (wind < 64) return "Severe Cyclonic Storm";
  if (wind < 90) return "Very Severe Cyclonic Storm";
  if (wind < 120) return "Extremely Severe Cyclonic Storm";
  return "Super Cyclonic Storm";
}

function forecastTrajectory(m) {
  const rand = mulberry32(seedFromLatLon(m.lat, m.lon) + 7);
  const steeringDir = 300 + (rand() - 0.5) * 40; // deg, roughly NW recurvature
  const steeringSpeed = 8 + rand() * 8; // knots
  const points = [];
  let lat = m.lat,
    lon = m.lon,
    wind = m.wind;

  [12, 24, 36, 48, 60, 72].forEach((hours, i) => {
    const dirRad = (steeringDir * Math.PI) / 180;
    const distDeg = (steeringSpeed * 12) / 60; // deg per 12h step, rough
    lat += Math.cos(dirRad) * distDeg * (0.9 + rand() * 0.2);
    lon += Math.sin(dirRad) * distDeg * (0.9 + rand() * 0.2);
    wind = Math.max(15, wind + (lat > 20 ? -8 : rand() > 0.5 ? 2 : -3));
    points.push({ lat, lon, hours, windKnots: Math.round(wind) });
  });
  return points;
}

/* ------------------------------------------------------------
   6. SIDEBAR METRICS RENDER
   ------------------------------------------------------------ */

function renderMetrics(m) {
  state.lastMetrics = m;

  setText("valDvorakT", `T${m.tNumber} (CI ${m.tNumber})`);
  setText("valEyeDetected", m.eyeDetected ? `DETECTED (${m.eyeRadiusKm} km)` : "NOT DETECTED");
  setText("valRIRisk", `${m.riRisk.toFixed(1)}% (${m.riRisk > 60 ? "HIGH" : m.riRisk > 30 ? "MODERATE" : "LOW"})`);
  setText("valGPI", `${m.gpi} (${m.gpiCategory})`);
  setText("valSST", `${m.sst} °C`);
  setText("valOHC", `${m.ohc} kJ/cm²`);
  setText("valMSLP", `${m.mslp} hPa`);
  setText("valWindSpeed", `${m.wind} knots`);
  setText("valShear", `${m.shear} knots`);

  const statusBadge = document.getElementById("detectionStatusBadge");
  if (statusBadge) statusBadge.textContent = m.tNumber >= 2.5 ? "ACTIVE" : "MONITORING";
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

/* ------------------------------------------------------------
   7. SATELLITE PANEL — MULTI-CURVE RENDERING
   ------------------------------------------------------------ */

const CURVES = {
  ir: { label: "IR 10.8 µm" },
  bd: { label: "BD-Curve" },
  wv: { label: "WV 6.2 µm" },
  vis: { label: "VIS 0.6 µm" }
};

// Simulated cloud-top temperature field (0=warm/ocean .. 255=very cold/tall cloud)
function cloudField(size, m, phase) {
  const field = new Float32Array(size * size);
  const cx = size / 2 + Math.sin(phase * 0.3) * 3;
  const cy = size / 2 + Math.cos(phase * 0.3) * 3;
  const coreRadius = clamp(22 - m.tNumber * 1.2, 10, 22);
  const armCount = m.tNumber >= 4.5 ? 4 : m.tNumber >= 3.5 ? 3 : 2;
  const eyeR = m.eyeDetected ? clamp(m.eyeRadiusKm / 4, 2, 9) : 0;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = x - cx,
        dy = y - cy;
      const r = Math.sqrt(dx * dx + dy * dy);
      const theta = Math.atan2(dy, dx);
      let v = Math.max(0, 1 - r / (coreRadius + 14)) ** 1.5 * 235;

      for (let a = 0; a < armCount; a++) {
        const off = (2 * Math.PI * a) / armCount + phase * 0.15;
        const spiralR = 7 * Math.exp(0.16 * (theta - off + Math.PI * 2 * Math.floor((r / 40) )));
        const expected = 7 * Math.exp(0.16 * ((theta - off) % (2 * Math.PI)));
        const diff = Math.abs(r - Math.abs(expected)) ;
        v += Math.exp(-(diff * diff) / (2 * 5 * 5)) * 140 * (r < coreRadius + 20 ? 1 : 0);
      }

      if (eyeR > 0 && r < eyeR + 2) {
        const edge = clamp(1 - (r - eyeR) / 3, 0, 1);
        v = v * (1 - edge) + 25 * edge;
      }

      field[y * size + x] = clamp(v + (Math.random() - 0.5) * 6, 0, 255);
    }
  }
  return field;
}

// value(0-255, higher = colder/taller cloud) -> [r,g,b] per curve
function curveColor(v, curve) {
  const t = v / 255;
  switch (curve) {
    case "ir": {
      const g = Math.round(t * 255);
      return [g, g, g];
    }
    case "bd": {
      // Classic Dvorak-style posterized enhancement bands
      if (t > 0.93) return [255, 0, 0]; // extreme cold core
      if (t > 0.85) return [255, 255, 255];
      if (t > 0.75) return [20, 20, 20];
      if (t > 0.6) return [235, 235, 235];
      if (t > 0.4) return [140, 140, 140];
      return [15, 15, 30];
    }
    case "wv": {
      const b = Math.round(60 + t * 195);
      const g = Math.round(t * 120);
      return [10, g, b];
    }
    case "vis": {
      const g = Math.round(30 + t * 200);
      return [g, g, Math.round(g * 0.95)];
    }
    default:
      return [t * 255, t * 255, t * 255];
  }
}

function drawSatellitePanel() {
  const canvas = document.getElementById("satelliteCanvas");
  if (!canvas || !state.lastMetrics) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width,
    H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const channels = Array.from(state.activeChannels);
  if (channels.length === 0) channels.push("ir");

  const cols = channels.length <= 1 ? 1 : channels.length <= 2 ? 2 : 2;
  const rows = Math.ceil(channels.length / cols);
  const cellW = Math.floor(W / cols);
  const cellH = Math.floor(H / rows);
  const fieldSize = 96;
  const field = cloudField(fieldSize, state.lastMetrics, state.animPhase);

  channels.forEach((ch, idx) => {
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    const ox = col * cellW;
    const oy = row * cellH;
    renderCurveCell(ctx, field, fieldSize, ox, oy, cellW, cellH, ch);

    if (channels.length > 1) {
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.fillRect(ox + 4, oy + 4, ctx.measureText(CURVES[ch].label).width + 14, 16);
      ctx.fillStyle = "#fff";
      ctx.font = "10px 'Outfit', sans-serif";
      ctx.fillText(CURVES[ch].label, ox + 10, oy + 15);
    }
  });

  // grid separators for multi-panel
  if (channels.length > 1) {
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth = 1;
    for (let c = 1; c < cols; c++) {
      ctx.beginPath();
      ctx.moveTo(c * cellW, 0);
      ctx.lineTo(c * cellW, H);
      ctx.stroke();
    }
    for (let r = 1; r < rows; r++) {
      ctx.beginPath();
      ctx.moveTo(0, r * cellH);
      ctx.lineTo(W, r * cellH);
      ctx.stroke();
    }
  }

  const legend = document.getElementById("satLegend");
  if (legend) {
    const names = channels.map((c) => CURVES[c].label).join(" · ");
    legend.textContent = channels.length > 1 ? `☁️ Showing all curves: ${names}` : `☁️ ${names}: White/Blue = Cold, Grey = Warm`;
  }
}

function renderCurveCell(ctx, field, fieldSize, ox, oy, cellW, cellH, curve) {
  const img = ctx.createImageData(fieldSize, fieldSize);
  for (let i = 0; i < fieldSize * fieldSize; i++) {
    const [r, g, b] = curveColor(field[i], curve);
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = 255;
  }
  // draw to an offscreen canvas then scale into the cell (crisp nearest-neighbor upscale)
  const off = document.createElement("canvas");
  off.width = fieldSize;
  off.height = fieldSize;
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, ox, oy, cellW, cellH);

  const cx = ox + cellW / 2;
  const cy = oy + cellH / 2;
  const scale = Math.min(cellW, cellH) / fieldSize;

  if (state.overlays.crosshair) {
    ctx.strokeStyle = "rgba(56,189,248,0.85)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - 10, cy);
    ctx.lineTo(cx + 10, cy);
    ctx.moveTo(cx, cy - 10);
    ctx.lineTo(cx, cy + 10);
    ctx.stroke();
  }
  if (state.overlays.cdo) {
    ctx.strokeStyle = "rgba(251,146,60,0.8)";
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.arc(cx, cy, 30 * scale, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  if (state.overlays.eye && state.lastMetrics.eyeDetected) {
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, clamp(state.lastMetrics.eyeRadiusKm / 4, 2, 9) * scale, 0, Math.PI * 2);
    ctx.stroke();
  }
}

/* ------------------------------------------------------------
   8. EVENTS
   ------------------------------------------------------------ */

function bindEvents() {
  const fetchBtn = document.getElementById("btnFetchData");
  if (fetchBtn) fetchBtn.addEventListener("click", () => runPipeline());

  const stormSelect = document.getElementById("selectStorm");
  if (stormSelect)
    stormSelect.addEventListener("change", (e) => {
      const s = STORMS[e.target.value];
      state.activeStorm = e.target.value;
      if (s) {
        setInputVal("inputLat", s.lat);
        setInputVal("inputLon", s.lon);
        runPipeline();
      }
    });

  const layerSwitcher = document.getElementById("mapLayerSwitcher");
  if (layerSwitcher) layerSwitcher.addEventListener("change", (e) => setBasemap(e.target.value));

  document.querySelectorAll(".channel-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const ch = btn.dataset.channel;
      if (state.activeChannels.has(ch) && state.activeChannels.size > 1) {
        state.activeChannels.delete(ch);
        btn.classList.remove("active");
      } else {
        state.activeChannels.add(ch);
        btn.classList.add("active");
      }
      drawSatellitePanel();
    });
  });

  bindOverlayToggle("toggleEye", "eye");
  bindOverlayToggle("toggleCrosshair", "crosshair");
  bindOverlayToggle("toggleCDO", "cdo");

  const probe = document.getElementById("thermalProbe");
  const canvas = document.getElementById("satelliteCanvas");
  if (canvas && probe) {
    canvas.addEventListener("mousemove", (e) => {
      const rect = canvas.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
      const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
      const m = state.lastMetrics;
      if (!m) return;
      const tempC = Math.round(-30 - (y / canvas.height) * 60);
      setText("probeTemp", `${tempC} °C`);
      setText("probeHeight", `${(6 + Math.abs(tempC) / 8).toFixed(1)} km`);
      setText("probeClass", m.imdCategory);
      probe.style.left = `${e.clientX - rect.left + 14}px`;
      probe.style.top = `${e.clientY - rect.top + 14}px`;
      probe.style.display = "flex";
    });
    canvas.addEventListener("mouseleave", () => (probe.style.display = "none"));
  }

  const playBtn = document.getElementById("btnPlayPause");
  if (playBtn) playBtn.addEventListener("click", togglePlayback);

  const stepBack = document.getElementById("btnStepBack");
  if (stepBack) stepBack.addEventListener("click", () => stepFrame(-1));

  const stepFwd = document.getElementById("btnStepFwd");
  if (stepFwd) stepFwd.addEventListener("click", () => stepFrame(1));

  const timeSlider = document.getElementById("timeSlider");
  if (timeSlider)
    timeSlider.addEventListener("input", () => {
      updateTimeLabel();
      drawSatellitePanel();
    });

  const exportBtn = document.getElementById("btnDownloadSnapshot");
  if (exportBtn) exportBtn.addEventListener("click", exportSnapshot);

  if (state.map) {
    state.map.on("click", (e) => {
      setInputVal("inputLat", e.latlng.lat.toFixed(2));
      setInputVal("inputLon", e.latlng.lng.toFixed(2));
      runPipeline();
    });
  }
}

function bindOverlayToggle(id, key) {
  const el = document.getElementById(id);
  if (!el) return;
  el.addEventListener("change", () => {
    state.overlays[key] = el.checked;
    drawSatellitePanel();
  });
}

function setInputVal(id, val) {
  const el = document.getElementById(id);
  if (el) el.value = val;
}

function togglePlayback() {
  const btn = document.getElementById("btnPlayPause");
  state.playing = !state.playing;
  if (btn) btn.textContent = state.playing ? "⏸️" : "▶️";
  if (state.playing) {
    const speed = Number(document.getElementById("speedSlider")?.value || 5);
    state.playTimer = setInterval(() => {
      state.animPhase += 0.15;
      drawSatellitePanel();
    }, Math.max(60, 260 - speed * 20));
  } else {
    clearInterval(state.playTimer);
  }
}

function stepFrame(dir) {
  const slider = document.getElementById("timeSlider");
  if (!slider) return;
  slider.value = clamp(Number(slider.value) + dir, Number(slider.min), Number(slider.max));
  updateTimeLabel();
  state.animPhase += dir * 0.4;
  drawSatellitePanel();
}

function updateTimeLabel() {
  const slider = document.getElementById("timeSlider");
  const label = document.getElementById("timeLabel");
  if (!slider || !label) return;
  const stepsAgo = Number(slider.max) - Number(slider.value);
  label.textContent = stepsAgo === 0 ? "LIVE" : `-${stepsAgo * 30}min`;
}

function exportSnapshot() {
  const canvas = document.getElementById("satelliteCanvas");
  if (!canvas) return;
  const link = document.createElement("a");
  link.download = "cyclone-ai-satellite.png";
  link.href = canvas.toDataURL("image/png");
  link.click();
}

/* ------------------------------------------------------------
   9. PIPELINE (backend attempt -> local fallback)
   ------------------------------------------------------------ */

async function tryBackend(lat, lon) {
  try {
    const res = await fetch(API.cyclone, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        data: { target_location: { latitude: lat, longitude: lon } }
      }),
      signal: AbortSignal.timeout(2500)
    });
    if (!res.ok) throw new Error("bad response");
    return await res.json();
  } catch (e) {
    return null; // offline / no backend — fall back to local model
  }
}

async function runPipeline() {
  const lat = Number(document.getElementById("inputLat")?.value ?? 14.5);
  const lon = Number(document.getElementById("inputLon")?.value ?? 86.2);

  const backendResult = await tryBackend(lat, lon);
  const m = backendResult ? mapBackendToMetrics(backendResult, lat, lon) : computeMetrics(lat, lon);

  renderMetrics(m);
  updateStormMarker(lat, lon);
  state.map && state.map.setView([lat, lon], state.map.getZoom() < 5 ? 6 : state.map.getZoom(), { animate: true });
  drawForecastTrack({ lat, lon }, forecastTrajectory(m));
  drawSatellitePanel();
}

function mapBackendToMetrics(payload, lat, lon) {
  try {
    const g = payload.cyclogenesis_assessment;
    const i = payload.intensity_and_eye_detection;
    return {
      lat,
      lon,
      sst: g.environmental_factors.sst_celsius,
      shear: g.environmental_factors.vertical_wind_shear_knots,
      rh: g.environmental_factors.relative_humidity_pct,
      wind: i.estimated_max_sustained_wind_knots,
      mslp: 1010 - i.central_pressure_deficit_hpa,
      ctt: i.cloud_top_min_temp_celsius,
      gpi: g.genesis_potential_index,
      gpiCategory: g.cyclogenesis_risk,
      ohc: 75,
      eyeDetected: i.eye_detected,
      eyeRadiusKm: i.eye_radius_km,
      tNumber: i.dvorak_t_number,
      imdCategory: i.imd_cyclone_category,
      riRisk: clamp(i.dvorak_t_number * 12, 5, 92)
    };
  } catch (e) {
    return computeMetrics(lat, lon);
  }
}

/* ------------------------------------------------------------
   10. INIT
   ------------------------------------------------------------ */

async function initializeApp() {
  initThemeToggle();
  initMap();
  document.querySelectorAll(".channel-btn").forEach((b) => {
    if (b.dataset.channel === "ir") b.classList.add("active");
  });
  bindEvents();
  updateTimeLabel();
  await runPipeline();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeApp);
} else {
  initializeApp();
}

window.CycloneAI = {
  state,
  setBasemap,
  runPipeline,
  drawSatellitePanel
};