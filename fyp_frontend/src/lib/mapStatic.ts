/**
 * mapStatic.ts — Build Mapbox Static Images API URLs for a day's map.
 *
 * Turns a backend GeoJSON FeatureCollection (point pins + one route LineString
 * per travel profile) into a single <img> URL — a real map with markers and the
 * selected route drawn, without pulling in the heavy mapbox-gl runtime.
 *
 * Requires a public Mapbox token (VITE_MAPBOX_TOKEN). Returns null when there is
 * no token or nothing to draw, so callers can fall back to a text summary.
 */
import type { GeoJSONFeatureCollection } from './api';
import { MAPBOX_TOKEN } from './api';

const PIN_COLOR: Record<string, string> = {
  hotel: '2563eb', // blue
  airport: '64748b', // slate
  restaurant: 'f59e0b', // amber
  attraction: '10b981', // emerald
  activity: '10b981',
};
const DEFAULT_PIN = '6366f1'; // indigo
const ROUTE_COLOR = '2563eb';

/** Encode [lat,lng] pairs as a Google/Mapbox precision-5 polyline. */
function encodePolyline(points: [number, number][]): string {
  let result = '';
  let prevLat = 0;
  let prevLng = 0;

  const encodeVal = (value: number): string => {
    let v = value < 0 ? ~(value << 1) : value << 1;
    let out = '';
    while (v >= 0x20) {
      out += String.fromCharCode((0x20 | (v & 0x1f)) + 63);
      v >>= 5;
    }
    out += String.fromCharCode(v + 63);
    return out;
  };

  for (const [lat, lng] of points) {
    const iLat = Math.round(lat * 1e5);
    const iLng = Math.round(lng * 1e5);
    result += encodeVal(iLat - prevLat);
    result += encodeVal(iLng - prevLng);
    prevLat = iLat;
    prevLng = iLng;
  }
  return result;
}

// ─────────────────────────────────────────────────────────────────────
// Marker-only maps (emergency / place lookups from chat [MAP: …] lines)
// ─────────────────────────────────────────────────────────────────────

export interface MarkerPoint {
  lat: number;
  lng: number;
  label?: string;
  /** Street address from the [MAP:] line — used to re-geocode the pin. */
  address?: string;
}

/**
 * Build a static map URL pinning one or more standalone locations
 * (police station, hospital, embassy, …). Multiple pins get letter
 * labels (a, b, c…) matching the caption order. Returns null when
 * there is no token or no valid point.
 */
export function buildMarkersMapUrl(
  points: MarkerPoint[],
  opts: { width?: number; height?: number; color?: string } = {},
): string | null {
  if (!MAPBOX_TOKEN || !Array.isArray(points) || points.length === 0) return null;
  const { width = 640, height = 320, color = 'ef4444' } = opts;

  const valid = points
    .filter(
      (p) =>
        p != null &&
        typeof p === 'object' &&
        Number.isFinite(p.lat) &&
        Number.isFinite(p.lng) &&
        p.lat >= -90 &&
        p.lat <= 90 &&
        p.lng >= -180 &&
        p.lng <= 180,
    )
    .slice(0, 10);
  if (valid.length === 0) return null;

  const letters = 'abcdefghij';
  const pins = valid
    .map((p, i) => {
      const icon = valid.length > 1 ? letters[i] : 'marker';
      return `pin-l-${icon}+${color}(${p.lng.toFixed(5)},${p.lat.toFixed(5)})`;
    })
    .join(',');

  // Single pin: fixed street-level zoom ('auto' zooms in too far on one point).
  // The Static Images API rejects `padding` combined with a manual
  // center/zoom (HTTP 422) — it is only valid with the 'auto' view.
  const single = valid.length === 1;
  const view = single
    ? `${valid[0].lng.toFixed(5)},${valid[0].lat.toFixed(5)},14`
    : 'auto';
  const padding = single ? '' : 'padding=60&';

  return (
    `https://api.mapbox.com/styles/v1/mapbox/streets-v12/static/` +
    `${pins}/${view}/${width}x${height}@2x?${padding}access_token=${MAPBOX_TOKEN}`
  );
}

export interface StaticMapOptions {
  profile?: string; // which route profile's line to draw
  width?: number;
  height?: number;
}

/**
 * Build a Mapbox static image URL for one day's FeatureCollection.
 * Returns null if there is no token or no drawable geometry.
 */
export function buildStaticMapUrl(
  fc: GeoJSONFeatureCollection | undefined | null,
  opts: StaticMapOptions = {},
): string | null {
  if (!MAPBOX_TOKEN || !fc || !Array.isArray(fc.features) || fc.features.length === 0) {
    return null;
  }

  const { profile = 'driving', width = 640, height = 360 } = opts;

  // Pins from Point features
  const markers: string[] = [];
  for (const f of fc.features) {
    if (f.geometry?.type !== 'Point') continue;
    const [lng, lat] = f.geometry.coordinates as [number, number];
    if (
      !Number.isFinite(lng) ||
      !Number.isFinite(lat) ||
      lat < -90 ||
      lat > 90 ||
      lng < -180 ||
      lng > 180
    ) continue;
    const color = PIN_COLOR[f.properties?.type] ?? DEFAULT_PIN;
    markers.push(`pin-s+${color}(${lng.toFixed(5)},${lat.toFixed(5)})`);
  }
  if (markers.length === 0) return null;

  // Route line: prefer the requested profile, else the first route feature
  const routeFeatures = fc.features.filter(
    (f) => f.geometry?.type === 'LineString' && f.properties?.type === 'route',
  );
  const chosen =
    routeFeatures.find((f) => f.properties?.profile === profile) ?? routeFeatures[0];

  const overlays: string[] = [];
  if (chosen) {
    const coords = (chosen.geometry.coordinates as [number, number][]) || [];
    // geojson is [lng,lat]; polyline wants [lat,lng]
    const latlngs: [number, number][] = coords
      .filter((c) => Array.isArray(c) && c.length >= 2)
      .map((c) => [c[1], c[0]]);
    if (latlngs.length >= 2) {
      const encoded = encodeURIComponent(encodePolyline(latlngs));
      overlays.push(`path-4+${ROUTE_COLOR}-0.75(${encoded})`);
    }
  }
  // Path first so pins render on top
  overlays.push(...markers);

  const overlayStr = overlays.join(',');
  let url =
    `https://api.mapbox.com/styles/v1/mapbox/streets-v12/static/` +
    `${overlayStr}/auto/${width}x${height}@2x?padding=48&access_token=${MAPBOX_TOKEN}`;

  // Static Images API rejects very long URLs (~8192). If the encoded route path
  // blows the budget, drop it and keep the pins.
  if (url.length > 7800 && overlays.length > markers.length) {
    const pinsOnly = markers.join(',');
    url =
      `https://api.mapbox.com/styles/v1/mapbox/streets-v12/static/` +
      `${pinsOnly}/auto/${width}x${height}@2x?padding=48&access_token=${MAPBOX_TOKEN}`;
  }
  return url;
}
