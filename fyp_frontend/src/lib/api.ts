/**
 * api.ts — Backend API client for the Travel Assistant AI.
 *
 * Wraps the FastAPI backend (fyp_backend) endpoints:
 *   • POST /api/profile        → onboarding (Name / Origin Country / State)
 *   • GET  /api/profile        → check onboarding status
 *   • POST /api/form/submit    → bootstrap a new trip from a structured form
 *   • POST /api/chat/message   → continue a conversation on an existing session
 *
 * Every request injects the `X-User-ID` header the backend expects
 * (see app/api/dependencies.py::verify_user_token). Auth is deferred to v2;
 * for now this is a stable per-device id.
 *
 * Config comes from Vite env vars (see .env.local):
 *   • VITE_API_BASE_URL  — e.g. http://localhost:8000
 *   • VITE_USER_ID       — a stable user id sent as X-User-ID
 *   • VITE_MAPBOX_TOKEN  — (optional) public Mapbox token for static day maps
 */

const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ??
  'http://localhost:8000';

const USER_ID: string =
  (import.meta.env.VITE_USER_ID as string | undefined) ?? 'test-user-001';

export const MAPBOX_TOKEN: string =
  (import.meta.env.VITE_MAPBOX_TOKEN as string | undefined) ?? '';

// ── Types mirroring the backend response schemas ────────────────────

/** Budget context attached to itinerary-bearing responses. */
export interface BudgetInfo {
  total: number;
  currency: string;
  allocation: Record<string, number>;
}

export interface ChatSuccessResponse {
  status: 'success';
  chat_reply: string;
  draft_itinerary: DayItinerary[];
  daily_map_info: Record<string, GeoJSONFeatureCollection>;
  destination_country_code: string;
  /** True only when this turn actually changed the itinerary/budget/trip. */
  itinerary_modified: boolean;
  total_budget: number;
  currency: string;
  budget_allocation: Record<string, number>;
  budget_confirmation: null;
}

export interface ChatBudgetConfirmationResponse {
  status: 'budget_confirmation_required';
  chat_reply: string;
  draft_itinerary: [];
  daily_map_info: Record<string, never>;
  itinerary_modified: false;
  total_budget: 0;
  currency: '';
  budget_allocation: Record<string, never>;
  budget_confirmation: ChatBudgetConfirmation;
}

export interface ChatBudgetCheckUnavailableResponse {
  status: 'budget_check_unavailable';
  chat_reply: string;
  draft_itinerary: [];
  daily_map_info: Record<string, never>;
  itinerary_modified: false;
  total_budget: 0;
  currency: '';
  budget_allocation: Record<string, never>;
  budget_confirmation: null;
}

export interface ChatPlanningUnavailableResponse {
  status: 'planning_unavailable';
  reason: PlanningUnavailableReason;
  chat_reply: string;
  retryable: true;
  draft_itinerary: [];
  daily_map_info: Record<string, never>;
  itinerary_modified: false;
  total_budget: 0;
  currency: '';
  budget_allocation: Record<string, never>;
  budget_confirmation: null;
}

export type ChatResponse =
  | ChatSuccessResponse
  | ChatBudgetConfirmationResponse
  | ChatBudgetCheckUnavailableResponse
  | ChatPlanningUnavailableResponse;

export interface ChatHistoryMessage {
  role: 'user' | 'ai';
  content: string;
  itinerary?: DayItinerary[];
  maps?: Record<string, GeoJSONFeatureCollection>;
  budget?: BudgetInfo;
  budget_confirmation?: ChatBudgetConfirmation | null;
  destination_country_code?: string;
}

export interface ChatHistorySession {
  id: string;
  title: string;
  destination: string;
  updated_at: string;
  messages: ChatHistoryMessage[];
}

export interface ChatHistoryResponse {
  status: 'success';
  sessions: ChatHistorySession[];
}

/** FinalResponse from app/schemas/responses.py */
export interface FinalResponse {
  status: 'success';
  chat_reply: string;
  itinerary: DayItinerary[];
  daily_geojson_maps: Record<string, GeoJSONFeatureCollection>;
  destination_country_code: string;

  /** Verified cities actually used for planning. */
  resolved_cities: string[];

  total_budget: number;
  currency: string;
  budget_allocation: Record<string, number>;
  session_id: string;
}

/** OnboardingRequest from app/schemas/requests.py — all fields mandatory. */
export interface OnboardingRequest {
  name: string;
  origin_country: string;
  origin_state: string;
}

/** GET /api/profile response */
export interface ProfileResponse {
  status: string;
  onboarded: boolean;
  profile: Record<string, any>;
}

/**
 * InitialFormRequest from app/schemas/requests.py.
 * NOTE: origin_country / origin_state are NO LONGER sent here — the backend
 * reads them from the onboarding profile. Destination localities are optional
 * at the public form boundary. When omitted, the backend resolves and verifies
 * a planning city before budget, provider, and itinerary operations begin.
 */
export interface TripFormRequest {
  country: string;
  city: string[];
  num_people: number;
  /** User-entered amount. Omitted only when they explicitly request guidance. */
  total_budget?: number;
  /** Explicit opt-in: permits a provider-grounded recommendation. */
  request_budget_recommendation?: boolean;
  /** Cached evidence token returned by a prior budget check. */
  budget_assessment_id?: string;
  start_date: string; // YYYY-MM-DD
  end_date: string; // YYYY-MM-DD
}

export interface BudgetEvidenceResponse {
  outbound_flight_price: number;
  return_flight_price: number;
  hotel_price_per_night: number;
  hotel_nights: number;
}

export interface ChatBudgetConfirmation {
  status: 'budget_confirmation_required';
  reason: 'insufficient_budget' | 'recommendation_requested';
  chat_reply: string;
  budget_assessment_id: string;
  /** Present for initial-form budget confirmations. */
  resolved_cities?: string[];
  stated_budget: number | null;
  recommended_minimum_budget: number;
  base_currency: string;
  destination_currency: string;
  expires_at: string;
  evidence: BudgetEvidenceResponse;
  itinerary: null;
  daily_geojson_maps: null;
}

export interface ChatMessageOptions {
  budget_action?: 'accept_recommended';
  budget_assessment_id?: string;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const hasOnlyKeys = (
  value: Record<string, unknown>,
  allowed: readonly string[],
  required: readonly string[] = [],
): boolean => {
  const allowedKeys = new Set(allowed);
  return Object.keys(value).every((key) => allowedKeys.has(key))
    && required.every((key) => Object.prototype.hasOwnProperty.call(value, key));
};

const isNonEmptyString = (value: unknown): value is string =>
  typeof value === 'string' && value.trim().length > 0;

const isNonEmptyStringArray = (value: unknown): value is string[] =>
  Array.isArray(value)
  && value.length > 0
  && value.every((item) => isNonEmptyString(item));

const isCountryCode = (value: unknown): value is string =>
  typeof value === 'string' && /^[A-Z]{2}$/.test(value);

export const isValidCurrency = (value: unknown): value is string => {
  if (typeof value !== 'string' || !/^[A-Z]{3}$/.test(value)) return false;
  try {
    new Intl.NumberFormat('en', { style: 'currency', currency: value });
    return true;
  } catch {
    return false;
  }
};

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

export function isChatBudgetConfirmation(value: unknown): value is ChatBudgetConfirmation {
  if (!isRecord(value)
    || !hasOnlyKeys(value, [
      'status', 'reason', 'chat_reply', 'budget_assessment_id',
      'resolved_cities',
      'stated_budget',
      'recommended_minimum_budget', 'base_currency', 'destination_currency',
      'expires_at', 'evidence', 'itinerary', 'daily_geojson_maps',
    ], [
      'status', 'reason', 'chat_reply', 'budget_assessment_id',
      'stated_budget',
      'recommended_minimum_budget', 'base_currency', 'destination_currency',
      'expires_at', 'evidence', 'itinerary', 'daily_geojson_maps',
    ])
    || value.status !== 'budget_confirmation_required'
    || (value.reason !== 'insufficient_budget' && value.reason !== 'recommendation_requested')
    || !isSafeAssistantContent(value.chat_reply)
    || typeof value.budget_assessment_id !== 'string'
    || !value.budget_assessment_id.trim()
    || (
      value.resolved_cities !== undefined
      && !isNonEmptyStringArray(value.resolved_cities)
    )
    || !(value.stated_budget === null || (
      isFiniteNumber(value.stated_budget)
      && value.stated_budget > 0
    ))
    || !isFiniteNumber(value.recommended_minimum_budget) || value.recommended_minimum_budget <= 0
    || !isValidCurrency(value.base_currency)
    || !isValidCurrency(value.destination_currency)
    || typeof value.expires_at !== 'string'
    || !Number.isFinite(Date.parse(value.expires_at))
    || !isRecord(value.evidence)
    || !hasOnlyKeys(value.evidence, [
      'outbound_flight_price', 'return_flight_price', 'hotel_price_per_night', 'hotel_nights',
    ], [
      'outbound_flight_price', 'return_flight_price', 'hotel_price_per_night', 'hotel_nights',
    ])
    || !isFiniteNumber(value.evidence.outbound_flight_price) || value.evidence.outbound_flight_price <= 0
    || !isFiniteNumber(value.evidence.return_flight_price) || value.evidence.return_flight_price <= 0
    || !isFiniteNumber(value.evidence.hotel_price_per_night) || value.evidence.hotel_price_per_night < 0
    || !Number.isInteger(value.evidence.hotel_nights)
    || Number(value.evidence.hotel_nights) < 0
    || ((value.evidence.hotel_nights === 0) !== (value.evidence.hotel_price_per_night === 0))
    || value.itinerary !== null
    || value.daily_geojson_maps !== null) return false;
  return true;
}

const isInitialBudgetConfirmation = (
  value: unknown,
): value is BudgetConfirmationResponse =>
  isChatBudgetConfirmation(value)
  && isNonEmptyStringArray(value.resolved_cities);

export const isUnexpiredChatBudgetConfirmation = (decision: ChatBudgetConfirmation): boolean =>
  Date.parse(decision.expires_at) > Date.now();

export interface BudgetConfirmationResponse {
  status: 'budget_confirmation_required';
  reason: 'insufficient_budget' | 'recommendation_requested';
  chat_reply: string;
  budget_assessment_id: string;

  /** Verified planning cities selected or accepted by the backend. */
  resolved_cities: string[];

  stated_budget: number | null;
  recommended_minimum_budget: number;
  base_currency: string;
  destination_currency: string;
  expires_at: string;
  evidence: BudgetEvidenceResponse;
  itinerary: null;
  daily_geojson_maps: null;
}

export interface BudgetCheckUnavailableResponse {
  status: 'budget_check_unavailable';
  reason:
    | 'provider_data_unavailable'
    | 'assessment_cache_unavailable'
    | 'assessment_expired_or_invalid'
    | 'destination_resolution_unavailable';
  chat_reply: string;
  itinerary: null;
  daily_geojson_maps: null;
}

export type PlanningUnavailableReason =
  | 'validation_failed'
  | 'provider_data_unavailable'
  | 'review_unavailable'
  | 'deadline_exhausted';

export interface PlanningUnavailableResponse {
  status: 'planning_unavailable';
  reason: PlanningUnavailableReason;
  chat_reply: string;
  retryable: true;
  itinerary: null;
  daily_geojson_maps: null;
}

export type TripSubmissionResponse =
  | FinalResponse
  | BudgetConfirmationResponse
  | BudgetCheckUnavailableResponse
  | PlanningUnavailableResponse;

// ── Itinerary data shapes (for the structured renderer) ─────────────

export interface FlightItem {
  airline?: string;
  flight_number?: string;
  airline_logo?: string;
  travel_class?: string;
  airplane?: string;
  departure_airport?: Record<string, any> | null;
  arrival_airport?: Record<string, any> | null;
  departure_time?: string;
  arrival_time?: string;
  /** Total minutes across all legs. */
  duration?: number;
  /** 0 = direct flight. */
  stops?: number;
  /** Layover airport names for connecting flights. */
  layovers?: string[];
  price?: number;
  booking_url?: string | null;
  /** Cheapest real option found, but it exceeds the category budget. */
  over_budget?: boolean;
}

export interface HotelItem {
  hotel_name?: string;
  hotel_class?: number;
  /** Guest rating (e.g. 4.4) and how many reviews it is based on. */
  overall_rating?: number | null;
  reviews?: number | null;
  description?: string;
  price_per_night?: number;
  amenities?: string[];
  check_in_time?: string;
  check_out_time?: string;
  location?: { lat?: number; lng?: number; country_code?: string | null } | null;
  image?: string;
  booking_url?: string;
  /** Cheapest real option found, but it exceeds the category budget. */
  over_budget?: boolean;
}

export interface ActivityItem {
  name?: string;
  type?: 'attraction' | 'restaurant' | string;
  description?: string | null;
  category?: string | string[] | null;
  rating?: number | null;
  address?: string;
  thumbnail?: string | null;
  suggested_time?: string | null;
  estimated_cost?: number;
  is_estimated?: boolean | null;
  order?: number;

  location?: {
    place_name?: string;
    latitude?: number;
    longitude?: number;
    country_code?: string;

    // Configured trip city owned by the backend.
    requested_city: string;

    // Actual provider-grounded administrative locality.
    verified_locality: string;

    // Present when the real locality differs from the configured
    // planning city but is still inside the permitted nearby radius.
    distance_from_requested_city_km?: number | null;
  } | null;
}

export interface RouteProfileMetric {
  distance_km?: number;
  duration_mins?: number;
}

export interface RouteInfo {
  ordered_stops?: string[];
  profiles?: Record<string, RouteProfileMetric>;
}

export interface DayItinerary {
  day: number;
  date?: string;
  flight?: FlightItem[] | null;
  hotel?: HotelItem | null;
  activities?: ActivityItem[];
  route?: RouteInfo | null;
  day_total_cost?: number;
}

// ── GeoJSON (daily maps) ────────────────────────────────────────────

export interface GeoJSONFeature {
  type: 'Feature';
  geometry: { type: 'Point' | 'LineString'; coordinates: any };
  properties: Record<string, any>;
}

export interface GeoJSONFeatureCollection {
  type: 'FeatureCollection';
  features: GeoJSONFeature[];
}

/** Error thrown for any non-2xx response, carrying the backend detail. */
export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, detail: unknown, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

// ── Core request helpers ────────────────────────────────────────────

class DuplicateJsonKeyError extends SyntaxError {}

/**
 * Walk the JSON grammar before parsing so object-member names remain visible.
 * JSON.parse necessarily discards an earlier value when an object repeats a
 * key, which would otherwise let a non-canonical payload reach our guards.
 */
function assertNoDuplicateJsonKeys(source: string): void {
  let cursor = 0;

  const fail = (): never => {
    throw new SyntaxError('Invalid JSON response.');
  };
  const skipWhitespace = () => {
    while (/\s/.test(source[cursor] ?? '')) cursor += 1;
  };
  const readString = (): string => {
    if (source[cursor] !== '"') return fail();
    const start = cursor;
    cursor += 1;
    while (cursor < source.length) {
      const character = source[cursor];
      if (character === '"') {
        cursor += 1;
        return JSON.parse(source.slice(start, cursor));
      }
      if (character === '\\') {
        cursor += 1;
        const escape = source[cursor];
        if (escape === 'u') {
          if (!/^[0-9a-fA-F]{4}$/.test(source.slice(cursor + 1, cursor + 5))) return fail();
          cursor += 5;
          continue;
        }
        if (!'"\\/bfnrt'.includes(escape ?? '')) return fail();
        cursor += 1;
        continue;
      }
      if (character.charCodeAt(0) < 0x20) return fail();
      cursor += 1;
    }
    return fail();
  };
  const readValue = (): void => {
    skipWhitespace();
    const character = source[cursor];
    if (character === '{') {
      cursor += 1;
      skipWhitespace();
      const keys = new Set<string>();
      if (source[cursor] === '}') {
        cursor += 1;
        return;
      }
      while (cursor < source.length) {
        const key = readString();
        if (keys.has(key)) {
          throw new DuplicateJsonKeyError(`Response contains duplicate JSON key "${key}".`);
        }
        keys.add(key);
        skipWhitespace();
        if (source[cursor] !== ':') return fail();
        cursor += 1;
        readValue();
        skipWhitespace();
        if (source[cursor] === '}') {
          cursor += 1;
          return;
        }
        if (source[cursor] !== ',') return fail();
        cursor += 1;
        skipWhitespace();
      }
      return fail();
    }
    if (character === '[') {
      cursor += 1;
      skipWhitespace();
      if (source[cursor] === ']') {
        cursor += 1;
        return;
      }
      while (cursor < source.length) {
        readValue();
        skipWhitespace();
        if (source[cursor] === ']') {
          cursor += 1;
          return;
        }
        if (source[cursor] !== ',') return fail();
        cursor += 1;
      }
      return fail();
    }
    if (character === '"') {
      readString();
      return;
    }
    const remainder = source.slice(cursor);
    const literal = /^(?:true|false|null)/.exec(remainder)?.[0];
    if (literal) {
      cursor += literal.length;
      return;
    }
    const number = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(remainder)?.[0];
    if (!number) return fail();
    cursor += number.length;
  };

  readValue();
  skipWhitespace();
  if (cursor !== source.length) fail();
}

async function parseResponse(res: Response, path: string): Promise<unknown> {
  const raw = await res.text();
  let parsed: unknown = null;
  if (raw) {
    try {
      assertNoDuplicateJsonKeys(raw);
      parsed = JSON.parse(raw);
    } catch (error) {
      if (error instanceof DuplicateJsonKeyError) {
        throw new ApiError(res.ok ? 502 : res.status, null, error.message);
      }
      parsed = raw;
    }
  }

  if (!res.ok) {
    const detail = (isRecord(parsed) ? parsed.detail : undefined) ?? parsed ?? res.statusText;
    const message =
      typeof detail === 'string' ? detail : `Request to ${path} failed with ${res.status}`;
    throw new ApiError(res.status, detail, message);
  }
  return parsed;
}

/**
 * Default per-request timeout. Without one, a hung backend request leaves the
 * UI stuck on the loading spinner forever ("everything freezes"). Individual
 * calls can pass a longer budget (trip planning legitimately takes minutes).
 */
const DEFAULT_TIMEOUT_MS = 60_000;
/** Chat replies: backend hard-caps the workflow at 240 s. */
const CHAT_TIMEOUT_MS = 250_000;
/** Initial trip planning: backend hard-caps the workflow at 480 s. */
const FORM_TIMEOUT_MS = 490_000;

async function fetchWithTimeout(
  path: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(`${API_BASE_URL}${path}`, { ...init, signal: controller.signal });
  } catch (networkErr) {
    if ((networkErr as any)?.name === 'AbortError') {
      throw new ApiError(
        0,
        networkErr,
        `The request timed out after ${Math.round(timeoutMs / 1000)}s. ` +
          'The server may be overloaded — please try again.',
      );
    }
    throw new ApiError(
      0,
      networkErr,
      `Cannot reach backend at ${API_BASE_URL}. Is the server running and CORS configured?`,
    );
  } finally {
    clearTimeout(timer);
  }
}

async function post(
  path: string,
  body: unknown,
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<unknown> {
  const res = await fetchWithTimeout(
    path,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-User-ID': USER_ID },
      body: JSON.stringify(body),
    },
    timeoutMs,
  );
  return parseResponse(res, path);
}

async function get(path: string, timeoutMs: number = DEFAULT_TIMEOUT_MS): Promise<unknown> {
  const res = await fetchWithTimeout(
    path,
    {
      method: 'GET',
      headers: { 'X-User-ID': USER_ID },
      // Profile and checkpoint history are mutable server state. Revalidation
      // must reach the backend so a database reset is reflected immediately.
      cache: 'no-store',
    },
    timeoutMs,
  );
  return parseResponse(res, path);
}

// ── Public API ──────────────────────────────────────────────────────

/** Persist the one-time onboarding profile (Name / Origin). */
export function submitProfile(profile: OnboardingRequest): Promise<{ status: string; user_id: string }> {
  return post('/api/profile/', profile).then((value) => {
    if (!isRecord(value) || typeof value.status !== 'string' || typeof value.user_id !== 'string') {
      throw new ApiError(502, null, 'The profile service returned an invalid response.');
    }
    return { status: value.status, user_id: value.user_id };
  });
}

/** Check whether this user has completed onboarding. */
export function getProfile(): Promise<ProfileResponse> {
  return get('/api/profile/').then((value) => {
    if (!isRecord(value)
      || typeof value.status !== 'string'
      || typeof value.onboarded !== 'boolean'
      || !isRecord(value.profile)) {
      throw new ApiError(502, null, 'The profile service returned an invalid response.');
    }
    return { status: value.status, onboarded: value.onboarded, profile: value.profile };
  });
}

/** Send a chat message on an existing session and get the AI reply. */
export function sendChatMessage(
  sessionId: string,
  userMessage: string,
  options?: ChatMessageOptions,
): Promise<ChatResponse> {
  return post(
    '/api/chat/message',
    {
      session_id: sessionId,
      user_message: userMessage,
      ...(options?.budget_action !== undefined ? { budget_action: options.budget_action } : {}),
      ...(options?.budget_assessment_id !== undefined
        ? { budget_assessment_id: options.budget_assessment_id }
        : {}),
    },
    CHAT_TIMEOUT_MS,
  ).then((value) => {
    if (!isChatResponse(value)) {
      throw new ApiError(502, null, 'The chat service returned an invalid response.');
    }
    return value;
  });
}

const isOptionalString = (value: unknown): boolean =>
  value == null || typeof value === 'string';

const isOptionalFiniteNumber = (value: unknown): boolean =>
  value == null || isFiniteNumber(value);

const isNonNegativeNumber = (value: unknown): value is number =>
  isFiniteNumber(value) && value >= 0;

const isoDateUtc = (value: unknown): number | null => {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const [year, month, day] = value.split('-').map(Number);
  const parsed = new Date(Date.UTC(year, month - 1, day));
  return parsed.getUTCFullYear() === year
    && parsed.getUTCMonth() === month - 1
    && parsed.getUTCDate() === day
    ? parsed.getTime()
    : null;
};

const isIsoDate = (value: unknown): value is string => isoDateUtc(value) !== null;

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === 'string');

const isFlightAirport = (value: unknown): boolean => value == null || (
  isRecord(value)
  && hasOnlyKeys(value, ['id', 'name', 'time'])
  && ['id', 'name', 'time'].every((key) => value[key] === undefined || typeof value[key] === 'string')
);

const isFlightItem = (value: unknown): boolean => {
  if (!isRecord(value) || !hasOnlyKeys(value, [
    'airline', 'flight_number', 'airline_logo', 'travel_class', 'airplane',
    'departure_airport', 'arrival_airport', 'departure_time', 'arrival_time',
    'duration', 'stops', 'layovers', 'price', 'booking_url', 'over_budget',
  ])) return false;
  const strings = [
    'airline', 'flight_number', 'airline_logo', 'travel_class', 'airplane',
    'departure_time', 'arrival_time',
  ];
  return strings.every((key) => value[key] === undefined || typeof value[key] === 'string')
    && (value.booking_url === undefined || isOptionalString(value.booking_url))
    && (value.duration === undefined || (Number.isInteger(value.duration) && Number(value.duration) >= 0))
    && (value.stops === undefined || (Number.isInteger(value.stops) && Number(value.stops) >= 0))
    && (value.price === undefined || isNonNegativeNumber(value.price))
    && (value.layovers === undefined || isStringArray(value.layovers))
    && (value.over_budget === undefined || typeof value.over_budget === 'boolean')
    && isFlightAirport(value.departure_airport)
    && isFlightAirport(value.arrival_airport);
};

const isHotelItem = (value: unknown): boolean => {
  if (!isRecord(value) || !hasOnlyKeys(value, [
    'hotel_name', 'hotel_class', 'overall_rating', 'reviews', 'description',
    'price_per_night', 'amenities', 'check_in_time', 'check_out_time', 'location',
    'image', 'booking_url', 'over_budget',
  ])) return false;
  const strings = [
    'hotel_name', 'description', 'check_in_time', 'check_out_time', 'image', 'booking_url',
  ];
  if (!strings.every((key) => value[key] === undefined || typeof value[key] === 'string')
    || (value.hotel_class !== undefined
      && (!isNonNegativeNumber(value.hotel_class) || value.hotel_class > 5))
    || (value.overall_rating !== undefined
      && value.overall_rating !== null
      && (!isNonNegativeNumber(value.overall_rating) || value.overall_rating > 5))
    || (value.reviews !== undefined
      && value.reviews !== null
      && (!Number.isInteger(value.reviews) || Number(value.reviews) < 0))
    || (value.price_per_night !== undefined && !isNonNegativeNumber(value.price_per_night))
    || (value.amenities !== undefined && !isStringArray(value.amenities))
    || (value.over_budget !== undefined && typeof value.over_budget !== 'boolean')) return false;
  if (value.location == null) return true;
  return isRecord(value.location)
    && hasOnlyKeys(value.location, ['lat', 'lng', 'country_code'])
    && isOptionalFiniteNumber(value.location.lat)
    && isOptionalFiniteNumber(value.location.lng)
    && (value.location.country_code == null || isCountryCode(value.location.country_code))
    && ((value.location.lat == null) === (value.location.lng == null));
};

const MAX_ACTIVITY_CITY_DISTANCE_KM = 75;

const normalizeCityKey = (value: string): string =>
  value
    .trim()
    .replace(/\s+/gu, ' ')
    .toLocaleLowerCase('en-US');


const isActivityItem = (value: unknown): value is ActivityItem => {
  if (
    !isRecord(value)
    || !hasOnlyKeys(
      value,
      [
        'name',
        'type',
        'description',
        'category',
        'rating',
        'address',
        'thumbnail',
        'suggested_time',
        'estimated_cost',
        'is_estimated',
        'order',
        'location',
      ],
      [
        'name',
        'type',
        'address',
        'estimated_cost',
        'order',
        'location',
      ],
    )
  ) {
    return false;
  }

  if (
    !isNonEmptyString(value.name)
    || (value.type !== 'attraction' && value.type !== 'restaurant')
    || !isNonEmptyString(value.address)
    || !isNonNegativeNumber(value.estimated_cost)
    || !Number.isInteger(value.order)
    || Number(value.order) < 1
    || !isOptionalString(value.description)
    || !isOptionalString(value.category)
    || !isOptionalString(value.thumbnail)
    || !isOptionalString(value.suggested_time)
    || (
      value.rating != null
      && (
        !isNonNegativeNumber(value.rating)
        || value.rating > 5
      )
    )
    || (
      value.is_estimated != null
      && typeof value.is_estimated !== 'boolean'
    )
  ) {
    return false;
  }

  const location = value.location;

  if (
    !isRecord(location)
    || !hasOnlyKeys(
      location,
      [
        'place_name',
        'country_code',
        'latitude',
        'longitude',
        'requested_city',
        'verified_locality',
        'distance_from_requested_city_km',
      ],
      [
        'place_name',
        'country_code',
        'latitude',
        'longitude',
        'requested_city',
        'verified_locality',
      ],
    )
    || !isNonEmptyString(location.place_name)
    || !isCountryCode(location.country_code)
    || !isNonEmptyString(location.requested_city)
    || !isNonEmptyString(location.verified_locality)
    || !isFiniteNumber(location.latitude)
    || !isFiniteNumber(location.longitude)
  ) {
    return false;
  }

  const requestedCityKey = normalizeCityKey(
    location.requested_city,
  );

  const verifiedLocalityKey = normalizeCityKey(
    location.verified_locality,
  );

  const localityMatchesRequestedCity =
    requestedCityKey === verifiedLocalityKey;

  const distance = location.distance_from_requested_city_km;

  const nearbyCityProof =
    distance != null
    && isNonNegativeNumber(distance)
    && distance <= MAX_ACTIVITY_CITY_DISTANCE_KM;

  /*
   * Normal case:
   *   requested_city    = Tokyo
   *   verified_locality = Tokyo
   *
   * Nearby metropolitan case:
   *   requested_city    = Tokyo
   *   verified_locality = Urayasu
   *   distance          = 15.x km
   *
   * Both are valid.
   */
  if (
    !localityMatchesRequestedCity
    && !nearbyCityProof
  ) {
    return false;
  }

  return true;
};

const ROUTE_PROFILES = ['driving', 'walking', 'cycling'] as const;

const isRouteInfo = (value: unknown): boolean => {
  if (!isRecord(value)
    || !hasOnlyKeys(value, ['ordered_stops', 'profiles'])
    || (value.ordered_stops !== undefined && !isStringArray(value.ordered_stops))
    || (value.profiles !== undefined && !isRecord(value.profiles))) return false;
  if (!isRecord(value.profiles)) return value.profiles === undefined;
  if (!hasOnlyKeys(value.profiles, ROUTE_PROFILES)) return false;
  return Object.values(value.profiles).every((metric) => metric == null || (
    isRecord(metric)
    && hasOnlyKeys(
      metric,
      ['distance_km', 'duration_mins'],
      ['distance_km', 'duration_mins'],
    )
    && isNonNegativeNumber(metric.distance_km)
    && isNonNegativeNumber(metric.duration_mins)
  ));
};

const isCompleteDay = (value: unknown, expectedDay: number): value is DayItinerary => {
  if (!isRecord(value) || !hasOnlyKeys(value, [
    'day', 'date', 'flight', 'hotel', 'activities', 'route', 'day_total_cost',
  ], ['day', 'date', 'activities', 'day_total_cost'])) return false;
  if (value.day !== expectedDay
    || !isIsoDate(value.date)
    || !isNonNegativeNumber(value.day_total_cost)
    || !Array.isArray(value.activities)
    || value.activities.length === 0
    || !value.activities.every(isActivityItem)
    || !value.activities.every((activity, index) => activity.order === index + 1)) return false;
  if (value.flight != null
    && (!Array.isArray(value.flight) || !value.flight.every(isFlightItem))) return false;
  if (value.hotel != null && !isHotelItem(value.hotel)) return false;
  return value.route == null || isRouteInfo(value.route);
};

/** Strict recursive guard shared by live and restored itinerary rendering. */
export const isCompleteItinerary = (value: unknown): value is DayItinerary[] => {
  if (!Array.isArray(value)
    || value.length === 0
    || !value.every((day, index) => isCompleteDay(day, index + 1))) return false;
  return value.every((day, index) => {
    if (index === 0) return true;
    return isoDateUtc(day.date)! - isoDateUtc(value[index - 1].date)! === 86_400_000;
  });
};

const isCoordinatePair = (value: unknown): value is [number, number] =>
  Array.isArray(value) && value.length === 2 && value.every(isFiniteNumber);

const isGeoJSONFeature = (value: unknown): value is GeoJSONFeature => {
  if (!isRecord(value)
    || !hasOnlyKeys(value, ['type', 'geometry', 'properties'], ['type', 'geometry', 'properties'])
    || value.type !== 'Feature'
    || !isRecord(value.geometry)
    || !hasOnlyKeys(value.geometry, ['type', 'coordinates'], ['type', 'coordinates'])
    || !isRecord(value.properties)) return false;
  if (value.geometry.type === 'Point') {
    return hasOnlyKeys(
      value.properties,
      ['name', 'type', 'order'],
      ['name', 'type', 'order'],
    )
      && isNonEmptyString(value.properties.name)
      && ['activity', 'attraction', 'restaurant', 'hotel', 'airport'].includes(
        String(value.properties.type),
      )
      && Number.isInteger(value.properties.order)
      && Number(value.properties.order) >= 0
      && isCoordinatePair(value.geometry.coordinates);
  }
  if (value.geometry.type === 'LineString') {
    return hasOnlyKeys(
      value.properties,
      ['type', 'profile', 'distance_km', 'duration_mins'],
      ['type', 'profile', 'distance_km', 'duration_mins'],
    )
      && value.properties.type === 'route'
      && ROUTE_PROFILES.includes(
        value.properties.profile as typeof ROUTE_PROFILES[number],
      )
      && isNonNegativeNumber(value.properties.distance_km)
      && isNonNegativeNumber(value.properties.duration_mins)
      && Array.isArray(value.geometry.coordinates)
      && value.geometry.coordinates.length >= 2
      && value.geometry.coordinates.every(isCoordinatePair);
  }
  return false;
};

const isGeoJSONFeatureCollection = (value: unknown): value is GeoJSONFeatureCollection =>
  isRecord(value)
  && hasOnlyKeys(value, ['type', 'features'], ['type', 'features'])
  && value.type === 'FeatureCollection'
  && Array.isArray(value.features)
  && value.features.every(isGeoJSONFeature);

const pointKey = (
  name: string,
  longitude: number,
  latitude: number,
  type: string,
  order: number,
): string => `${name.trim()}\u0000${longitude}\u0000${latitude}\u0000${type}\u0000${order}`;

const expectedDayPoints = (day: DayItinerary): string[] => {
  const points = (day.activities ?? []).map((activity) => pointKey(
    activity.name!,
    activity.location!.longitude!,
    activity.location!.latitude!,
    activity.type!,
    activity.order!,
  ));
  const hotel = day.hotel;
  if (hotel?.hotel_name?.trim()
    && hotel.location?.lat != null
    && hotel.location.lng != null) {
    points.push(pointKey(
      hotel.hotel_name,
      hotel.location.lng,
      hotel.location.lat,
      'hotel',
      0,
    ));
  }
  return points.sort();
};

const expectedArrivalAirportName = (day: DayItinerary): string | null => {
  const airport = day.flight?.[0]?.arrival_airport;
  if (isNonEmptyString(airport?.name)) return airport.name.trim();
  if (isNonEmptyString(airport?.id)) return `${airport.id.trim()} airport`;
  return null;
};

const isCompleteMaps = (
  value: unknown,
  itinerary: DayItinerary[],
): value is Record<string, GeoJSONFeatureCollection> => {
  if (!isRecord(value)) return false;
  const keys = Object.keys(value);
  if (keys.length !== itinerary.length
    || !keys.every((key) => /^[1-9]\d*$/.test(key))
    || !itinerary.every((day) => Object.prototype.hasOwnProperty.call(value, String(day.day)))) {
    return false;
  }
  return itinerary.every((day) => {
    const dailyMap = value[String(day.day)];
    if (!isGeoJSONFeatureCollection(dailyMap)) return false;
    const routeFeatures = dailyMap.features.filter((feature) => (
      feature.geometry.type === 'LineString'
    ));
    const expectedRouteProfiles = Object.entries(day.route?.profiles ?? {})
      .filter(([, metric]) => metric != null)
      .sort(([left], [right]) => left.localeCompare(right));
    const actualRouteProfiles = routeFeatures
      .map((feature) => [String(feature.properties.profile), feature] as const)
      .sort(([left], [right]) => left.localeCompare(right));
    if (actualRouteProfiles.length !== expectedRouteProfiles.length
      || !actualRouteProfiles.every(([profile, feature], index) => {
        const [expectedProfile, metric] = expectedRouteProfiles[index];
        return profile === expectedProfile
          && isNonNegativeNumber(feature.properties.distance_km)
          && feature.properties.distance_km === metric?.distance_km
          && isNonNegativeNumber(feature.properties.duration_mins)
          && feature.properties.duration_mins === metric?.duration_mins;
      })) return false;
    const orderedRoutePoints = dailyMap.features.filter((feature) => (
      feature.geometry.type === 'Point'
    ));
    const airportPoints = orderedRoutePoints.filter((feature) => (
      feature.properties.type === 'airport'
    ));
    if (airportPoints.length > 1) return false;
    if (airportPoints.length === 1) {
      const expectedAirport = day.day === 1 ? expectedArrivalAirportName(day) : null;
      const airport = airportPoints[0];
      if (expectedAirport == null
        || airport.properties.name.trim() !== expectedAirport
        || airport.properties.order !== 0) return false;
    }
    const expectedStops = day.route?.ordered_stops ?? [];
    const actualStops = orderedRoutePoints.map((feature) => feature.properties.name?.trim() ?? '');
    if (day.route != null && (
      expectedStops.length !== actualStops.length
      || expectedStops.some((stop, index) => stop.trim() !== actualStops[index])
    )) return false;
    if (routeFeatures.length > 0) {
      if (orderedRoutePoints.length < 2) return false;
      const expectedStart = orderedRoutePoints[0].geometry.coordinates;
      const expectedEnd = orderedRoutePoints[orderedRoutePoints.length - 1].geometry.coordinates;
      if (routeFeatures.some((feature) => {
        const coordinates = feature.geometry.coordinates;
        const start = coordinates[0];
        const end = coordinates[coordinates.length - 1];
        return start[0] !== expectedStart[0]
          || start[1] !== expectedStart[1]
          || end[0] !== expectedEnd[0]
          || end[1] !== expectedEnd[1];
      })) return false;
    }
    const actualPoints = dailyMap.features
      .filter((feature) => feature.geometry.type === 'Point' && feature.properties.type !== 'airport')
      .map((feature) => pointKey(
        feature.properties.name?.trim() ?? '',
        feature.geometry.coordinates[0],
        feature.geometry.coordinates[1],
        String(feature.properties.type),
        Number(feature.properties.order),
      ))
      .sort();
    if (actualPoints.some((key) => key.startsWith('\u0000'))) return false;
    const expectedPoints = expectedDayPoints(day);
    return actualPoints.length === expectedPoints.length
      && actualPoints.every((point, index) => point === expectedPoints[index]);
  });
};

const BUDGET_CATEGORIES = [
  'transportation', 'accommodation', 'food', 'activity', 'shopping', 'emergency_fund',
] as const;

const isBudgetAllocation = (
  value: unknown,
  totalBudget: number,
): value is Record<string, number> => {
  if (!isRecord(value) || !hasOnlyKeys(value, BUDGET_CATEGORIES)) return false;
  let allocated = 0;
  for (const amount of Object.values(value)) {
    if (!isNonNegativeNumber(amount)) return false;
    allocated += amount;
  }
  return Math.abs(allocated - totalBudget) <= 0.05;
};

export const isBudgetInfo = (value: unknown): value is BudgetInfo =>
  isRecord(value)
  && hasOnlyKeys(value, ['total', 'currency', 'allocation'], ['total', 'currency', 'allocation'])
  && isNonNegativeNumber(value.total)
  && isValidCurrency(value.currency)
  && isBudgetAllocation(value.allocation, value.total);

const isCompletePlan = (
  itinerary: unknown,
  maps: unknown,
  destinationCountryCode: unknown,
): itinerary is DayItinerary[] => {
  if (!isCountryCode(destinationCountryCode) || !isCompleteItinerary(itinerary)) return false;
  if (!itinerary.every((day) => (
    (day.activities ?? []).every((activity) => (
      activity.location?.country_code === destinationCountryCode
    ))
    && (day.hotel?.location?.country_code == null
      || day.hotel.location.country_code === destinationCountryCode)
  ))) return false;
  return isCompleteMaps(maps, itinerary);
};

export interface CompleteItinerarySnapshot {
  itinerary: DayItinerary[];
  maps: Record<string, GeoJSONFeatureCollection>;
  budget: BudgetInfo;
  destination_country_code?: string;
}

export const isCompleteItinerarySnapshot = (
  value: unknown,
): value is CompleteItinerarySnapshot => {
  if (!isRecord(value)
    || !hasOnlyKeys(value, [
      'itinerary', 'maps', 'budget', 'destination_country_code',
    ], ['itinerary', 'maps', 'budget'])
    || !isCompleteItinerary(value.itinerary)
    || !isBudgetInfo(value.budget)) return false;
  const countries = new Set(value.itinerary.flatMap((day) => [
    ...(day.activities ?? []).map((activity) => activity.location!.country_code!),
    ...(day.hotel?.location?.country_code ? [day.hotel.location.country_code] : []),
  ]));
  if (countries.size !== 1) return false;
  const inferredCountry = [...countries][0];
  let destinationCountry = inferredCountry;
  if (Object.prototype.hasOwnProperty.call(value, 'destination_country_code')) {
    if (!isCountryCode(value.destination_country_code)) return false;
    destinationCountry = value.destination_country_code;
  }
  return isCompletePlan(value.itinerary, value.maps, destinationCountry);
};

export function hasCompleteLiveSuccessSnapshot(
  response: unknown,
): response is ChatSuccessResponse & { itinerary_modified: true } {
  return isRecord(response)
    && hasOnlyKeys(response, [
      'status', 'chat_reply', 'draft_itinerary', 'daily_map_info',
      'destination_country_code', 'itinerary_modified', 'total_budget', 'currency',
      'budget_allocation', 'budget_confirmation',
    ], [
      'status', 'chat_reply', 'draft_itinerary', 'daily_map_info',
      'destination_country_code', 'itinerary_modified', 'total_budget', 'currency',
      'budget_allocation', 'budget_confirmation',
    ])
    && response.status === 'success'
    && isSafeAssistantContent(response.chat_reply)
    && response.itinerary_modified === true
    && isCompletePlan(
      response.draft_itinerary,
      response.daily_map_info,
      response.destination_country_code,
    )
    && isNonNegativeNumber(response.total_budget)
    && isValidCurrency(response.currency)
    && isBudgetAllocation(response.budget_allocation, response.total_budget)
    && response.budget_confirmation === null;
}

export function hasCompleteInitialSuccessSnapshot(
  response: unknown,
): response is FinalResponse {
  return isRecord(response)
    && hasOnlyKeys(response, [
      'status', 'chat_reply', 'itinerary', 'daily_geojson_maps',
      'destination_country_code', 'resolved_cities',
      'total_budget', 'currency', 'budget_allocation', 'session_id',
    ], [
      'status', 'chat_reply', 'itinerary', 'daily_geojson_maps',
      'destination_country_code', 'resolved_cities',
      'total_budget', 'currency', 'budget_allocation', 'session_id',
    ])
    && response.status === 'success'
    && isCompletePlan(
      response.itinerary,
      response.daily_geojson_maps,
      response.destination_country_code,
    )
    && isNonEmptyStringArray(response.resolved_cities)
    && isNonNegativeNumber(response.total_budget)
    && isValidCurrency(response.currency)
    && isBudgetAllocation(response.budget_allocation, response.total_budget)
    && isNonEmptyString(response.session_id)
    && isSafeAssistantContent(response.chat_reply);
}

const INTERNAL_ASSISTANT_MARKERS = [
  'accepted_plan_snapshot',
  'candidate_plan',
  'daily_map_info',
  'draft_itinerary',
  'planning_outcome',
  'tool_calls',
  'internal state',
  'working itinerary summary',
] as const;

const containsJsonContainer = (content: string): boolean => {
  for (let start = 0; start < content.length; start += 1) {
    const opener = content[start];
    if (opener !== '{' && opener !== '[') continue;
    const stack: string[] = [];
    let inString = false;
    let escaped = false;
    for (let index = start; index < content.length; index += 1) {
      const character = content[index];
      if (inString) {
        if (escaped) escaped = false;
        else if (character === '\\') escaped = true;
        else if (character === '"') inString = false;
        continue;
      }
      if (character === '"') {
        inString = true;
        continue;
      }
      if (character === '{' || character === '[') stack.push(character);
      else if (character === '}' || character === ']') {
        const expected = character === '}' ? '{' : '[';
        if (stack.pop() !== expected) break;
        if (stack.length === 0) {
          try {
            const parsed = JSON.parse(content.slice(start, index + 1));
            if (parsed !== null && typeof parsed === 'object') {
              if (!Array.isArray(parsed)) return true;
              const wholeReply = content.slice(0, start).trim() === ''
                && content.slice(index + 1).trim() === '';
              const isSingleIntegerCitation = parsed.length === 1
                && Number.isInteger(parsed[0]);
              if (wholeReply || !isSingleIntegerCitation) return true;
            }
          } catch {
            // This opener is ordinary prose rather than valid JSON.
          }
          break;
        }
      }
    }
  }
  return false;
};

export const isSafeAssistantContent = (content: unknown): content is string => {
  if (!isNonEmptyString(content)) return false;
  const folded = content.toLocaleLowerCase();
  return !containsJsonContainer(content)
    && !INTERNAL_ASSISTANT_MARKERS.some((marker) => folded.includes(marker));
};

const isEmptyRecord = (value: unknown): value is Record<string, never> =>
  isRecord(value) && Object.keys(value).length === 0;

const isPlanningUnavailableReason = (value: unknown): value is PlanningUnavailableReason =>
  value === 'validation_failed'
  || value === 'provider_data_unavailable'
  || value === 'review_unavailable'
  || value === 'deadline_exhausted';

const isInitialPlanningUnavailable = (
  value: unknown,
): value is PlanningUnavailableResponse => isRecord(value)
  && hasOnlyKeys(value, [
    'status', 'reason', 'chat_reply', 'retryable', 'itinerary', 'daily_geojson_maps',
  ], [
    'status', 'reason', 'chat_reply', 'retryable', 'itinerary', 'daily_geojson_maps',
  ])
  && value.status === 'planning_unavailable'
  && isPlanningUnavailableReason(value.reason)
  && isSafeAssistantContent(value.chat_reply)
  && value.retryable === true
  && value.itinerary === null
  && value.daily_geojson_maps === null;

const isInitialBudgetUnavailable = (
  value: unknown,
): value is BudgetCheckUnavailableResponse => isRecord(value)
  && hasOnlyKeys(value, [
    'status', 'reason', 'chat_reply', 'itinerary', 'daily_geojson_maps',
  ], ['status', 'reason', 'chat_reply', 'itinerary', 'daily_geojson_maps'])
  && value.status === 'budget_check_unavailable'
  && (
    value.reason === 'provider_data_unavailable'
    || value.reason === 'assessment_cache_unavailable'
    || value.reason === 'assessment_expired_or_invalid'
    || value.reason === 'destination_resolution_unavailable'
  )
  && isSafeAssistantContent(value.chat_reply)
  && value.itinerary === null
  && value.daily_geojson_maps === null;

export const isTripSubmissionResponse = (value: unknown): value is TripSubmissionResponse =>
  hasCompleteInitialSuccessSnapshot(value)
  || isInitialBudgetConfirmation(value)
  || isInitialBudgetUnavailable(value)
  || isInitialPlanningUnavailable(value);

const CHAT_PLAN_FREE_KEYS = [
  'status', 'chat_reply', 'draft_itinerary', 'daily_map_info', 'itinerary_modified',
  'total_budget', 'currency', 'budget_allocation', 'budget_confirmation',
] as const;

const isChatPlanFreeBase = (value: Record<string, unknown>): boolean =>
  Array.isArray(value.draft_itinerary)
  && value.draft_itinerary.length === 0
  && isEmptyRecord(value.daily_map_info)
  && value.itinerary_modified === false
  && value.total_budget === 0
  && value.currency === ''
  && isEmptyRecord(value.budget_allocation)
  && isSafeAssistantContent(value.chat_reply);

export const isChatResponse = (value: unknown): value is ChatResponse => {
  if (!isRecord(value) || typeof value.status !== 'string') return false;
  if (value.status === 'success') {
    return hasOnlyKeys(value, [
      'status', 'chat_reply', 'draft_itinerary', 'daily_map_info',
      'destination_country_code', 'itinerary_modified', 'total_budget', 'currency',
      'budget_allocation', 'budget_confirmation',
    ], [
      'status', 'chat_reply', 'draft_itinerary', 'daily_map_info',
      'destination_country_code', 'itinerary_modified', 'total_budget', 'currency',
      'budget_allocation', 'budget_confirmation',
    ])
      && isSafeAssistantContent(value.chat_reply)
      && typeof value.itinerary_modified === 'boolean'
      && isCompletePlan(
        value.draft_itinerary,
        value.daily_map_info,
        value.destination_country_code,
      )
      && isNonNegativeNumber(value.total_budget)
      && isValidCurrency(value.currency)
      && isBudgetAllocation(value.budget_allocation, value.total_budget)
      && value.budget_confirmation === null;
  }
  if (value.status === 'planning_unavailable') {
    return hasOnlyKeys(value, [...CHAT_PLAN_FREE_KEYS, 'reason', 'retryable'], [
      ...CHAT_PLAN_FREE_KEYS, 'reason', 'retryable',
    ])
      && isChatPlanFreeBase(value)
      && isPlanningUnavailableReason(value.reason)
      && value.retryable === true
      && value.budget_confirmation === null;
  }
  if (value.status === 'budget_confirmation_required') {
    return hasOnlyKeys(value, CHAT_PLAN_FREE_KEYS, CHAT_PLAN_FREE_KEYS)
      && isChatPlanFreeBase(value)
      && isChatBudgetConfirmation(value.budget_confirmation);
  }
  if (value.status === 'budget_check_unavailable') {
    return hasOnlyKeys(value, CHAT_PLAN_FREE_KEYS, CHAT_PLAN_FREE_KEYS)
      && isChatPlanFreeBase(value)
      && value.budget_confirmation === null;
  }
  return false;
};

export const isChatHistoryResponse = (value: unknown): value is ChatHistoryResponse => {
  if (!isRecord(value)
    || !hasOnlyKeys(value, ['status', 'sessions'], ['status', 'sessions'])
    || value.status !== 'success'
    || !Array.isArray(value.sessions)) return false;
  return value.sessions.every((session) => isRecord(session)
    && hasOnlyKeys(session, ['id', 'title', 'destination', 'updated_at', 'messages'], [
      'id', 'title', 'destination', 'updated_at', 'messages',
    ])
    && isNonEmptyString(session.id)
    && typeof session.title === 'string'
    && typeof session.destination === 'string'
    && typeof session.updated_at === 'string'
    && Number.isFinite(Date.parse(session.updated_at))
    && Array.isArray(session.messages)
    && session.messages.every((message) => {
      if (!isRecord(message)
        || !hasOnlyKeys(message, [
        'role', 'content', 'itinerary', 'maps', 'budget', 'budget_confirmation',
        'destination_country_code',
        ], ['role', 'content'])
        || (message.role !== 'user' && message.role !== 'ai')
        || typeof message.content !== 'string'
        || (message.budget_confirmation !== undefined
          && message.budget_confirmation !== null
          && !isChatBudgetConfirmation(message.budget_confirmation))) return false;
      if (message.role === 'ai' && !isSafeAssistantContent(message.content)) return false;

      const hasItinerarySnapshot = ['itinerary', 'maps', 'budget', 'destination_country_code']
        .some((key) => Object.prototype.hasOwnProperty.call(message, key));
      if (!hasItinerarySnapshot) return true;
      return isCompleteItinerarySnapshot({
        itinerary: message.itinerary,
        maps: message.maps,
        budget: message.budget,
        ...(Object.prototype.hasOwnProperty.call(message, 'destination_country_code')
          ? { destination_country_code: message.destination_country_code }
          : {}),
      });
    }));
};

/** Load the current user's persisted conversations from Supabase. */
export function getChatHistory(limit: number = 10): Promise<ChatHistoryResponse> {
  return get(`/api/chat/history?limit=${limit}`).then((value) => {
    if (!isChatHistoryResponse(value)) {
      throw new ApiError(502, null, 'Saved trips returned an invalid response.');
    }
    return value;
  });
}

/** Check budget first; only a successful result contains an itinerary/session. */
export function submitTripForm(form: TripFormRequest): Promise<TripSubmissionResponse> {
  return post('/api/form/submit', form, FORM_TIMEOUT_MS).then((value) => {
    if (!isTripSubmissionResponse(value)) {
      throw new ApiError(502, null, 'The planning service returned an invalid response.');
    }
    return value;
  });
}

/** GET /api/geocode response — found:false means "use your fallback". */
export interface GeocodeResponse {
  status: string;
  found: boolean;
  name?: string;
  address?: string;
  lat?: number;
  lng?: number;
}

/**
 * Resolve a place name/address to exact coordinates (Google Maps data via
 * the backend). `lat`/`lng` optionally bias the search near a known area.
 */
export function geocodePlace(query: string, lat?: number, lng?: number): Promise<GeocodeResponse> {
  let path = `/api/geocode/?q=${encodeURIComponent(query)}`;
  if (typeof lat === 'number' && typeof lng === 'number') {
    path += `&lat=${lat}&lng=${lng}`;
  }
  return get(path).then((value) => {
    if (!isRecord(value)) {
      throw new ApiError(
        502,
        null,
        'The geocoding service returned an invalid response.',
      );
    }

    const statusValue = value.status;
    const foundValue = value.found;

    if (typeof statusValue !== 'string' || typeof foundValue !== 'boolean') {
      throw new ApiError(
        502,
        null,
        'The geocoding service returned an invalid response.',
      );
    }

    const result: GeocodeResponse = {
      status: statusValue,
      found: foundValue,
    };

    if (value.name !== undefined) {
      if (typeof value.name !== 'string') {
        throw new ApiError(
          502,
          null,
          'The geocoding service returned an invalid response.',
        );
      }
      result.name = value.name;
    }

    if (value.address !== undefined) {
      if (typeof value.address !== 'string') {
        throw new ApiError(
          502,
          null,
          'The geocoding service returned an invalid response.',
        );
      }
      result.address = value.address;
    }

    if (value.lat !== undefined) {
      if (!isFiniteNumber(value.lat)) {
        throw new ApiError(
          502,
          null,
          'The geocoding service returned an invalid response.',
        );
      }
      result.lat = value.lat;
    }

    if (value.lng !== undefined) {
      if (!isFiniteNumber(value.lng)) {
        throw new ApiError(
          502,
          null,
          'The geocoding service returned an invalid response.',
        );
      }
      result.lng = value.lng;
    }

    return result;
  });
}

/** Simple GET liveness probe against /health. */
export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

/** Generate a fresh session id for a new conversation thread. */
export function newSessionId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `sess-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export const apiConfig = { baseUrl: API_BASE_URL, userId: USER_ID, hasMapbox: !!MAPBOX_TOKEN };
