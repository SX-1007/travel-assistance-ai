/**
 * ItineraryView.tsx — Structured renderer for the planner's itinerary data.
 *
 * Renders the backend `draft_itinerary` (+ `daily_map_info` GeoJSON) as
 * first-class UI: per-day flight, hotel card, attraction/restaurant cards, and
 * a Mapbox static map showing all stops + the route. Users can toggle the route
 * travel profile (driving / walking / cycling).
 *
 * Place costs are AI estimates (flagged `is_estimated`) — an "estimated" note is
 * shown wherever they appear, per the backend contract.
 */
import React, { useState } from 'react';
import {
  Plane, Hotel, Utensils, Camera, Star, MapPin, Clock, Route as RouteIcon,
  ExternalLink, Car, Footprints, Bike, Info, CalendarDays, Wallet,
  AlertTriangle,
} from 'lucide-react';
import type {
  DayItinerary, GeoJSONFeatureCollection, ActivityItem, FlightItem, HotelItem,
  BudgetInfo,
} from '../../lib/api';
import { apiConfig, isCompleteItinerarySnapshot } from '../../lib/api';
import { buildStaticMapUrl } from '../../lib/mapStatic';

const PROFILE_META: Record<string, { icon: React.ElementType; label: string }> = {
  driving: { icon: Car, label: 'Driving' },
  walking: { icon: Footprints, label: 'Walking' },
  cycling: { icon: Bike, label: 'Cycling' },
};

const fmtMoney = (v: any): string =>
  v === null || v === undefined || v === '' ? '—' : `${Number(v).toLocaleString()}`;

/** Small red badge for items that exceed their category allocation. */
const OverBudgetBadge = () => (
  <span className="inline-flex items-center gap-1 rounded-full bg-red-100 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-red-600">
    <AlertTriangle size={10} /> Over budget
  </span>
);

// ── Budget allocation summary ───────────────────────────────────────
const BUDGET_LABELS: Record<string, string> = {
  transportation: 'Transportation',
  accommodation: 'Accommodation',
  food: 'Food',
  activity: 'Activities',
  shopping: 'Shopping',
  emergency_fund: 'Emergency fund',
};

const BudgetSummary = ({ budget }: { budget: BudgetInfo }) => {
  const entries = Object.entries(budget.allocation || {}).filter(
    // "flight" is an internal alias of transportation on old sessions.
    ([key, value]) => key !== 'flight' && value != null,
  );
  if (entries.length === 0 && !budget.total) return null;
  const currency = budget.currency || '';
  return (
    <div className="overflow-hidden rounded-2xl border border-blue-200 bg-white shadow-sm">
      <div className="flex items-center justify-between bg-blue-50 px-4 py-2.5">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-blue-700">
          <Wallet size={14} /> Budget allocation
        </div>
        {!!budget.total && (
          <div className="text-sm font-bold text-blue-700">
            {fmtMoney(budget.total)} <span className="text-xs font-medium">{currency} total</span>
          </div>
        )}
      </div>
      {entries.length > 0 && (
        <div className="grid grid-cols-2 gap-2 p-3 sm:grid-cols-3">
          {entries.map(([key, value]) => {
            const pct = budget.total > 0 ? Math.round((Number(value) / budget.total) * 100) : null;
            return (
              <div key={key} className="rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-2">
                <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                  {BUDGET_LABELS[key] ?? key}
                </div>
                <div className="text-sm font-bold text-slate-800">
                  {fmtMoney(value)} <span className="text-[10px] font-medium text-slate-500">{currency}</span>
                  {pct != null && (
                    <span className="ml-1 text-[10px] font-medium text-slate-400">({pct}%)</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

// ── Flight ──────────────────────────────────────────────────────────

/** Minutes → "7h 35m" (null when unknown/zero). */
const fmtDuration = (mins: any): string | null => {
  const m = Number(mins);
  if (!Number.isFinite(m) || m <= 0) return null;
  const h = Math.floor(m / 60);
  const r = Math.round(m % 60);
  return h > 0 ? `${h}h ${r}m` : `${r}m`;
};

/** "2026-08-01 09:15" → { date: '2026-08-01', time: '09:15' }. */
const splitDateTime = (s?: string): { date: string; time: string } => {
  const str = (s || '').trim();
  const m = str.match(/^(\d{4}-\d{2}-\d{2})[T ](.+)$/);
  return m ? { date: m[1], time: m[2] } : { date: '', time: str };
};

const AirportSide = ({
  airport,
  when,
  align,
}: {
  airport?: Record<string, any> | null;
  when?: string;
  align: 'left' | 'right';
}) => {
  const { date, time } = splitDateTime(when);
  const cls = align === 'right' ? 'text-right' : 'text-left';
  return (
    <div className={`min-w-0 ${cls}`}>
      <div className="text-lg font-bold leading-tight text-slate-900">{time || '—'}</div>
      <div className="text-xs font-semibold text-sky-700">{airport?.id || ''}</div>
      {airport?.name && (
        <div className="truncate text-[11px] text-slate-500" title={airport.name}>
          {airport.name}
        </div>
      )}
      {date && <div className="text-[10px] text-slate-400">{date}</div>}
    </div>
  );
};

type FlightLabel = 'Outbound flight' | 'Return flight' | 'Flight';

const FlightCard = ({
  flight,
  label,
}: {
  flight: FlightItem;
  label: FlightLabel;
}) => {
  const duration = fmtDuration(flight.duration);
  const stops = flight.stops ?? 0;
  const stopLabel =
    stops === 0
      ? 'Direct'
      : `${stops} stop${stops > 1 ? 's' : ''}` +
        (flight.layovers && flight.layovers.length > 0
          ? ` · via ${flight.layovers.join(', ')}`
          : '');
  const bookingLabel = label === 'Outbound flight'
    ? 'Book outbound flight'
    : label === 'Return flight'
      ? 'Book return flight'
      : 'Book flight';
  return (
    <div
      role="group"
      aria-label={`${label} ticket`}
      className="rounded-xl border border-sky-200 bg-sky-50/40 p-3"
    >
      {/* Header: label + price */}
      <div className="mb-2 flex items-center gap-2">
        <span className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-sky-700">
          <Plane size={14} /> {label}
        </span>
        {flight.over_budget && <OverBudgetBadge />}
        <span className="ml-auto text-base font-bold text-sky-700">{fmtMoney(flight.price)}</span>
      </div>

      {/* Airline row */}
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm">
        {flight.airline_logo && (
          <img
            src={flight.airline_logo}
            alt=""
            className="h-5 w-5 shrink-0 object-contain"
            onError={(e) => ((e.target as HTMLImageElement).style.display = 'none')}
          />
        )}
        <span className="font-semibold text-slate-800">{flight.airline || 'Airline'}</span>
        {flight.flight_number && (
          <span className="text-xs text-slate-500">{flight.flight_number}</span>
        )}
        {flight.travel_class && (
          <span className="rounded bg-sky-100 px-1.5 py-0.5 text-[10px] font-medium text-sky-700">
            {flight.travel_class}
          </span>
        )}
        {flight.airplane && <span className="text-[10px] text-slate-400">{flight.airplane}</span>}
      </div>

      {/* Departure → duration/stops → arrival */}
      <div className="mt-2 flex items-center gap-3">
        <AirportSide airport={flight.departure_airport} when={flight.departure_time} align="left" />
        <div className="flex min-w-0 flex-1 flex-col items-center px-1">
          {duration && (
            <div className="flex items-center gap-1 text-[10px] font-medium text-slate-500">
              <Clock size={10} /> {duration}
            </div>
          )}
          <div className="relative my-1 h-px w-full bg-sky-300">
            <Plane
              size={12}
              className="absolute -top-[5px] left-1/2 -translate-x-1/2 fill-sky-50 text-sky-500"
            />
          </div>
          <div
            className={`truncate text-[10px] font-medium ${stops === 0 ? 'text-emerald-600' : 'text-amber-600'}`}
            title={stopLabel}
          >
            {stopLabel}
          </div>
        </div>
        <AirportSide airport={flight.arrival_airport} when={flight.arrival_time} align="right" />
      </div>

      {flight.booking_url && (
        <a
          href={flight.booking_url}
          aria-label={bookingLabel}
          target="_blank"
          rel="noreferrer"
          className="mt-3 inline-flex items-center gap-1.5 rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-sky-700"
        >
          Book flight <ExternalLink size={12} />
        </a>
      )}
    </div>
  );
};

// ── Hotel ───────────────────────────────────────────────────────────
const HotelCard = ({ hotel }: { hotel: HotelItem }) => (
  <div className="overflow-hidden rounded-xl border border-indigo-200 bg-white">
    <div className="flex items-center gap-2 bg-indigo-50 px-3 py-2 text-xs font-bold uppercase tracking-wider text-indigo-700">
      <Hotel size={14} /> Accommodation
      {hotel.over_budget && <OverBudgetBadge />}
    </div>
    <div className="flex gap-3 p-3">
      {hotel.image ? (
        <img
          src={hotel.image}
          alt={hotel.hotel_name || 'Hotel'}
          className="h-20 w-20 shrink-0 rounded-lg object-cover"
          onError={(e) => ((e.target as HTMLImageElement).style.display = 'none')}
        />
      ) : null}
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
          <h5 className="truncate font-bold text-slate-900">{hotel.hotel_name || 'Hotel'}</h5>
          {!!hotel.hotel_class && (
            <span className="rounded bg-indigo-50 px-1.5 py-0.5 text-[10px] font-medium text-indigo-700">
              {hotel.hotel_class}-star
            </span>
          )}
          {hotel.overall_rating != null && (
            <span className="flex items-center gap-0.5 text-xs text-amber-500">
              <Star size={12} className="fill-current" /> {hotel.overall_rating}
              {hotel.reviews != null && (
                <span className="text-[10px] text-slate-400">
                  ({Number(hotel.reviews).toLocaleString()} reviews)
                </span>
              )}
            </span>
          )}
        </div>
        {hotel.description && (
          <p className="mt-0.5 line-clamp-2 text-xs text-slate-500">{hotel.description}</p>
        )}
        <div className="mt-1 text-sm font-semibold text-indigo-700">
          {fmtMoney(hotel.price_per_night)} <span className="text-xs font-normal text-slate-500">/ night</span>
        </div>
        {(hotel.check_in_time || hotel.check_out_time) && (
          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-slate-500">
            {hotel.check_in_time && (
              <span className="flex items-center gap-1">
                <Clock size={10} /> Check-in {hotel.check_in_time}
              </span>
            )}
            {hotel.check_out_time && (
              <span className="flex items-center gap-1">
                <Clock size={10} /> Check-out {hotel.check_out_time}
              </span>
            )}
          </div>
        )}
        {hotel.amenities && hotel.amenities.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1">
            {hotel.amenities.slice(0, 6).map((a, i) => (
              <span key={i} className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-600">
                {a}
              </span>
            ))}
            {hotel.amenities.length > 6 && (
              <span className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-400">
                +{hotel.amenities.length - 6} more
              </span>
            )}
          </div>
        )}
        {hotel.booking_url && (
          <a
            href={hotel.booking_url}
            target="_blank"
            rel="noreferrer"
            className="mt-2 inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-indigo-700"
          >
            Book hotel <ExternalLink size={12} />
          </a>
        )}
      </div>
    </div>
  </div>
);

// ── Activity (attraction / restaurant) ──────────────────────────────
// NOTE: full static class strings (no `bg-${accent}` interpolation) so the
// Tailwind JIT actually emits them.
const ACT_THEME = {
  attraction: {
    border: 'border-emerald-200',
    badge: 'bg-emerald-100 text-emerald-700',
    iconBox: 'bg-emerald-50 text-emerald-500',
    tag: 'bg-emerald-50 text-emerald-700',
  },
  restaurant: {
    border: 'border-amber-200',
    badge: 'bg-amber-100 text-amber-700',
    iconBox: 'bg-amber-50 text-amber-500',
    tag: 'bg-amber-50 text-amber-700',
  },
} as const;

const ActivityCard = ({ act, index }: { act: ActivityItem; index: number }) => {
  const isFood = act.type === 'restaurant';
  const Icon = isFood ? Utensils : Camera;
  const t = isFood ? ACT_THEME.restaurant : ACT_THEME.attraction;
  const [imageFailed, setImageFailed] = useState(false);
  const hasImage = Boolean(act.thumbnail && !imageFailed);
  return (
    <div className={`flex gap-3 rounded-xl border ${t.border} bg-white p-3`}>
      <div className="flex flex-col items-center">
        <span
          className={`flex h-6 w-6 items-center justify-center rounded-full ${t.badge} text-[11px] font-bold`}
        >
          {index}
        </span>
      </div>
      {hasImage ? (
        <img
          src={act.thumbnail ?? undefined}
          alt={act.name || ''}
          className="h-16 w-16 shrink-0 rounded-lg object-cover"
          onError={() => setImageFailed(true)}
        />
      ) : (
        <div
          aria-label={`No photo available for ${act.name || 'this place'}`}
          className={`flex h-16 w-16 shrink-0 items-center justify-center rounded-lg ${t.iconBox}`}
        >
          <Icon size={22} />
        </div>
      )}
      <div className="min-w-0 flex-1">
        <div className="flex items-start gap-2">
          <h5 className="min-w-0 flex-1 break-words font-semibold text-slate-900">{act.name || 'Place'}</h5>
          {act.rating != null && (
            <span className="flex shrink-0 items-center gap-0.5 text-xs text-amber-500">
              <Star size={11} className="fill-current" /> {act.rating}
            </span>
          )}
        </div>
        <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-slate-500">
          <span className={`rounded px-1.5 py-0.5 font-medium ${t.tag}`}>
            {isFood ? 'Restaurant' : 'Attraction'}
          </span>
          {act.suggested_time && (
            <span className="flex items-center gap-0.5">
              <Clock size={10} /> {act.suggested_time}
            </span>
          )}
          {act.address && (
            <span className="flex max-w-full items-start gap-0.5">
              <MapPin size={10} className="mt-px shrink-0" />
              <span className="break-words">{act.address}</span>
            </span>
          )}
        </div>
        <div className="mt-1 text-sm font-semibold text-slate-700">
          {fmtMoney(act.estimated_cost)}
          {act.is_estimated && (
            <span className="ml-1 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-normal text-slate-500">
              est.
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

// ── Route + map ─────────────────────────────────────────────────────
const RouteMap = ({
  route,
  fc,
}: {
  route: DayItinerary['route'];
  fc: GeoJSONFeatureCollection | undefined;
}) => {
  const profiles = route?.profiles ? Object.keys(route.profiles) : [];
  const defaultProfile = profiles.includes('driving') ? 'driving' : profiles[0] || 'driving';
  const [profile, setProfile] = useState(defaultProfile);

  const mapUrl = buildStaticMapUrl(fc, { profile });
  const metric = route?.profiles?.[profile];

  return (
    <div className="overflow-hidden rounded-xl border border-emerald-200 bg-white">
      <div className="flex items-center gap-2 bg-emerald-50 px-3 py-2 text-xs font-bold uppercase tracking-wider text-emerald-700">
        <RouteIcon size={14} /> Daily Route &amp; Map
      </div>

      {/* Profile toggle */}
      {profiles.length > 0 && (
        <div className="flex flex-wrap gap-1.5 px-3 pt-3">
          {profiles.map((p) => {
            const meta = PROFILE_META[p] ?? { icon: RouteIcon, label: p };
            const M = meta.icon;
            const active = p === profile;
            return (
              <button
                key={p}
                onClick={() => setProfile(p)}
                className={
                  'flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-semibold transition-colors ' +
                  (active
                    ? 'bg-emerald-600 text-white'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200')
                }
              >
                <M size={12} /> {meta.label}
                {route?.profiles?.[p]?.distance_km != null && (
                  <span className={active ? 'opacity-90' : 'text-slate-400'}>
                    {route.profiles[p].distance_km}km
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}

      <div className="p-3">
        {mapUrl ? (
          <img src={mapUrl} alt="Day route map" className="w-full rounded-lg border border-slate-200" />
        ) : (
          <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50 p-4 text-center text-xs text-slate-500">
            {apiConfig.hasMapbox
              ? 'No mappable locations for this day yet.'
              : 'Set VITE_MAPBOX_TOKEN in .env.local to render the interactive day map.'}
          </div>
        )}

        {route?.ordered_stops && route.ordered_stops.length > 0 && (
          <div className="mt-2 text-xs text-slate-600">
            <span className="font-semibold text-slate-500">Path: </span>
            {route.ordered_stops.join('  →  ')}
          </div>
        )}
        {metric && (
          <div className="mt-1 flex gap-4 text-xs text-slate-500">
            {metric.distance_km != null && <span>Distance: <b>{metric.distance_km} km</b></span>}
            {metric.duration_mins != null && <span>Time: <b>{metric.duration_mins} min</b></span>}
          </div>
        )}
      </div>
    </div>
  );
};

// ── Day + top-level ─────────────────────────────────────────────────
const DayBlock = ({
  day,
  fc,
  flightLabel,
}: {
  day: DayItinerary;
  fc: GeoJSONFeatureCollection | undefined;
  flightLabel: FlightLabel;
}) => {
  const flight = Array.isArray(day.flight) ? day.flight[0] : (day.flight as any);
  // Old agent edits may contain ad-hoc partial dictionaries. Do not present
  // those as verified places; valid backend activities always carry a name and
  // an explicit attraction/restaurant type.
  const activities = (day.activities || []).filter(
    (activity) => Boolean(
      activity?.name?.trim()
      && (activity.type === 'attraction' || activity.type === 'restaurant'),
    ),
  );
  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center gap-3 border-b border-slate-100 px-5 py-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-blue-600 text-sm font-bold text-white">
          {day.day}
        </div>
        <div className="min-w-0">
          <h3 className="font-bold text-slate-900">Day {day.day}</h3>
          {day.date && (
            <div className="flex items-center gap-1 text-xs text-slate-500">
              <CalendarDays size={11} /> {day.date}
            </div>
          )}
        </div>
        <div className="ml-auto text-right">
          <div className="text-[10px] uppercase tracking-wider text-slate-400">Day cost</div>
          <div className="font-bold text-slate-800">{fmtMoney(day.day_total_cost)}</div>
        </div>
      </div>

      <div className="space-y-3 p-4">
        {flight && <FlightCard flight={flight} label={flightLabel} />}
        {day.hotel && <HotelCard hotel={day.hotel} />}
        {activities.length > 0 && (
          <div className="space-y-2">
            {activities.map((a, i) => (
              <ActivityCard key={i} act={a} index={i+1} />
            ))}
          </div>
        )}
        {(day.route || fc) && <RouteMap route={day.route} fc={fc} />}
      </div>
    </div>
  );
};

interface ItineraryViewProps {
  itinerary: DayItinerary[];
  maps?: Record<string, GeoJSONFeatureCollection> | null;
  budget?: BudgetInfo | null;
  destinationCountryCode?: string;
}

export default function ItineraryView(props: ItineraryViewProps) {
  const { itinerary, maps, budget } = props;
  if (!isCompleteItinerarySnapshot({
    itinerary,
    maps,
    budget,
    ...(Object.prototype.hasOwnProperty.call(props, 'destinationCountryCode')
      ? { destination_country_code: props.destinationCountryCode }
      : {}),
  })) return null;
  return (
    <div className="space-y-4">
      <div className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700">
        <Info size={14} className="mt-0.5 shrink-0" />
        <span>
          Attraction &amp; restaurant prices marked <b>“est.”</b> are AI estimates kept within your
          allocated budget — treat them as a guide, not a quote.
        </span>
      </div>
      {budget && <BudgetSummary budget={budget} />}
      {itinerary.map((day, index) => {
        const flightLabel: FlightLabel = itinerary.length === 1
          ? 'Flight'
          : index === 0
            ? 'Outbound flight'
            : index === itinerary.length - 1
              ? 'Return flight'
              : 'Flight';
        return (
          <DayBlock
            key={day.day}
            day={day}
            fc={maps?.[String(day.day)]}
            flightLabel={flightLabel}
          />
        );
      })}
    </div>
  );
}
