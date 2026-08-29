import React, { useState, useEffect, useRef } from 'react';
import {
  Plane, Map, Calendar, Compass, MessageSquare, Menu, Settings,
  User, Paperclip, Send, MapPin, Plus, History, Sunrise, ChevronRight, X,
  ShieldAlert, Phone, Building, Loader2, AlertTriangle,
  Globe, Users, DollarSign, Search, Home
} from 'lucide-react';
import {
  sendChatMessage, submitTripForm, submitProfile, getProfile,
  getChatHistory, newSessionId, ApiError,
  hasCompleteInitialSuccessSnapshot, hasCompleteLiveSuccessSnapshot,
  isChatBudgetConfirmation,
  isSafeAssistantContent,
  isUnexpiredChatBudgetConfirmation,
  type BudgetCheckUnavailableResponse, type BudgetConfirmationResponse,
  type ChatBudgetConfirmation, type ChatHistorySession, type ChatMessageOptions,
  type PlanningUnavailableResponse,
  type TripFormRequest, type TripSubmissionResponse,
} from '../lib/api';
import { loadProfileForGate } from '../lib/profileGate';
import MessageContent from './components/MessageContent';
import ChatBudgetConfirmationCard from './components/ChatBudgetConfirmationCard';
import ItineraryCanvas, {
  findLatestItinerarySnapshot,
} from './components/ItineraryCanvas';

// Chat starts empty — the AI only responds after the user's first message,
// mirroring standard LLM chat behaviour (no auto-greeting).
const INITIAL_MESSAGES: any[] = [];

// --- RECENT TRIPS (Supabase is the only durable source of truth) ---
// A session appears here after the backend returns it or after the current tab
// creates it. Keeping a second browser copy made deleted history reappear and
// retained personal data on the device after a database reset.
type StoredSession = {
  id: string;
  title: string;
  destination: string;
  updatedAt: number;
  messages: any[];
};

type PendingAssessment = {
  assessmentId: string;
  generation: string;
  consumed: boolean;
};

const confirmationGeneration = (message: any, index: number): string => (
  typeof message?.budget_confirmation_generation === 'string'
    ? message.budget_confirmation_generation
    : `history:${index}`
);

const latestActionableAssessment = (
  messages: readonly any[],
): Omit<PendingAssessment, 'consumed'> | null => {
  const newest = messages[messages.length - 1];
  const decision = newest?.role === 'ai' ? newest.budget_confirmation : null;
  return isChatBudgetConfirmation(decision) && isUnexpiredChatBudgetConfirmation(decision)
    ? {
      assessmentId: decision.budget_assessment_id,
      generation: confirmationGeneration(newest, messages.length - 1),
    }
    : null;
};

const LEGACY_SESSIONS_PREFIX = 'wander_sessions_';
const MAX_STORED_SESSIONS = 10;

/** Remove chat copies written by releases that predate server-authoritative history. */
const clearLegacyStoredSessions = (): void => {
  try {
    if (typeof window === 'undefined') return;
    const storage = window.localStorage;
    Object.keys(storage)
      .filter((key) => key.startsWith(LEGACY_SESSIONS_PREFIX))
      .forEach((key) => storage.removeItem(key));
  } catch {
    // Storage can be disabled by browser privacy settings. The application no
    // longer depends on it, so cleanup failure must not block startup.
  }
};

const fromRemoteSession = (session: ChatHistorySession): StoredSession => {
  const parsedTime = Date.parse(session.updated_at);
  return {
    id: session.id,
    title: session.title,
    destination: session.destination,
    updatedAt: Number.isFinite(parsedTime) ? parsedTime : 0,
    messages: session.messages.filter(
      (message) => message.role !== 'ai' || isSafeAssistantContent(message.content),
    ),
  };
};

const relativeDay = (ts: number): string => {
  const d = new Date(ts);
  const now = new Date();
  const startOf = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diffDays = Math.round((startOf(now) - startOf(d)) / 86_400_000);
  if (diffDays <= 0) return 'Today';
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays} days ago`;
  return d.toLocaleDateString();
};

const useMediaQuery = (query: string): boolean => {
  const readMatch = () => typeof window !== 'undefined' && window.matchMedia(query).matches;
  const [matches, setMatches] = useState(readMatch);

  useEffect(() => {
    const media = window.matchMedia(query);
    const updateMatch = () => setMatches(media.matches);
    updateMatch();
    media.addEventListener('change', updateMatch);
    return () => media.removeEventListener('change', updateMatch);
  }, [query]);

  return matches;
};

// --- COMPONENTS ---

const DestinationCard = ({ title, location, image, description }: any) => (
  <div className="flex flex-col bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm hover:shadow-md transition-shadow max-w-sm mt-3">
    <div className="h-40 w-full overflow-hidden">
      <img src={image} alt={title} className="w-full h-full object-cover" />
    </div>
    <div className="p-4">
      <div className="flex items-center gap-1.5 text-slate-500 text-xs font-medium uppercase tracking-wider mb-1">
        <MapPin size={12} />
        {location}
      </div>
      <h4 className="text-lg font-bold text-slate-800 mb-2">{title}</h4>
      <p className="text-sm text-slate-600 line-clamp-2">{description}</p>
      <button className="mt-4 w-full bg-blue-50 hover:bg-blue-100 text-blue-600 font-medium py-2 rounded-lg text-sm transition-colors flex items-center justify-center gap-2">
        View Itinerary <ChevronRight size={16} />
      </button>
    </div>
  </div>
);

const ChatMessage = ({
  message,
  disabled,
  actionable,
  onAcceptRecommendedBudget,
}: {
  message: any;
  disabled: boolean;
  actionable: boolean;
  onAcceptRecommendedBudget: (decision: ChatBudgetConfirmation) => void;
}) => {
  const isAi = message.role === 'ai';
  
  return (
    <div className={`flex gap-4 p-6 ${isAi ? 'bg-slate-50' : 'bg-white'}`}>
      <div className="flex-shrink-0">
        {isAi ? (
          <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white shadow-sm">
            <Compass size={18} />
          </div>
        ) : (
          <div className="w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center text-slate-600 shadow-sm">
            <User size={18} />
          </div>
        )}
      </div>
      <div className="flex-1 max-w-3xl pt-1">
        {isAi ? (
          // AI replies use the structured itinerary / markdown renderer.
          <MessageContent content={message.content} />
        ) : (
          // User messages are plain text — preserve line breaks, no parsing.
          <div className="prose prose-slate max-w-none">
            {message.content.split('\n').map((line: string, i: number) => (
              <p key={i} className="text-slate-700 leading-relaxed m-0 mb-2 last:mb-0">{line}</p>
            ))}
          </div>
        )}
        
        {/* Render Rich Cards if AI provides them */}
        {message.cards && message.cards.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
            {message.cards.map((card: any, idx: number) => (
              <DestinationCard key={idx} {...card} />
            ))}
          </div>
        )}

        {isAi && isChatBudgetConfirmation(message.budget_confirmation) && (
          <ChatBudgetConfirmationCard
            decision={message.budget_confirmation}
            disabled={disabled || !actionable}
            onAccept={() => onAcceptRecommendedBudget(message.budget_confirmation)}
          />
        )}

      </div>
    </div>
  );
};

// --- NEW TRIP INITIALIZATION FORM ---
// Captures the mandatory fields the backend `/api/form/submit` endpoint
// needs to bootstrap a planning session. The three headline fields
// (Destination, Travel Dates, Total Budget) are required by the spec;
// origin country + number of travellers are additionally required by the
// backend contract (see InitialFormRequest / AgentState).
const NewTripForm = ({
  onSubmit,
  onCancel,
  submitting,
  error,
}: {
  onSubmit: (
    form: TripFormRequest,
    label: string,
  ) => Promise<TripSubmissionResponse | undefined>;
  onCancel: () => void;
  submitting: boolean;
  error: string | null;
}) => {
  const [country, setCountry] = useState('');
  const [city, setCity] = useState('');
  const [numPeople, setNumPeople] = useState(1);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [budget, setBudget] = useState('');
  const [budgetUnknown, setBudgetUnknown] = useState(false);
  const [budgetDecision, setBudgetDecision] = useState<
    BudgetConfirmationResponse | BudgetCheckUnavailableResponse | PlanningUnavailableResponse | null
  >(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [planningRetry, setPlanningRetry] = useState<{
    form: TripFormRequest;
    label: string;
  } | null>(null);
  const submitInFlightRef = useRef(false);

  const invalidateBudgetDecision = () => {
    setBudgetDecision((current) => (
      current?.status === 'planning_unavailable' ? current : null
    ));
    setLocalError(null);
  };

  const submitOnce = async (
    form: TripFormRequest,
    label: string,
  ): Promise<TripSubmissionResponse | undefined> => {
    if (submitInFlightRef.current) return undefined;
    submitInFlightRef.current = true;
    try {
      return await onSubmit(form, label);
    } finally {
      submitInFlightRef.current = false;
    }
  };

  const cities = () => city
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean);

  const destinationLabel = () => {
    const selectedCities = cities();
    return selectedCities.length
      ? `${selectedCities.join(', ')}, ${country.trim()}`
      : country.trim();
  };

  const baseTripForm = (): Omit<TripFormRequest, 'total_budget'> => ({
    country: country.trim(),
    city: cities(),
    num_people: numPeople,
    start_date: startDate,
    end_date: endDate,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (submitting) return;

    // ── Client-side validation of the mandatory fields ──
    // Origin comes from the onboarding profile now, so it is NOT collected here.
    // Destination state/city is optional. When omitted, the backend resolves
    // and verifies a suitable planning city before budget assessment.
    if (!country.trim()) return setLocalError('Please enter a destination country.');
    if (!startDate || !endDate) return setLocalError('Please pick your travel dates.');
    if (new Date(endDate) < new Date(startDate))
      return setLocalError('The end date must be on or after the start date.');
    const budgetNum = Number(budget);
    if (!budgetUnknown && (!budget || Number.isNaN(budgetNum) || budgetNum <= 0))
      return setLocalError('Please enter a total budget greater than 0.');
    if (numPeople < 1) return setLocalError('There must be at least 1 traveller.');

    setLocalError(null);
    setBudgetDecision(null);

    const form: TripFormRequest = budgetUnknown
      ? { ...baseTripForm(), request_budget_recommendation: true }
      : { ...baseTripForm(), total_budget: budgetNum };
    const label = destinationLabel();
    const savedSubmission = {
      form: { ...form, city: [...form.city] },
      label,
    };
    const response = await submitOnce(savedSubmission.form, savedSubmission.label);
    if (response?.status === 'planning_unavailable') {
      setPlanningRetry(savedSubmission);
      setBudgetDecision(response);
    } else if (response && response.status !== 'success') {
      setPlanningRetry(null);
      setBudgetDecision(response);
    }
  };

  const useRecommendedBudget = async () => {
    if (
      submitting
      || budgetDecision?.status !== 'budget_confirmation_required'
    ) return;

    setLocalError(null);

    // The budget assessment is bound to the exact city resolved by the
    // backend. Echo that trusted city list instead of re-sending [].
    const resolvedCities = [...budgetDecision.resolved_cities];

    const response = await submitOnce(
      {
        ...baseTripForm(),
        city: resolvedCities,
        total_budget: budgetDecision.recommended_minimum_budget,
        budget_assessment_id: budgetDecision.budget_assessment_id,
      },
      `${resolvedCities.join(', ')}, ${country.trim()}`,
    );

    if (response && response.status !== 'success') {
      setBudgetDecision(response);
    }
  };

  const enterAnotherBudget = () => {
    if (budgetDecision?.status === 'budget_confirmation_required') {
      // Preserve and expose the verified city used for the budget assessment.
      setCity(budgetDecision.resolved_cities.join(', '));
    }

    setBudgetDecision(null);
    setBudgetUnknown(false);
    setBudget('');
    setLocalError(null);
  };

  const retryBudgetCheck = async () => {
    if (submitting || budgetDecision?.status !== 'budget_check_unavailable') return;
    const form: TripFormRequest = budgetUnknown
      ? { ...baseTripForm(), request_budget_recommendation: true }
      : { ...baseTripForm(), total_budget: Number(budget) };
    setBudgetDecision(null);
    const response = await submitOnce(form, destinationLabel());
    if (response && response.status !== 'success') setBudgetDecision(response);
  };

  const retryPlanning = async () => {
    if (submitting
      || submitInFlightRef.current
      || budgetDecision?.status !== 'planning_unavailable'
      || !planningRetry) return;
    const response = await submitOnce(
      { ...planningRetry.form, city: [...planningRetry.form.city] },
      planningRetry.label,
    );
    if (response?.status === 'planning_unavailable') setBudgetDecision(response);
    else if (response && response.status !== 'success') {
      setPlanningRetry(null);
      setBudgetDecision(response);
    }
  };

  const formatMoney = (amount: number, currency: string) =>
    `${currency} ${new Intl.NumberFormat('en-MY', {
      maximumFractionDigits: 2,
    }).format(amount)}`;

  const inputClass =
    'w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-800 outline-none transition-colors focus:border-blue-500 focus:ring-1 focus:ring-blue-500 placeholder:text-slate-400';
  const labelClass =
    'flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5';

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-2xl mx-auto px-6 py-10">
        <div className="mb-8 text-center">
          <div className="w-14 h-14 bg-blue-100 text-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-4 rotate-3">
            <Plane size={28} />
          </div>
          <h2 className="text-2xl font-bold text-slate-800 mb-1">Plan a new trip</h2>
          <p className="text-slate-500 text-sm">
            I'll verify your budget first, then plan immediately when it is sufficient.
          </p>
        </div>

        <form
          onSubmit={handleSubmit}
          className="bg-white border border-slate-200 rounded-2xl shadow-sm p-6 space-y-5"
        >
          {/* Destination */}
          <div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className={labelClass}>
                  <Globe size={13} /> Destination country
                </label>
                <input
                  value={country}
                  onChange={(e) => {
                    invalidateBudgetDecision();
                    setCountry(e.target.value);
                  }}
                  placeholder="e.g. Japan"
                  className={inputClass}
                />
              </div>
              <div>
                <label className={labelClass}>
                  <MapPin size={13} /> Destination state / city (optional)
                </label>
                <input
                  value={city}
                  onChange={(e) => {
                    invalidateBudgetDecision();
                    setCity(e.target.value);
                  }}
                  placeholder="e.g. Kyoto, Osaka — leave blank and AI will choose"
                  className={inputClass}
                />
              </div>
            </div>
          </div>

          {/* Travel dates */}
          <div>
            <label className={labelClass}>
              <Calendar size={13} /> Travel dates
            </label>
            <div className="grid grid-cols-2 gap-4">
              <input
                type="date"
                value={startDate}
                onChange={(e) => {
                  invalidateBudgetDecision();
                  setStartDate(e.target.value);
                }}
                className={inputClass}
                aria-label="Start date"
              />
              <input
                type="date"
                value={endDate}
                onChange={(e) => {
                  invalidateBudgetDecision();
                  setEndDate(e.target.value);
                }}
                className={inputClass}
                aria-label="End date"
              />
            </div>
          </div>

          {/* Budget + travellers */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className={labelClass}>
                <DollarSign size={13} /> Total budget
              </label>
              <input
                type="number"
                min={1}
                step="any"
                value={budget}
                disabled={budgetUnknown}
                onChange={(e) => {
                  invalidateBudgetDecision();
                  setBudget(e.target.value);
                }}
                placeholder="e.g. 3000"
                className={`${inputClass} disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400`}
              />
              <label className="mt-2 flex cursor-pointer items-center gap-2 text-xs text-slate-600">
                <input
                  type="checkbox"
                  checked={budgetUnknown}
                  onChange={(e) => {
                    invalidateBudgetDecision();
                    setBudgetUnknown(e.target.checked);
                    if (e.target.checked) setBudget('');
                  }}
                  className="h-4 w-4 rounded border-slate-300 text-blue-600"
                />
                I don't know my budget
              </label>
            </div>
            <div>
              <label className={labelClass}>
                <Users size={13} /> Travellers
              </label>
              <input
                type="number"
                min={1}
                value={numPeople}
                onChange={(e) => {
                  invalidateBudgetDecision();
                  setNumPeople(Math.max(1, Number(e.target.value) || 1));
                }}
                className={inputClass}
              />
            </div>
          </div>

          {budgetDecision?.status === 'budget_confirmation_required' && (
            <section
              aria-label="Budget confirmation"
              className="space-y-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-slate-700"
            >
              <div>
                <h3 className="font-semibold text-amber-900">
                  Budget confirmation required
                </h3>

                <p className="mt-1 text-xs font-medium text-amber-900">
                  Planning area: {budgetDecision.resolved_cities.join(', ')}, {country.trim()}
                </p>

                <p className="mt-1 text-xs leading-relaxed text-amber-800">
                  {budgetDecision.chat_reply} No itinerary has been generated yet.
                </p>
              </div>
              <div className="rounded-lg bg-white p-3 shadow-sm">
                {budgetDecision.stated_budget !== null && (
                  <p className="mb-2 text-xs text-slate-600">
                    Your entered budget: {formatMoney(
                      budgetDecision.stated_budget,
                      budgetDecision.base_currency,
                    )}
                  </p>
                )}
                <p className="text-xs uppercase tracking-wide text-slate-500">Recommended minimum</p>
                <p className="mt-1 text-xl font-bold text-slate-900">
                  {formatMoney(
                    budgetDecision.recommended_minimum_budget,
                    budgetDecision.base_currency,
                  )}
                </p>
              </div>
              <div className="space-y-1 text-xs">
                <p className="pb-1 text-slate-600">
                  {budgetDecision.evidence.hotel_nights === 0
                    ? 'Current cheapest flight prices must fit the fixed 25% transportation allocation; no overnight accommodation is required.'
                    : 'Current cheapest provider prices must fit the fixed allocation: 25% transportation and 35% accommodation.'}
                </p>
                <p>
                  Outbound flight: {formatMoney(
                    budgetDecision.evidence.outbound_flight_price,
                    budgetDecision.destination_currency,
                  )}
                </p>
                <p>
                  Return flight: {formatMoney(
                    budgetDecision.evidence.return_flight_price,
                    budgetDecision.destination_currency,
                  )}
                </p>
                <p>{budgetDecision.evidence.hotel_nights === 0
                  ? 'No overnight hotel is required for this day trip.'
                  : <>Hotel: {budgetDecision.evidence.hotel_nights} nights × {formatMoney(
                    budgetDecision.evidence.hotel_price_per_night,
                    budgetDecision.destination_currency,
                  )} per night</>}
                </p>
                <p className="pt-1 text-slate-500">
                  This cached price check expires at{' '}
                  {new Date(budgetDecision.expires_at).toLocaleString()}.
                </p>
              </div>
              <div className="flex flex-col gap-2 sm:flex-row">
                <button
                  type="button"
                  onClick={enterAnotherBudget}
                  disabled={submitting}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                >
                  Enter another budget
                </button>
                <button
                  type="button"
                  onClick={useRecommendedBudget}
                  disabled={submitting}
                  className="rounded-lg bg-blue-600 px-3 py-2 text-xs font-semibold text-white hover:bg-blue-700 disabled:bg-slate-300"
                >
                  {submitting ? 'Planning…' : 'Use recommended budget'}
                </button>
              </div>
            </section>
          )}

          {budgetDecision?.status === 'budget_check_unavailable' && (
            <section
              role="alert"
              className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800"
            >
              <h3 className="font-semibold">Budget check unavailable</h3>
              <p className="mt-1 text-xs leading-relaxed">{budgetDecision.chat_reply}</p>
              <p className="mt-1 text-xs font-medium">No itinerary was generated.</p>
              <button
                type="button"
                onClick={retryBudgetCheck}
                disabled={submitting}
                className="mt-3 rounded-lg border border-red-200 bg-white px-3 py-2 text-xs font-semibold hover:bg-red-100"
              >
                {submitting ? 'Retrying…' : 'Retry budget check'}
              </button>
            </section>
          )}

          {budgetDecision?.status === 'planning_unavailable' && (
            <section
              role="alert"
              className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800"
            >
              <h3 className="font-semibold">Planning unavailable</h3>
              <p className="mt-1 text-xs leading-relaxed">{budgetDecision.chat_reply}</p>
              <p className="mt-1 text-xs font-medium">
                Your trip details are unchanged and no itinerary was generated.
              </p>
              <button
                type="button"
                onClick={retryPlanning}
                disabled={submitting}
                className="mt-3 rounded-lg border border-red-200 bg-white px-3 py-2 text-xs font-semibold hover:bg-red-100 disabled:opacity-50"
              >
                {submitting ? 'Retrying…' : 'Retry planning'}
              </button>
            </section>
          )}

          {(localError || error) && (
            <div
              role="alert"
              className="flex items-start gap-2 text-xs text-red-600 bg-red-50 border border-red-100 rounded-lg px-3 py-2"
            >
              <AlertTriangle size={14} className="mt-0.5 shrink-0" />
              <span>{localError || error}</span>
            </div>
          )}

          <div className="flex items-center gap-3 pt-1">
            <button
              type="button"
              onClick={onCancel}
              disabled={submitting}
              className="px-4 py-2.5 rounded-lg text-sm font-medium text-slate-600 hover:bg-slate-100 transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="flex-1 flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white font-medium py-2.5 rounded-lg transition-colors"
            >
              {submitting ? (
                <>
                  <Loader2 size={18} className="animate-spin" /> Checking budget…
                </>
              ) : (
                <>
                  Generate itinerary <ChevronRight size={18} />
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

// --- ONBOARDING (one-time profile: Name / Origin Country / State) ---
// Captured once and persisted to the backend (POST /api/profile) BEFORE any
// trip planning. The trip form then no longer asks for the user's origin.
const OnboardingScreen = ({ onComplete }: { onComplete: (homeCountry: string) => void }) => {
  const [name, setName] = useState('');
  const [originCountry, setOriginCountry] = useState('');
  const [originState, setOriginState] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const inputClass =
    'w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-800 outline-none transition-colors focus:border-blue-500 focus:ring-1 focus:ring-blue-500 placeholder:text-slate-400';
  const labelClass =
    'flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5';

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (busy) return;
    if (!name.trim()) return setErr('Please enter your name.');
    if (!originCountry.trim()) return setErr('Please enter your home country.');
    if (!originState.trim()) return setErr('Please enter your home state / province.');
    setErr(null);
    setBusy(true);
    try {
      await submitProfile({
        name: name.trim(),
        origin_country: originCountry.trim(),
        origin_state: originState.trim(),
      });
      onComplete(originCountry.trim());
    } catch (e2) {
      setErr(
        e2 instanceof ApiError
          ? `${e2.message}${e2.status ? ` (HTTP ${e2.status})` : ''}`
          : 'Could not save your profile. Is the backend running?',
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-screen w-full items-center justify-center bg-slate-50 p-4">
      <div className="w-full max-w-md">
        <div className="mb-6 text-center">
          <div className="mx-auto mb-4 flex h-14 w-14 rotate-3 items-center justify-center rounded-2xl bg-blue-100 text-blue-600">
            <Plane size={28} className="fill-current" />
          </div>
          <h1 className="text-2xl font-bold text-slate-800">Welcome to Wander AI</h1>
          <p className="mt-1 text-sm text-slate-500">
            Tell us a little about you. We'll remember it so you don't have to enter it for every trip.
          </p>
        </div>

        <form onSubmit={submit} className="space-y-4 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <div>
            <label className={labelClass}><User size={13} /> Your name</label>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Alex Tan" className={inputClass} />
          </div>
          <div>
            <label className={labelClass}><Home size={13} /> Home country</label>
            <input value={originCountry} onChange={(e) => setOriginCountry(e.target.value)} placeholder="e.g. Malaysia" className={inputClass} />
          </div>
          <div>
            <label className={labelClass}>
              <MapPin size={13} /> Home state / province
            </label>
            <input value={originState} onChange={(e) => setOriginState(e.target.value)} placeholder="e.g. Selangor" className={inputClass} />
          </div>

          {err && (
            <div className="flex items-start gap-2 rounded-lg border border-red-100 bg-red-50 px-3 py-2 text-xs text-red-600">
              <AlertTriangle size={14} className="mt-0.5 shrink-0" />
              <span>{err}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={busy}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 py-2.5 font-medium text-white transition-colors hover:bg-blue-700 disabled:bg-slate-300"
          >
            {busy ? (<><Loader2 size={18} className="animate-spin" /> Saving…</>) : (<>Continue <ChevronRight size={18} /></>)}
          </button>
        </form>
      </div>
    </div>
  );
};

export default function App() {
  const [messages, setMessages] = useState<any[]>(INITIAL_MESSAGES);
  const [input, setInput] = useState('');
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showEmergency, setShowEmergency] = useState(false);
  const [itineraryCanvasOpen, setItineraryCanvasOpen] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // 'chat' = normal conversation view; 'form' = new-trip initialization form.
  const [mode, setMode] = useState<'chat' | 'form'>('chat');
  // Human-readable destination for the active trip; used to give the
  // emergency handoff prompts geographic context.
  const [destination, setDestination] = useState<string>('');
  // One session id per conversation thread; generated lazily on first send
  // and echoed back on every subsequent /api/chat/message call.
  const [sessionId, setSessionId] = useState<string>(() => newSessionId());
// null = still checking; false = must onboard; true = ready to use the app.
const [onboarded, setOnboarded] = useState<boolean | null>(null);
// A profile-read failure is different from a confirmed missing profile.
// Keep the user behind a retryable verification screen instead of incorrectly
// sending an existing user back to onboarding.
const [profileLoadError, setProfileLoadError] = useState<string | null>(null);
// Existing users remain behind the loading gate until Supabase history has
// answered, preventing stale client data from flashing before validation.
const [historyLoaded, setHistoryLoaded] = useState(false);
  // Incremented when the browser tab regains focus so externally deleted
  // Supabase profile/checkpoint rows are reflected without a manual reload.
  const [serverSyncRevision, setServerSyncRevision] = useState(0);
  // Profile revalidation completes before history is refreshed. Keeping this
  // separate lets focus-triggered checks run in the background without putting
  // an already-rendered app back behind the initial loading gate.
  const [historySyncRevision, setHistorySyncRevision] = useState(0);
  // User's home country from the onboarding profile — used so emergency
  // embassy lookups target the RIGHT country's embassy (not e.g. the US one).
  const [homeCountry, setHomeCountry] = useState<string>('');
  // In-memory view of Supabase history plus work created in the current tab.
  const [recentTrips, setRecentTrips] = useState<StoredSession[]>([]);
  const [pendingAssessments, setPendingAssessments] = useState<Record<string, PendingAssessment>>({});
  const [clock, setClock] = useState(Date.now());
  const initialHistoryHydratedRef = useRef(false);
  const restoringSessionRef = useRef<string | null>(null);
  const sendInFlightRef = useRef(false);
  const sessionIdRef = useRef(sessionId);
  const requestTokenRef = useRef(0);
  const activeRequestTokenRef = useRef<number | null>(null);
  const pendingAssessmentsRef = useRef<Record<string, PendingAssessment>>({});
  const openItineraryButtonRef = useRef<HTMLButtonElement>(null);
  const isMobileViewport = useMediaQuery('(max-width: 1023px)');
  const latestItinerary = findLatestItinerarySnapshot(messages);
  const latestItineraryRevision = latestItinerary
    ? `${sessionId}:${latestItinerary.messageIndex}`
    : '';

  const setPendingAssessment = (
    targetSessionId: string,
    assessment: Omit<PendingAssessment, 'consumed'> | null,
  ) => {
    const next = { ...pendingAssessmentsRef.current };
    if (assessment) {
      const current = next[targetSessionId];
      next[targetSessionId] = current?.assessmentId === assessment.assessmentId
        && current.generation === assessment.generation
        ? current
        : { ...assessment, consumed: false };
    }
    else delete next[targetSessionId];
    pendingAssessmentsRef.current = next;
    setPendingAssessments(next);
  };

  const consumePendingAssessment = (
    targetSessionId: string,
    assessmentId: string,
    generation: string,
  ): boolean => {
    const current = pendingAssessmentsRef.current[targetSessionId];
    if (
      !current
      || current.consumed
      || current.assessmentId !== assessmentId
      || current.generation !== generation
    ) return false;
    const next = {
      ...pendingAssessmentsRef.current,
      [targetSessionId]: { ...current, consumed: true },
    };
    pendingAssessmentsRef.current = next;
    setPendingAssessments(next);
    return true;
  };

  const invalidateActiveRequest = () => {
    requestTokenRef.current += 1;
    activeRequestTokenRef.current = null;
    sendInFlightRef.current = false;
    setLoading(false);
  };

  const activateSession = (nextSessionId: string) => {
    invalidateActiveRequest();
    sessionIdRef.current = nextSessionId;
    setSessionId(nextSessionId);
  };

  useEffect(() => {
    const active = pendingAssessments[sessionId];
    if (!active || active.consumed) return;
    const decision = messages.find((message, index) => (
      isChatBudgetConfirmation(message?.budget_confirmation)
      && message.budget_confirmation.budget_assessment_id === active.assessmentId
      && confirmationGeneration(message, index) === active.generation
    ))?.budget_confirmation;
    if (!decision) return;
    const delay = Date.parse(decision.expires_at) - Date.now();
    if (delay <= 0) {
      setPendingAssessment(sessionId, null);
      return;
    }
    const timer = window.setTimeout(
      () => setClock(Date.now()),
      Math.min(delay + 1, 2_147_483_647),
    );
    return () => window.clearTimeout(timer);
  }, [sessionId, messages, pendingAssessments, clock]);

  useEffect(() => {
    setItineraryCanvasOpen(Boolean(latestItineraryRevision));
  }, [latestItineraryRevision]);

  useEffect(() => {
    if (latestItinerary && !itineraryCanvasOpen) {
      openItineraryButtonRef.current?.focus();
    }
  }, [itineraryCanvasOpen, latestItineraryRevision]);

  const mobileCanvasActive = Boolean(
    isMobileViewport && latestItinerary && itineraryCanvasOpen,
  );
  // React 18's runtime only serializes the HTML inert attribute when it is
  // passed as a string, while its current TypeScript declaration is boolean.
  const mobileInertAttributes = mobileCanvasActive
    ? ({ inert: '' } as unknown as React.HTMLAttributes<HTMLElement>)
    : {};

  // Purge legacy browser copies before the profile/history requests finish.
  // New releases never write chat or itinerary data to localStorage.
  useEffect(() => {
    clearLegacyStoredSessions();
  }, []);

  useEffect(() => {
    // Do not start another full profile + history synchronization while the
    // initial hydration is still running. A page refresh can emit focus or
    // visibility events shortly after mount, which previously duplicated the
    // initial /api/profile and /api/chat/history requests.
    let lastRevalidationAt = Date.now();

    const REVALIDATE_COOLDOWN_MS = 60_000;

    const revalidate = () => {
      // Initial profile/history hydration is already retrieving the latest
      // server state. Do not launch a duplicate request during that process.
      if (!initialHistoryHydratedRef.current) {
        return;
      }

      const now = Date.now();

      // Switching briefly between the browser, VS Code and terminal should not
      // reload the complete persisted conversation history every time.
      if (now - lastRevalidationAt < REVALIDATE_COOLDOWN_MS) {
        return;
      }

      lastRevalidationAt = now;

      setServerSyncRevision(
        (revision) => revision + 1,
      );
    };

    const revalidateWhenVisible = () => {
      if (document.visibilityState === 'visible') {
        revalidate();
      }
    };

    window.addEventListener(
      'focus',
      revalidate,
    );

    document.addEventListener(
      'visibilitychange',
      revalidateWhenVisible,
    );

    return () => {
      window.removeEventListener(
        'focus',
        revalidate,
      );

      document.removeEventListener(
        'visibilitychange',
        revalidateWhenVisible,
      );
    };
  }, []);

  // On load, check onboarding status. A confirmed missing profile is a full
  // reset signal, so all in-memory conversation state is cleared as well.
  useEffect(() => {
    let mounted = true;
    loadProfileForGate(getProfile)
      .then((r) => {
        if (!mounted) return;
        setProfileLoadError(null);
        const isOnboarded = !!r.onboarded;
        if (!isOnboarded) {
          initialHistoryHydratedRef.current = false;
          restoringSessionRef.current = null;
          setRecentTrips([]);
          setMessages(INITIAL_MESSAGES);
          activateSession(newSessionId());
          setDestination('');
          setMode('chat');
          setHistoryLoaded(true);
        }
        setOnboarded(isOnboarded);
        setHomeCountry(String(r.profile?.home_country ?? ''));
        if (isOnboarded) {
          setHistorySyncRevision((revision) => revision + 1);
        }
      })
      .catch((err) => {
        if (!mounted) return;
        // IMPORTANT: a transport/503/invalid-response failure does NOT prove
        // that the Supabase row is absent. Preserve the current app state and
        // let the user retry verification instead of showing onboarding.
        const message = err instanceof ApiError
          ? err.message
          : err instanceof Error
            ? err.message
            : 'Could not verify your saved profile.';
        setProfileLoadError(message);
      });
    return () => { mounted = false; };
  }, [serverSyncRevision]);

  // Supabase is authoritative. Every successful response replaces the complete
  // Recent Trips list. Initial hydration may restore or clear the active
  // workspace; background synchronization updates Recent Trips only.
  useEffect(() => {
    if (onboarded !== true) return;
    const isInitialHydration = !initialHistoryHydratedRef.current;
    let mounted = true;
    // `historyLoaded` starts false for the initial request. Later focus-triggered
    // refreshes deliberately leave it true so the current interface stays visible.
    getChatHistory(MAX_STORED_SESSIONS)
      .then((response) => {
        if (!mounted) return;
        const remoteSessions = response.sessions
          .map(fromRemoteSession)
          .filter((session) => session.id)
          .sort((a, b) => b.updatedAt - a.updatedAt)
          .slice(0, MAX_STORED_SESSIONS);
        setRecentTrips(remoteSessions);
        if (!isInitialHydration) return;

        const newest = remoteSessions.find((session) => session.messages.length > 0);
        if (newest) {
          restoringSessionRef.current = newest.id;
          activateSession(newest.id);
          setMessages(newest.messages);
          setPendingAssessment(newest.id, latestActionableAssessment(newest.messages));
          setDestination(newest.destination);
          setMode('chat');
        } else {
          restoringSessionRef.current = null;
          activateSession(newSessionId());
          setMessages(INITIAL_MESSAGES);
          setDestination('');
          setMode('chat');
        }
        setError(null);
      })
      .catch(() => {
        if (!mounted || !isInitialHydration) return;
        // Fail closed: unverified local history must never be displayed as if
        // it still existed in Supabase.
        restoringSessionRef.current = null;
        setRecentTrips([]);
        activateSession(newSessionId());
        setMessages(INITIAL_MESSAGES);
        setDestination('');
        setMode('chat');
        setError('Could not verify saved trips. No cached history was loaded.');
      })
      .finally(() => {
        if (mounted && isInitialHydration) {
          initialHistoryHydratedRef.current = true;
          setHistoryLoaded(true);
        }
      });
    return () => { mounted = false; };
  }, [onboarded, historySyncRevision]);

  // Keep the current tab's Recent Trips list in sync. Durable persistence is
  // performed by the backend checkpointer, never by browser storage.
  useEffect(() => {
    if (messages.length === 0) return;
    if (restoringSessionRef.current === sessionId) {
      restoringSessionRef.current = null;
      return;
    }
    const firstUser = messages.find((m) => m.role === 'user');
    const title =
      destination ||
      (firstUser ? String(firstUser.content).replace(/\s+/g, ' ').slice(0, 48) : 'New trip');
    setRecentTrips((prev) => {
      const entry: StoredSession = {
        id: sessionId,
        title,
        destination,
        updatedAt: Date.now(),
        messages,
      };
      const next = [entry, ...prev.filter((s) => s.id !== sessionId)].slice(
        0,
        MAX_STORED_SESSIONS,
      );
      return next;
    });
  }, [messages, destination, sessionId]);

  // Restore a stored conversation (same session id → backend state continues).
  const openTrip = (trip: StoredSession) => {
    if (trip.id === sessionId) return;
    restoringSessionRef.current = trip.messages.length > 0 ? trip.id : null;
    activateSession(trip.id);
    setMessages(trip.messages);
    setPendingAssessment(trip.id, latestActionableAssessment(trip.messages));
    setDestination(trip.destination);
    setError(null);
    setInput('');
    setMode('chat');
  };

  const errText = (err: unknown) =>
    err instanceof ApiError
      ? `${err.message}${err.status ? ` (HTTP ${err.status})` : ''}`
      : err instanceof Error
        ? err.message
        : 'Unknown error contacting the backend.';

  // Core send routine. `promptText` is what the backend receives; `displayText`
  // (optional) is what the user sees — this lets us inject "hidden" prompts
  // (e.g. from the Emergency modal) while showing a friendly label in the feed.
  const sendMessage = async (
    promptText: string,
    displayText?: string,
    options?: ChatMessageOptions,
  ) => {
    const text = promptText.trim();
    if (!text || loading || sendInFlightRef.current) return;
    const sourceSessionId = sessionIdRef.current;
    const requestToken = ++requestTokenRef.current;
    activeRequestTokenRef.current = requestToken;
    sendInFlightRef.current = true;
    // Any newer user decision supersedes the previous confirmation generation
    // immediately. A fresh confirmation response will install its own token.
    setPendingAssessment(sourceSessionId, null);
    // Optimistically render the user's message and clear the composer.
    setMessages(prev => [...prev, { role: 'user', content: (displayText ?? text) }]);
    setInput('');
    setError(null);
    setLoading(true);

    try {
      const res = options
        ? await sendChatMessage(sourceSessionId, text, options)
        : await sendChatMessage(sourceSessionId, text);

      const isCurrentRequest = () =>
        activeRequestTokenRef.current === requestToken && sessionIdRef.current === sourceSessionId;
      if (!isCurrentRequest()) return;

      const aiResponse: any = {
        role: 'ai',
        content: res.chat_reply || '(The assistant returned an empty reply.)',
      };

      if (res.status === 'budget_confirmation_required') {
        if (isChatBudgetConfirmation(res.budget_confirmation)) {
          const generation = `live:${requestToken}`;
          aiResponse.budget_confirmation = res.budget_confirmation;
          aiResponse.budget_confirmation_generation = generation;
          setPendingAssessment(
            sourceSessionId,
            isUnexpiredChatBudgetConfirmation(res.budget_confirmation)
              ? {
                assessmentId: res.budget_confirmation.budget_assessment_id,
                generation,
              }
              : null,
          );
        } else {
          setPendingAssessment(sourceSessionId, null);
          setError('The budget confirmation could not be loaded. Please try again.');
        }
      } else if (res.status === 'budget_check_unavailable') {
        setError(res.chat_reply || 'The provider budget check is currently unavailable.');
      }

      // Only render the itinerary cards when THIS turn actually changed the
      // plan (itinerary / budget / trip requirements). Plain Q&A answers stay
      // text-only — the backend flags modifications via `itinerary_modified`.
      const hasItinerary =
        hasCompleteLiveSuccessSnapshot(res);
      if (hasItinerary) {
        aiResponse.itinerary = res.draft_itinerary;
        aiResponse.maps = res.daily_map_info;
        aiResponse.budget = {
          total: res.total_budget,
          currency: res.currency,
          allocation: res.budget_allocation,
        };
        aiResponse.destination_country_code = res.destination_country_code;
      }

      setMessages(prev => [...prev, aiResponse]);
    } catch (err) {
      if (activeRequestTokenRef.current !== requestToken || sessionIdRef.current !== sourceSessionId) {
        return;
      }
      const msg = errText(err);
      setError(msg);
      setMessages(prev => [
        ...prev,
        { role: 'ai', content: `⚠️ Request failed: ${msg}` },
      ]);
    } finally {
      if (activeRequestTokenRef.current === requestToken && sessionIdRef.current === sourceSessionId) {
        activeRequestTokenRef.current = null;
        sendInFlightRef.current = false;
        setLoading(false);
      }
    }
  };

  const handleSend = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(input);
  };

  // Emergency handoff: close the modal, drop a concise user-facing label into
  // the feed, and send the detailed hidden prompt to the agent (which can call
  // its nearby-search / knowledge tools to return real-time info).
  const triggerEmergency = (displayLabel: string, hiddenPrompt: string) => {
    setShowEmergency(false);
    setMode('chat');
    sendMessage(hiddenPrompt, displayLabel);
  };

  const place = destination || 'my current trip destination';
  const home = homeCountry || 'my home country';
  const emergencyActions = {
    // ONE button covers police + ambulance + fire numbers (they were always
    // returned together anyway — separate buttons were redundant).
    allNumbers: () =>
      triggerEmergency(
        '📞 Get the police, ambulance & fire emergency numbers',
        `EMERGENCY: What are the official POLICE, AMBULANCE and FIRE emergency phone numbers ` +
          `to dial in ${place}? List each service with its exact number, clearly labelled.`,
      ),
    policeNearest: () =>
      triggerEmergency(
        '🚓 Find the nearest police station',
        `EMERGENCY: I need help now. Find the nearest police station to ${place}. ` +
          `Return its name, full address, phone number if available, and distance from me, ` +
          `and include a [MAP: latitude, longitude | name] line for it so I can see it on a map.`,
      ),
    medicalNearest: () =>
      triggerEmergency(
        '🚑 Find the nearest hospital & fire station',
        `EMERGENCY: I need urgent medical or fire help near ${place}. ` +
          `Find the nearest hospital and fire station and return their names, full addresses, ` +
          `and distances, and include a [MAP: latitude, longitude | name] line for each.`,
      ),
    embassyNearest: () =>
      triggerEmergency(
        `🏛️ Find the nearest ${home} embassy / consulate`,
        `EMERGENCY: I am a citizen of ${home}. Find the nearest ${home} embassy or consulate ` +
          `located in ${place} — it MUST be the embassy OF ${home}, not any other country's embassy. ` +
          `Return its name, full address, phone number, and distance from me, ` +
          `and include a [MAP: latitude, longitude | name] line for it.`,
      ),
  };

  // Submit the structured new-trip form and render the itinerary as the
  // first AI message in a fresh session.
  const handleFormSubmit = async (form: TripFormRequest, label: string) => {
    if (loading) return undefined;
    const sourceSessionId = sessionIdRef.current;
    const requestToken = ++requestTokenRef.current;
    activeRequestTokenRef.current = requestToken;
    setError(null);
    setLoading(true);
    try {
      const res = await submitTripForm(form);
      if (activeRequestTokenRef.current !== requestToken || sessionIdRef.current !== sourceSessionId) {
        return undefined;
      }

      // Budget responses deliberately contain no session or itinerary. The
      // form owns the confirmation UI and planning remains blocked.
      if (res.status !== 'success') return res;
      if (!hasCompleteInitialSuccessSnapshot(res)) {
        setError('The planning service did not return a complete itinerary. Please retry.');
        return undefined;
      }

      // Adopt the backend-issued session id for conversation continuity.
      activateSession(res.session_id);

      const resolvedDestinationLabel =
        res.resolved_cities.length > 0
          ? `${res.resolved_cities.join(', ')}, ${form.country.trim()}`
          : label;

      setDestination(resolvedDestinationLabel);

      const aiResponse: any = {
        role: 'ai',
        content: res.chat_reply,
      };

      aiResponse.itinerary = res.itinerary;
      aiResponse.maps = res.daily_geojson_maps;
      aiResponse.budget = {
        total: res.total_budget,
        currency: res.currency,
        allocation: res.budget_allocation,
      };
      aiResponse.destination_country_code = res.destination_country_code;

      setMessages([aiResponse]);
      setMode('chat');
      return res;
    } catch (err) {
      if (activeRequestTokenRef.current !== requestToken || sessionIdRef.current !== sourceSessionId) {
        return undefined;
      }
      // Keep the user on the form so they can correct and retry.
      setError(errText(err));
      return undefined;
    } finally {
      if (activeRequestTokenRef.current === requestToken && sessionIdRef.current === sourceSessionId) {
        activeRequestTokenRef.current = null;
        setLoading(false);
      }
    }
  };

  const handleNewTrip = () => {
    restoringSessionRef.current = null;
    setMessages(INITIAL_MESSAGES);
    activateSession(newSessionId());
    setDestination('');
    setError(null);
    setInput('');
    setMode('form');
  };

  // ── Onboarding gate ──
  if (onboarded === null) {
    if (profileLoadError) {
      return (
        <div className="flex h-screen w-full items-center justify-center bg-slate-50 px-4">
          <div className="w-full max-w-md rounded-2xl border border-amber-200 bg-white p-6 text-center shadow-sm">
            <AlertTriangle size={28} className="mx-auto mb-3 text-amber-500" />
            <h2 className="text-base font-semibold text-slate-800">Could not verify your profile</h2>
            <p className="mt-2 text-sm text-slate-500">
              Your saved profile was not deleted. The app could not confirm it from the backend.
            </p>
            <button
              type="button"
              onClick={() => {
                setProfileLoadError(null);
                setServerSyncRevision((revision) => revision + 1);
              }}
              className="mt-4 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
            >
              Retry
            </button>
          </div>
        </div>
      );
    }
    return (
      <div className="flex h-screen w-full items-center justify-center bg-slate-50 text-slate-500">
        <Loader2 size={20} className="mr-2 animate-spin" /> Loading…
      </div>
    );
  }
  if (onboarded && !historyLoaded) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-slate-50 text-slate-500">
        <Loader2 size={20} className="mr-2 animate-spin" /> Loading…
      </div>
    );
  }
  if (!onboarded) {
    return (
      <OnboardingScreen
        onComplete={(hc) => {
          setHomeCountry(hc);
          setHistoryLoaded(false);
          setOnboarded(true);
        }}
      />
    );
  }

  return (
    <div className="flex h-screen w-full bg-white font-sans text-slate-800 overflow-hidden relative">

      {/* SIDEBAR */}
      <div
        {...mobileInertAttributes}
        aria-hidden={mobileCanvasActive || undefined}
        className={`transition-all duration-300 ease-in-out flex flex-col bg-slate-50 border-r border-slate-200 ${sidebarOpen ? 'w-64' : 'w-0 opacity-0'} overflow-hidden`}
      >
        <div className="p-4 flex items-center justify-between border-b border-slate-200 shrink-0">
          <div className="flex items-center gap-2 text-blue-600 font-bold text-lg">
            <Plane size={24} className="fill-current" />
            <span>Wander AI</span>
          </div>
          <button onClick={() => setSidebarOpen(false)} className="md:hidden text-slate-500 hover:text-slate-800">
            <X size={20} />
          </button>
        </div>
        
        <div className="p-3">
          <button
            onClick={handleNewTrip}
            className="w-full flex items-center gap-2 bg-white border border-slate-200 hover:border-blue-400 hover:text-blue-600 text-slate-700 py-2.5 px-4 rounded-lg font-medium transition-colors shadow-sm"
          >
            <Plus size={18} />
            <span>New Trip</span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-3">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 mt-4 px-2">Recent Trips</div>
          {recentTrips.length === 0 ? (
            <p className="px-2 text-xs leading-relaxed text-slate-400">
              No trips yet — plan a new trip or ask me a question and it will show up here.
            </p>
          ) : (
            <div className="space-y-1">
              {recentTrips.map((trip) => (
                <button
                  key={trip.id}
                  onClick={() => openTrip(trip)}
                  className={`w-full flex items-center gap-3 px-2 py-2 text-sm rounded-md transition-colors text-left ${
                    trip.id === sessionId
                      ? 'bg-blue-50 text-blue-700'
                      : 'text-slate-600 hover:bg-slate-100'
                  }`}
                >
                  <MessageSquare
                    size={16}
                    className={`shrink-0 ${trip.id === sessionId ? 'text-blue-500' : 'text-slate-400'}`}
                  />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate">{trip.title}</span>
                    <span className="block text-[11px] text-slate-400">{relativeDay(trip.updatedAt)}</span>
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="p-4 border-t border-slate-200 shrink-0">
          <button className="flex items-center gap-3 text-sm text-slate-600 hover:text-slate-900 w-full p-2 hover:bg-slate-100 rounded-md transition-colors">
            <Settings size={18} />
            <span>Settings</span>
          </button>
          <button className="flex items-center gap-3 text-sm text-slate-600 hover:text-slate-900 w-full p-2 mt-1 hover:bg-slate-100 rounded-md transition-colors">
            <User size={18} />
            <span>My Profile</span>
          </button>
        </div>
      </div>

      {/* MAIN CHAT AREA */}
      <div className="flex-1 flex flex-col min-w-0 bg-white relative">
        {/* Header */}
        <header
          {...mobileInertAttributes}
          aria-hidden={mobileCanvasActive || undefined}
          className="h-14 border-b border-slate-200 flex items-center px-4 justify-between bg-white/80 backdrop-blur-sm z-10 shrink-0"
        >
          <div className="flex items-center gap-3">
            {!sidebarOpen && (
              <button onClick={() => setSidebarOpen(true)} className="text-slate-500 hover:text-slate-800 p-1">
                <Menu size={20} />
              </button>
            )}
            <h1 className="font-medium text-slate-800">Plan a new trip</h1>
          </div>
          <div className="flex items-center gap-3">
            {latestItinerary && !itineraryCanvasOpen && (
              <button
                ref={openItineraryButtonRef}
                type="button"
                onClick={() => setItineraryCanvasOpen(true)}
                aria-label="Open latest itinerary"
                className="flex items-center gap-1.5 rounded-full border border-blue-100 bg-blue-50 px-3 py-1.5 text-sm font-semibold text-blue-600 transition-colors hover:bg-blue-100"
              >
                <Map size={16} />
                <span className="hidden sm:inline">Itinerary</span>
              </button>
            )}
            {/* EMERGENCY BUTTON */}
            <button 
              onClick={() => setShowEmergency(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-red-50 text-red-600 hover:bg-red-100 rounded-full border border-red-100 transition-colors text-sm font-semibold shadow-sm"
            >
              <ShieldAlert size={16} />
              <span className="hidden sm:inline">Emergency</span>
            </button>
            
            <div className="text-xs font-medium px-2.5 py-1.5 bg-blue-50 text-blue-600 rounded-full border border-blue-100 hidden md:block">
              Wander-Pro Model
            </div>
          </div>
        </header>

        {mode === 'form' ? (
          /* NEW TRIP INITIALIZATION FORM — intercepts the default chat view */
          <NewTripForm
            onSubmit={handleFormSubmit}
            onCancel={() => setMode('chat')}
            submitting={loading}
            error={error}
          />
        ) : (
          <div className="flex min-h-0 flex-1">
            <section
              aria-label="Conversation"
              {...mobileInertAttributes}
              aria-hidden={mobileCanvasActive || undefined}
              className="relative flex min-w-0 flex-1 flex-col bg-white"
            >
            {/* Chat History */}
            <div className="flex-1 overflow-y-auto pb-32">
              {messages.length === 0 && !loading && (
                <div className="max-w-3xl mx-auto px-6 py-12 text-center">
                  <div className="w-16 h-16 bg-blue-100 text-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-6 rotate-3">
                    <Plane size={32} />
                  </div>
                  <h2 className="text-2xl font-bold text-slate-800 mb-2">Where to next?</h2>
                  <p className="text-slate-500 mb-8">I can help you plan itineraries, find hidden gems, or book the best flights.</p>

                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 text-left">
                    {[
                      { icon: <Map className="text-emerald-500" size={20} />, text: "Plan a 3-day itinerary for Kyoto, Japan" },
                      { icon: <Sunrise className="text-orange-500" size={20} />, text: "Find cheap flights to Santorini for July" },
                      { icon: <Calendar className="text-blue-500" size={20} />, text: "What's the best time to visit Patagonia?" }
                    ].map((suggestion, i) => (
                      <button
                        key={i}
                        onClick={() => {
                          setInput(suggestion.text);
                        }}
                        className="flex flex-col gap-3 p-4 border border-slate-200 rounded-xl hover:border-blue-400 hover:shadow-sm bg-white transition-all"
                      >
                        {suggestion.icon}
                        <span className="text-sm font-medium text-slate-700">{suggestion.text}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((msg, idx) => {
                const generation = confirmationGeneration(msg, idx);
                return (
                <ChatMessage
                  key={idx}
                  message={msg}
                  disabled={loading}
                  actionable={Boolean(
                    isChatBudgetConfirmation(msg.budget_confirmation)
                    && isUnexpiredChatBudgetConfirmation(msg.budget_confirmation)
                    && pendingAssessments[sessionId]?.assessmentId
                      === msg.budget_confirmation.budget_assessment_id
                    && pendingAssessments[sessionId]?.generation === generation
                    && !pendingAssessments[sessionId]?.consumed,
                  )}
                  onAcceptRecommendedBudget={(decision) => {
                    if (!isUnexpiredChatBudgetConfirmation(decision)) {
                      setPendingAssessment(sessionId, null);
                      return;
                    }
                    if (!consumePendingAssessment(
                      sessionId,
                      decision.budget_assessment_id,
                      generation,
                    )) return;
                    sendMessage(
                      'Use the provider-grounded recommended budget currently pending for this trip.',
                      'Use recommended budget',
                      {
                        budget_action: 'accept_recommended',
                        budget_assessment_id: decision.budget_assessment_id,
                      },
                    );
                  }}
                />
                );
              })}

              {/* Typing / loading indicator */}
              {loading && (
                <div className="flex gap-4 p-6 bg-slate-50">
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white shadow-sm">
                      <Compass size={18} />
                    </div>
                  </div>
                  <div className="flex items-center gap-2 text-slate-500 pt-2">
                    <Loader2 size={16} className="animate-spin" />
                    <span className="text-sm">Planning your trip…</span>
                  </div>
                </div>
              )}
            </div>

            {/* Input Area */}
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-white via-white to-transparent pt-6 pb-6 px-4 md:px-8">
              <div className="max-w-3xl mx-auto relative">
                <form
                  onSubmit={handleSend}
                  className="bg-white border border-slate-300 rounded-2xl shadow-lg shadow-slate-200/50 flex flex-col overflow-hidden focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500 transition-all"
                >
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSend(e);
                      }
                    }}
                    placeholder="Ask about destinations, flights, or itineraries..."
                    className="w-full resize-none max-h-48 min-h-[56px] p-4 bg-transparent outline-none text-slate-800 placeholder:text-slate-400"
                    rows={1}
                  />

                  <div className="flex items-center justify-between px-3 pb-3">
                    <div className="flex items-center gap-1">
                      <button type="button" className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors tooltip" title="Attach itinerary or photos">
                        <Paperclip size={18} />
                      </button>
                    </div>

                    <button
                      type="submit"
                      disabled={!input.trim() || loading}
                      className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-200 disabled:text-slate-400 text-white p-2 rounded-xl transition-colors flex items-center justify-center"
                    >
                      {loading ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
                    </button>
                  </div>
                </form>
                {error && (
                  <div className="mt-2 flex items-start gap-2 text-xs text-red-600 bg-red-50 border border-red-100 rounded-lg px-3 py-2">
                    <AlertTriangle size={14} className="mt-0.5 shrink-0" />
                    <span>{error}</span>
                  </div>
                )}
              </div>
            </div>
            </section>

            {latestItinerary && itineraryCanvasOpen && (
              <ItineraryCanvas
                snapshot={{
                  itinerary: latestItinerary.itinerary,
                  maps: latestItinerary.maps,
                  budget: latestItinerary.budget,
                  ...(Object.prototype.hasOwnProperty.call(
                    latestItinerary,
                    'destination_country_code',
                  )
                    ? { destination_country_code: latestItinerary.destination_country_code }
                    : {}),
                }}
                onClose={() => setItineraryCanvasOpen(false)}
                isMobileModal={mobileCanvasActive}
              />
            )}
          </div>
        )}
      </div>

      {/* EMERGENCY MODAL */}
      {showEmergency && (
        <div className="absolute inset-0 bg-slate-900/40 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-xl w-full max-w-md overflow-hidden flex flex-col animate-in fade-in zoom-in-95 duration-200">
            <div className="p-4 border-b border-slate-100 flex justify-between items-center bg-red-50/50">
              <div className="flex items-center gap-2 text-red-600 font-bold">
                <ShieldAlert size={20} />
                <h2>Emergency Assistance</h2>
              </div>
              <button onClick={() => setShowEmergency(false)} className="text-slate-400 hover:text-slate-700 bg-white rounded-full p-1 transition-colors">
                <X size={20} />
              </button>
            </div>
            <div className="p-5">
              <p className="text-sm text-slate-500 mb-5">
                Ask the assistant for live emergency information{destination ? ` for ${destination}` : ' based on your current trip'}.
                Tap an action and the answer appears in your chat.
              </p>

              <div className="space-y-3">
                {/* Emergency numbers — ONE button returns police + ambulance + fire */}
                <div className="p-4 border border-red-100 bg-red-50/40 rounded-xl">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-10 h-10 rounded-full bg-red-50 flex items-center justify-center text-red-600 shrink-0">
                      <Phone size={18} />
                    </div>
                    <div>
                      <h4 className="font-bold text-slate-800 text-sm">Emergency Numbers</h4>
                      <p className="text-[11px] text-slate-500">Police · Ambulance · Fire — in one answer</p>
                    </div>
                  </div>
                  <button
                    onClick={emergencyActions.allNumbers}
                    className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-red-600 hover:bg-red-700 text-white text-xs font-semibold transition-colors"
                  >
                    <Phone size={14} /> Get All Emergency Numbers
                  </button>
                </div>

                {/* Police — nearest station */}
                <div className="p-4 border border-slate-200 rounded-xl">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-10 h-10 rounded-full bg-blue-50 flex items-center justify-center text-blue-600 shrink-0">
                      <ShieldAlert size={18} />
                    </div>
                    <h4 className="font-bold text-slate-800 text-sm">Police</h4>
                  </div>
                  <button
                    onClick={emergencyActions.policeNearest}
                    className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold transition-colors"
                  >
                    <Search size={14} /> Find Nearest Police Station
                  </button>
                </div>

                {/* Hospital & Fire — nearest */}
                <div className="p-4 border border-slate-200 rounded-xl">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-10 h-10 rounded-full bg-red-50 flex items-center justify-center text-red-600 shrink-0">
                      <Plus size={18} />
                    </div>
                    <h4 className="font-bold text-slate-800 text-sm">Hospital & Fire Station</h4>
                  </div>
                  <button
                    onClick={emergencyActions.medicalNearest}
                    className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold transition-colors"
                  >
                    <Search size={14} /> Find Nearest Hospital & Fire Station
                  </button>
                </div>

                {/* Embassy — of the user's HOME country */}
                <div className="p-4 border border-slate-200 rounded-xl">
                  <div className="flex items-center gap-3 mb-3">
                    <div className="w-10 h-10 rounded-full bg-slate-100 flex items-center justify-center text-slate-600 shrink-0">
                      <Building size={18} />
                    </div>
                    <div>
                      <h4 className="font-bold text-slate-800 text-sm">
                        Embassy{homeCountry ? ` of ${homeCountry}` : ''}
                      </h4>
                      <p className="text-[11px] text-slate-500">Based on your home country profile</p>
                    </div>
                  </div>
                  <button
                    onClick={emergencyActions.embassyNearest}
                    className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold transition-colors"
                  >
                    <Search size={14} /> Find Nearest Embassy / Consulate
                  </button>
                </div>
              </div>

            </div>
          </div>
        </div>
      )}

    </div>
  );
}
