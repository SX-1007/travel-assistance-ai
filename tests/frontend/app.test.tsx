import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';

const apiMocks = vi.hoisted(() => ({
  getProfile: vi.fn(),
  getChatHistory: vi.fn(),
  submitProfile: vi.fn(),
  sendChatMessage: vi.fn(),
  submitTripForm: vi.fn(),
  geocodePlace: vi.fn(),
  newSessionId: vi.fn(() => 'session-generated'),
}));

const createDeferred = <T,>() => {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
};

const validChatConfirmation = (overrides: Record<string, unknown> = {}) => ({
  status: 'budget_confirmation_required',
  reason: 'insufficient_budget',
  chat_reply: 'Confirm the grounded minimum.',
  budget_assessment_id: 'assessment-valid',
  stated_budget: 500,
  recommended_minimum_budget: 3500,
  base_currency: 'MYR',
  destination_currency: 'CNY',
  expires_at: '2099-08-16T13:00:00+00:00',
  evidence: {
    outbound_flight_price: 1200,
    return_flight_price: 1100,
    hotel_price_per_night: 180,
    hotel_nights: 3,
  },
  itinerary: null,
  daily_geojson_maps: null,
  ...overrides,
});

const validPlanSnapshot = (
  placeName: string = 'Tokyo Tower',
  countryCode: string = 'JP',
  hotel?: Record<string, unknown>,
) => ({
  itinerary: [{
    day: 1,
    date: '2026-08-01',
    flight: null,
    hotel: hotel ? {
      hotel_name: 'Validated Hotel',
      location: {
        lat: 35.659,
        lng: 139.746,
        country_code: countryCode,
      },
      ...hotel,
    } : null,
    activities: [{
      name: placeName,
      type: 'attraction',
      address: `${placeName} address`,
      estimated_cost: 10,
      order: 1,
      location: {
        place_name: placeName,
        latitude: 35.6586,
        longitude: 139.7454,
        country_code: countryCode,
        requested_city: countryCode === 'JP' ? 'Tokyo' : 'Singapore',
        verified_locality: countryCode === 'JP' ? 'Tokyo' : 'Singapore',
      },
    }],
    route: null,
    day_total_cost: 10,
  }],
  maps: {
    '1': {
      type: 'FeatureCollection',
      features: [{
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [139.7454, 35.6586] },
        properties: { name: placeName, type: 'attraction', order: 1 },
      }, ...(hotel ? [{
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [139.746, 35.659] },
        properties: {
          name: String(hotel.hotel_name ?? 'Validated Hotel'),
          type: 'hotel',
          order: 0,
        },
      }] : [])],
    },
  },
  budget: { total: 1000, currency: 'JPY', allocation: { transportation: 1000 } },
  destination_country_code: countryCode,
});

vi.mock('../../fyp_frontend/src/lib/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../fyp_frontend/src/lib/api')>();
  class ApiError extends Error {
    status: number;
    detail: unknown;
    constructor(status: number, detail: unknown, message: string) {
      super(message);
      this.name = 'ApiError';
      this.status = status;
      this.detail = detail;
    }
  }
  return {
    ...actual,
    ...apiMocks,
    ApiError,
    MAPBOX_TOKEN: '',
    apiConfig: { baseUrl: 'http://api.test', userId: 'user-1', hasMapbox: false },
  };
});

vi.mock('../../fyp_frontend/src/lib/mapStatic', () => ({
  buildMarkersMapUrl: vi.fn(() => null),
  buildStaticMapUrl: vi.fn(() => null),
}));

import App from '../../fyp_frontend/src/app/App';
import { findLatestItinerarySnapshot } from '../../fyp_frontend/src/app/components/ItineraryCanvas';


describe('App end-to-end component flows', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  beforeEach(() => {
    localStorage.clear();
    Object.values(apiMocks).forEach((mock) => mock.mockReset());
    apiMocks.newSessionId.mockReturnValue('session-generated');
    apiMocks.submitProfile.mockResolvedValue({ status: 'success', user_id: 'user-1' });
    apiMocks.getChatHistory.mockResolvedValue({ status: 'success', sessions: [] });
    apiMocks.sendChatMessage.mockResolvedValue({
      status: 'success',
      chat_reply: 'Hello from Wander',
      draft_itinerary: [],
      daily_map_info: {},
      itinerary_modified: false,
      total_budget: 0,
      currency: '',
      budget_allocation: {},
    });
  });

  it('shows a loading gate while profile status is pending', () => {
    apiMocks.getProfile.mockReturnValue(new Promise(() => undefined));
    render(<App />);
    expect(screen.getByText(/Loading/)).toBeInTheDocument();
  });

  it('shows onboarding when no complete profile exists', async () => {
    localStorage.setItem('wander_sessions_user-1', JSON.stringify([{
      id: 'stale-session',
      title: 'Private old trip',
      destination: 'Private old trip',
      updatedAt: Date.now(),
      messages: [{ role: 'user', content: 'stale personal message' }],
    }]));
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: false, profile: {} });
    render(<App />);
    expect(await screen.findByText('Welcome to Wander AI')).toBeInTheDocument();
    expect(localStorage.getItem('wander_sessions_user-1')).toBeNull();
    expect(screen.queryByText('Private old trip')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Continue/ }));
    expect(screen.getByText('Please enter your name.')).toBeInTheDocument();
    expect(apiMocks.submitProfile).not.toHaveBeenCalled();
  });

  it('submits trimmed onboarding fields and unlocks the main interface', async () => {
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: false, profile: {} });
    render(<App />);
    fireEvent.change(await screen.findByPlaceholderText('e.g. Alex Tan'), {
      target: { value: '  Alex  ' },
    });
    fireEvent.change(screen.getByPlaceholderText('e.g. Malaysia'), {
      target: { value: ' Malaysia ' },
    });
    fireEvent.change(screen.getByPlaceholderText('e.g. Selangor'), {
      target: { value: ' Selangor ' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Continue/ }));
    await waitFor(() => expect(apiMocks.submitProfile).toHaveBeenCalledWith({
      name: 'Alex',
      origin_country: 'Malaysia',
      origin_state: 'Selangor',
    }));
    expect(await screen.findByText('Where to next?')).toBeInTheDocument();
  });

  it('loads an existing onboarded profile and sends a chat message', async () => {
    apiMocks.getProfile.mockResolvedValue({
      status: 'success',
      onboarded: true,
      profile: { name: 'Alex', home_country: 'Malaysia', home_state: 'Selangor' },
    });
    render(<App />);
    const input = await screen.findByPlaceholderText('Ask about destinations, flights, or itineraries...');
    fireEvent.change(input, { target: { value: 'Best month for Japan?' } });
    fireEvent.submit(input.closest('form')!);
    await waitFor(() => expect(apiMocks.sendChatMessage).toHaveBeenCalledWith(
      'session-generated',
      'Best month for Japan?',
    ));
    expect(await screen.findByText('Hello from Wander')).toBeInTheDocument();
    expect(screen.getAllByText('Best month for Japan?').length).toBeGreaterThan(0);
  });

  it('never exposes API, user, or session diagnostics in the user interface', async () => {
    apiMocks.getProfile.mockResolvedValue({
      status: 'success', onboarded: true, profile: { home_country: 'Malaysia' },
    });
    render(<App />);
    await screen.findByPlaceholderText('Ask about destinations, flights, or itineraries...');
    expect(screen.queryByText(/API:/)).not.toBeInTheDocument();
    expect(screen.queryByText('http://api.test')).not.toBeInTheDocument();
    expect(screen.queryByText('user-1')).not.toBeInTheDocument();
    expect(screen.queryByText('session-g')).not.toBeInTheDocument();
  });

  it('restores persisted Supabase history into the sidebar and chat feed', async () => {
    apiMocks.getProfile.mockResolvedValue({
      status: 'success',
      onboarded: true,
      profile: { home_country: 'Malaysia' },
    });
    apiMocks.getChatHistory.mockResolvedValue({
      status: 'success',
      sessions: [{
        id: 'server-session',
        title: 'Tokyo, Japan',
        destination: 'Tokyo, Japan',
        updated_at: '2026-07-15T08:00:00+00:00',
        messages: [
          { role: 'user', content: 'Plan my Tokyo trip' },
          { role: 'ai', content: 'Here is your saved plan' },
        ],
      }],
    });

    render(<App />);

    expect(await screen.findByText('Here is your saved plan')).toBeInTheDocument();
    expect(screen.getByText('Tokyo, Japan')).toBeInTheDocument();
    expect(apiMocks.getChatHistory).toHaveBeenCalledWith(10);
  });

  it.each([
    '{"draft_itinerary":[{"day":1}]}',
    'Before {"candidate_plan":{"country":"Malaysia"}} after',
  ])('never renders unsafe assistant history on reload: %s', async (unsafe) => {
    apiMocks.getProfile.mockResolvedValue({
      status: 'success', onboarded: true, profile: { home_country: 'Malaysia' },
    });
    apiMocks.getChatHistory.mockResolvedValue({
      status: 'success',
      sessions: [{
        id: 'legacy-unsafe',
        title: 'Singapore',
        destination: 'Singapore',
        updated_at: '2026-07-15T08:00:00+00:00',
        messages: [
          { role: 'user', content: 'Retain this request' },
          { role: 'ai', content: unsafe },
          { role: 'ai', content: 'Safe saved answer.' },
        ],
      }],
    });

    render(<App />);

    expect(await screen.findByText('Retain this request')).toBeInTheDocument();
    expect(screen.getByText('Safe saved answer.')).toBeInTheDocument();
    expect(screen.queryByText(unsafe)).not.toBeInTheDocument();
  });

  it('renders only the newest itinerary in a canvas outside the conversation', async () => {
    apiMocks.getProfile.mockResolvedValue({
      status: 'success',
      onboarded: true,
      profile: { home_country: 'Malaysia' },
    });
    apiMocks.getChatHistory.mockResolvedValue({
      status: 'success',
      sessions: [{
        id: 'canvas-session',
        title: 'Tokyo, Japan',
        destination: 'Tokyo, Japan',
        updated_at: '2026-08-16T08:00:00+00:00',
        messages: [
          {
            role: 'ai',
            content: 'Created the initial Tokyo itinerary.',
            ...validPlanSnapshot('Old activity', 'JP', { hotel_name: 'Old Hotel' }),
          },
          { role: 'user', content: 'Please replace the hotel.' },
          {
            role: 'ai',
            content: 'Changed Day 1 accommodation from Old Hotel to New Hotel.',
            ...validPlanSnapshot('New activity', 'JP', { hotel_name: 'New Hotel' }),
          },
        ],
      }],
    });

    render(<App />);

    const canvas = await screen.findByRole('complementary', { name: 'Latest itinerary' });
    const conversation = screen.getByRole('region', { name: 'Conversation' });
    expect(within(canvas).getByText('New Hotel')).toBeInTheDocument();
    expect(within(canvas).queryByText('Old Hotel')).not.toBeInTheDocument();
    expect(within(conversation).getByText(
      'Changed Day 1 accommodation from Old Hotel to New Hotel.',
    )).toBeInTheDocument();
    expect(within(conversation).queryByText('New Hotel')).not.toBeInTheDocument();
    expect(screen.getAllByText('Day 1')).toHaveLength(1);
  });

  it('renders persisted itineraries whose provider category is a public string', async () => {
    apiMocks.getProfile.mockResolvedValue({
      status: 'success',
      onboarded: true,
      profile: { home_country: 'Malaysia' },
    });
    apiMocks.getChatHistory.mockResolvedValue({
      status: 'success',
      sessions: [{
        id: 'provider-category-session',
        title: 'Taiwan',
        destination: 'Taipei, Taiwan',
        updated_at: '2026-08-16T08:00:00+00:00',
        messages: [{
          role: 'ai',
          content: 'Created the Taiwan itinerary.',
          ...validPlanSnapshot('Taipei Ningxia Night Market', 'TW'),
        }],
      }],
    });

    render(<App />);

    const canvas = await screen.findByRole('complementary', { name: 'Latest itinerary' });
    expect(within(canvas).getByText('Taipei Ningxia Night Market')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Close itinerary' })).toBeInTheDocument();
  });

  it('skips malformed itinerary snapshots in favor of the latest valid version', () => {
    const validSnapshot = validPlanSnapshot(
      'Valid activity',
      'JP',
      { hotel_name: 'Valid Hotel', amenities: [] },
    );
    const validItinerary = validSnapshot.itinerary;

    const malformedSnapshots = [
      { itinerary: [{ day: 2, activities: 'not-an-array' }] },
      { itinerary: [null] },
      { itinerary: [{}] },
      { itinerary: [{ day: 2, date: { unsafe: true }, activities: [] }] },
      { itinerary: [{ day: 2, flight: [{ departure_time: 123 }], activities: [] }] },
      { itinerary: [{ day: 2, flight: { departure_time: 123 }, activities: [] }] },
      { itinerary: [{ day: 2, hotel: { hotel_name: { unsafe: true } }, activities: [] }] },
      {
        itinerary: [{
          day: 2,
          activities: [{ name: 'Museum', type: 'attraction', rating: { unsafe: true } }],
        }],
      },
      {
        itinerary: [{
          day: 2,
          activities: [{ name: 'Museum', type: 'attraction', category: { unsafe: true } }],
        }],
      },
      {
        itinerary: [{
          day: 2,
          route: { profiles: { driving: { distance_km: { unsafe: true } } } },
          activities: [],
        }],
      },
      {
        itinerary: validItinerary,
        maps: { '1': { type: 'FeatureCollection', features: [null] } },
      },
      {
        itinerary: validItinerary,
        budget: { total: 100, currency: { unsafe: true }, allocation: {} },
      },
    ];

    malformedSnapshots.forEach((malformedSnapshot) => {
      const snapshot = findLatestItinerarySnapshot([
        validSnapshot,
        malformedSnapshot,
      ]);
      expect(snapshot?.messageIndex).toBe(0);
      expect(snapshot?.itinerary).toBe(validItinerary);
    });
  });

  it('treats the full-screen mobile canvas as a focus-isolated modal', async () => {
    vi.spyOn(window, 'matchMedia').mockImplementation((query: string) => ({
      matches: query === '(max-width: 1023px)',
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }));
    apiMocks.getProfile.mockResolvedValue({
      status: 'success',
      onboarded: true,
      profile: { home_country: 'Malaysia' },
    });
    apiMocks.getChatHistory.mockResolvedValue({
      status: 'success',
      sessions: [{
        id: 'mobile-canvas-session',
        title: 'Mobile Tokyo trip',
        destination: 'Tokyo, Japan',
        updated_at: '2026-08-16T08:00:00+00:00',
        messages: [{
          role: 'ai',
          content: 'Created the mobile itinerary.',
          ...validPlanSnapshot('Mobile activity', 'JP', {
            hotel_name: 'Mobile Hotel',
            booking_url: 'https://example.com/mobile-hotel',
          }),
        }],
      }],
    });

    render(<App />);

    const dialog = await screen.findByRole('dialog', { name: 'Latest itinerary' });
    const conversation = document.querySelector('section[aria-label="Conversation"]');
    expect(dialog).toHaveAttribute('aria-modal', 'true');
    expect(conversation).toHaveAttribute('inert');
    expect(conversation).toHaveAttribute('aria-hidden', 'true');

    const closeButton = within(dialog).getByRole('button', { name: 'Close itinerary' });
    const bookingLink = within(dialog).getByRole('link', { name: /Book hotel/ });
    await waitFor(() => expect(closeButton).toHaveFocus());

    fireEvent.keyDown(closeButton, { key: 'Tab', shiftKey: true });
    expect(bookingLink).toHaveFocus();
    fireEvent.keyDown(bookingLink, { key: 'Tab' });
    expect(closeButton).toHaveFocus();

    fireEvent.click(closeButton);
    const openButton = await screen.findByRole('button', { name: 'Open latest itinerary' });
    await waitFor(() => expect(openButton).toHaveFocus());
    expect(conversation).not.toHaveAttribute('inert');
    expect(conversation).not.toHaveAttribute('aria-hidden');
  });

  it('keeps the canvas synchronized while switching among saved sessions', async () => {
    apiMocks.getProfile.mockResolvedValue({
      status: 'success',
      onboarded: true,
      profile: { home_country: 'Malaysia' },
    });
    apiMocks.getChatHistory.mockResolvedValue({
      status: 'success',
      sessions: [
        {
          id: 'tokyo-session',
          title: 'Tokyo trip',
          destination: 'Tokyo, Japan',
          updated_at: '2026-08-16T08:00:00+00:00',
          messages: [{
            role: 'ai',
            content: 'Latest Tokyo changes.',
            ...validPlanSnapshot('Tokyo activity', 'JP', { hotel_name: 'Tokyo Hotel' }),
          }],
        },
        {
          id: 'kyoto-session',
          title: 'Kyoto trip',
          destination: 'Kyoto, Japan',
          updated_at: '2026-08-15T08:00:00+00:00',
          messages: [{
            role: 'ai',
            content: 'Latest Kyoto changes.',
            ...validPlanSnapshot('Kyoto activity', 'JP', { hotel_name: 'Kyoto Hotel' }),
          }],
        },
        {
          id: 'seoul-session',
          title: 'Seoul questions',
          destination: 'Seoul, South Korea',
          updated_at: '2026-08-14T08:00:00+00:00',
          messages: [{ role: 'ai', content: 'Seoul has no itinerary yet.' }],
        },
      ],
    });

    render(<App />);

    let canvas = await screen.findByRole('complementary', { name: 'Latest itinerary' });
    expect(within(canvas).getByText('Tokyo Hotel')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Kyoto trip/ }));
    canvas = await screen.findByRole('complementary', { name: 'Latest itinerary' });
    expect(within(canvas).getByText('Kyoto Hotel')).toBeInTheDocument();
    expect(within(canvas).queryByText('Tokyo Hotel')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Seoul questions/ }));
    expect(await screen.findByText('Seoul has no itinerary yet.')).toBeInTheDocument();
    expect(screen.queryByRole('complementary', { name: 'Latest itinerary' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Open latest itinerary' })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Tokyo trip/ }));
    canvas = await screen.findByRole('complementary', { name: 'Latest itinerary' });
    expect(within(canvas).getByText('Tokyo Hotel')).toBeInTheDocument();
    expect(within(canvas).queryByText('Kyoto Hotel')).not.toBeInTheDocument();
  });

  it('skips a legacy single-flight object that never passed the public contract', async () => {
    apiMocks.getProfile.mockResolvedValue({
      status: 'success',
      onboarded: true,
      profile: { home_country: 'Malaysia' },
    });
    apiMocks.getChatHistory.mockResolvedValue({
      status: 'success',
      sessions: [{
        id: 'legacy-flight-session',
        title: 'Legacy flight trip',
        destination: 'Tokyo, Japan',
        updated_at: '2026-08-16T08:00:00+00:00',
        messages: [{
          role: 'ai',
          content: 'Restored the saved flight itinerary.',
          itinerary: [{
            day: 1,
            flight: {
              airline: 'Legacy Air',
              departure_time: '2026-09-01 09:00',
              arrival_time: '2026-09-01 16:00',
              stops: 0,
            },
            activities: [],
            day_total_cost: 400,
          }],
        }],
      }],
    });

    render(<App />);

    expect(await screen.findByText('Restored the saved flight itinerary.')).toBeInTheDocument();
    expect(screen.queryByRole('complementary', { name: 'Latest itinerary' })).not.toBeInTheDocument();
    expect(screen.queryByText('Legacy Air')).not.toBeInTheDocument();
    expect(within(screen.getByRole('region', { name: 'Conversation' })).getByText(
      'Restored the saved flight itinerary.',
    )).toBeInTheDocument();
  });

  it('treats an empty Supabase history as authoritative and removes legacy browser history', async () => {
    localStorage.setItem('wander_sessions_user-1', JSON.stringify([{
      id: 'stale-session',
      title: 'Saudi Arab',
      destination: 'Saudi Arab',
      updatedAt: Date.now(),
      messages: [{ role: 'ai', content: 'stale itinerary from local storage' }],
    }]));
    localStorage.setItem('unrelated_preference', 'keep-me');
    apiMocks.getProfile.mockResolvedValue({
      status: 'success',
      onboarded: true,
      profile: { name: 'Alex', home_country: 'Malaysia', home_state: 'Selangor' },
    });
    apiMocks.getChatHistory.mockResolvedValue({ status: 'success', sessions: [] });

    render(<App />);

    expect(await screen.findByText(/No trips yet/)).toBeInTheDocument();
    expect(screen.getByText('Where to next?')).toBeInTheDocument();
    expect(screen.queryByText('Saudi Arab')).not.toBeInTheDocument();
    expect(screen.queryByText('stale itinerary from local storage')).not.toBeInTheDocument();
    expect(localStorage.getItem('wander_sessions_user-1')).toBeNull();
    expect(localStorage.getItem('unrelated_preference')).toBe('keep-me');
  });

  it('fails closed when Supabase history cannot be verified', async () => {
    localStorage.setItem('wander_sessions_user-1', JSON.stringify([{
      id: 'stale-session',
      title: 'Taiwan',
      destination: 'Taiwan',
      updatedAt: Date.now(),
      messages: [{ role: 'user', content: 'private cached text' }],
    }]));
    apiMocks.getProfile.mockResolvedValue({
      status: 'success',
      onboarded: true,
      profile: { name: 'Alex', home_country: 'Malaysia', home_state: 'Selangor' },
    });
    apiMocks.getChatHistory.mockRejectedValue(new Error('database unavailable'));

    render(<App />);

    expect(await screen.findByText(/No cached history was loaded/)).toBeInTheDocument();
    expect(screen.getByText(/No trips yet/)).toBeInTheDocument();
    expect(screen.queryByText('Taiwan')).not.toBeInTheDocument();
    expect(screen.queryByText('private cached text')).not.toBeInTheDocument();
    expect(localStorage.getItem('wander_sessions_user-1')).toBeNull();
  });

  it('revalidates Supabase when the window regains focus after an external reset', async () => {
    apiMocks.getProfile.mockResolvedValue({
      status: 'success',
      onboarded: true,
      profile: { name: 'Alex', home_country: 'Malaysia', home_state: 'Selangor' },
    });
    apiMocks.getChatHistory.mockResolvedValue({
      status: 'success',
      sessions: [{
        id: 'server-session',
        title: 'China',
        destination: 'China',
        updated_at: '2026-07-15T08:00:00+00:00',
        messages: [{ role: 'ai', content: 'persisted plan' }],
      }],
    });
    render(<App />);
    expect(await screen.findByText('persisted plan')).toBeInTheDocument();

    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: false, profile: {} });
    fireEvent.focus(window);

    expect(await screen.findByText('Welcome to Wander AI')).toBeInTheDocument();
    expect(screen.queryByText('China')).not.toBeInTheDocument();
    expect(screen.queryByText('persisted plan')).not.toBeInTheDocument();
    expect(apiMocks.getProfile).toHaveBeenCalledTimes(2);
  });

  it('hydrates history normally after re-onboarding from an external profile reset', async () => {
    apiMocks.getProfile
      .mockResolvedValueOnce({
        status: 'success',
        onboarded: true,
        profile: { name: 'Alex', home_country: 'Malaysia', home_state: 'Selangor' },
      })
      .mockResolvedValueOnce({ status: 'success', onboarded: false, profile: {} });
    apiMocks.getChatHistory
      .mockResolvedValueOnce({
        status: 'success',
        sessions: [{
          id: 'old-session',
          title: 'Old trip',
          destination: 'Old trip',
          updated_at: '2026-08-15T08:00:00+00:00',
          messages: [{ role: 'ai', content: 'Old saved plan' }],
        }],
      })
      .mockResolvedValueOnce({ status: 'success', sessions: [] });

    render(<App />);
    expect(await screen.findByText('Old saved plan')).toBeInTheDocument();

    fireEvent.focus(window);
    expect(await screen.findByText('Welcome to Wander AI')).toBeInTheDocument();
    fireEvent.change(screen.getByPlaceholderText('e.g. Alex Tan'), {
      target: { value: 'Alex' },
    });
    fireEvent.change(screen.getByPlaceholderText('e.g. Malaysia'), {
      target: { value: 'Malaysia' },
    });
    fireEvent.change(screen.getByPlaceholderText('e.g. Selangor'), {
      target: { value: 'Selangor' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Continue/ }));

    expect(await screen.findByText('Where to next?')).toBeInTheDocument();
    expect(apiMocks.getChatHistory).toHaveBeenCalledTimes(2);
  });

  it('keeps the app visible while focus revalidation is pending', async () => {
    apiMocks.getProfile.mockResolvedValue({
      status: 'success',
      onboarded: true,
      profile: { name: 'Alex', home_country: 'Malaysia', home_state: 'Selangor' },
    });
    apiMocks.getChatHistory.mockResolvedValue({
      status: 'success',
      sessions: [{
        id: 'server-session',
        title: 'Japan',
        destination: 'Japan',
        updated_at: '2026-07-15T08:00:00+00:00',
        messages: [{ role: 'ai', content: 'visible plan' }],
      }],
    });
    render(<App />);
    expect(await screen.findByText('visible plan')).toBeInTheDocument();

    apiMocks.getProfile.mockReturnValue(new Promise(() => undefined));
    fireEvent.focus(window);

    expect(screen.getByText('visible plan')).toBeInTheDocument();
    expect(screen.queryByText(/^Loading/)).not.toBeInTheDocument();
  });

  it('keeps a typed new-trip draft active when delayed background history succeeds', async () => {
    const backgroundHistory = createDeferred<any>();
    apiMocks.getProfile.mockResolvedValue({
      status: 'success',
      onboarded: true,
      profile: { name: 'Alex', home_country: 'Malaysia', home_state: 'Selangor' },
    });
    apiMocks.getChatHistory
      .mockResolvedValueOnce({
        status: 'success',
        sessions: [{
          id: 'japan-session',
          title: 'Japan',
          destination: 'Japan',
          updated_at: '2026-08-15T08:00:00+00:00',
          messages: [{ role: 'ai', content: 'Saved Japan plan' }],
        }],
      })
      .mockReturnValueOnce(backgroundHistory.promise);

    render(<App />);
    expect(await screen.findByText('Saved Japan plan')).toBeInTheDocument();

    fireEvent.focus(window);
    await waitFor(() => expect(apiMocks.getChatHistory).toHaveBeenCalledTimes(2));
    fireEvent.click(screen.getByRole('button', { name: 'New Trip' }));
    fireEvent.change(screen.getByPlaceholderText('e.g. Japan'), {
      target: { value: 'South Korea' },
    });

    await act(async () => {
      backgroundHistory.resolve({
        status: 'success',
        sessions: [{
          id: 'taiwan-session',
          title: 'Taiwan',
          destination: 'Taiwan',
          updated_at: '2026-08-16T08:00:00+00:00',
          messages: [{ role: 'ai', content: 'Saved Taiwan plan' }],
        }],
      });
    });

    expect(screen.getByPlaceholderText('e.g. Japan')).toHaveValue('South Korea');
    expect(screen.getByRole('button', { name: /Taiwan/ })).toBeInTheDocument();
    expect(screen.queryByText('Saved Taiwan plan')).not.toBeInTheDocument();
  });

  it('keeps a typed new-trip draft active when delayed background history fails', async () => {
    const backgroundHistory = createDeferred<any>();
    apiMocks.getProfile.mockResolvedValue({
      status: 'success',
      onboarded: true,
      profile: { name: 'Alex', home_country: 'Malaysia', home_state: 'Selangor' },
    });
    apiMocks.getChatHistory
      .mockResolvedValueOnce({
        status: 'success',
        sessions: [{
          id: 'saved-session',
          title: 'Japan',
          destination: 'Japan',
          updated_at: '2026-08-15T08:00:00+00:00',
          messages: [{ role: 'ai', content: 'Saved Japan plan' }],
        }],
      })
      .mockReturnValueOnce(backgroundHistory.promise);

    render(<App />);
    fireEvent.click(await screen.findByRole('button', { name: 'New Trip' }));
    fireEvent.change(screen.getByPlaceholderText('e.g. Japan'), {
      target: { value: 'New Zealand' },
    });

    fireEvent.focus(window);
    await waitFor(() => expect(apiMocks.getChatHistory).toHaveBeenCalledTimes(2));
    await act(async () => {
      backgroundHistory.reject(new Error('background database outage'));
    });

    expect(screen.getByPlaceholderText('e.g. Japan')).toHaveValue('New Zealand');
    expect(screen.getByRole('button', { name: /Japan/ })).toBeInTheDocument();
    expect(screen.queryByText(/No cached history was loaded/)).not.toBeInTheDocument();
  });

  it('validates the new-trip form before calling the backend', async () => {
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    render(<App />);
    fireEvent.click(await screen.findByRole('button', { name: 'New Trip' }));
    fireEvent.click(screen.getByRole('button', { name: /Generate itinerary/ }));
    expect(screen.getByText('Please enter a destination country.')).toBeInTheDocument();
    expect(apiMocks.submitTripForm).not.toHaveBeenCalled();

    fireEvent.change(screen.getByPlaceholderText('e.g. Japan'), { target: { value: 'Japan' } });
    fireEvent.click(screen.getByRole('button', { name: /Generate itinerary/ }));
    expect(screen.getByText('Please enter at least one destination state or city.')).toBeInTheDocument();
    expect(apiMocks.submitTripForm).not.toHaveBeenCalled();
  });

  it('never renders raw response JSON controls for a successful plan', async () => {
    const plan = validPlanSnapshot();
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    apiMocks.submitTripForm.mockResolvedValue({
      status: 'success',
      chat_reply: 'Your validated itinerary is ready.',
      itinerary: plan.itinerary,
      daily_geojson_maps: plan.maps,
      destination_country_code: plan.destination_country_code,
      total_budget: plan.budget.total,
      currency: plan.budget.currency,
      budget_allocation: plan.budget.allocation,
      session_id: 'validated-session',
    });

    render(<App />);
    fireEvent.click(await screen.findByRole('button', { name: 'New Trip' }));
    fireEvent.change(screen.getByPlaceholderText('e.g. Japan'), { target: { value: 'Japan' } });
    fireEvent.change(screen.getByPlaceholderText('e.g. Kyoto, Osaka'), { target: { value: 'Tokyo' } });
    fireEvent.change(screen.getByLabelText('Start date'), { target: { value: '2026-08-01' } });
    fireEvent.change(screen.getByLabelText('End date'), { target: { value: '2026-08-05' } });
    fireEvent.change(screen.getByPlaceholderText('e.g. 3000'), { target: { value: '5000' } });
    fireEvent.click(screen.getByRole('button', { name: /Generate itinerary/ }));

    expect(await screen.findByText('Tokyo Tower')).toBeInTheDocument();
    expect(screen.queryByText(/Raw response/)).not.toBeInTheDocument();
    expect(screen.queryByText(/draft_itinerary/)).not.toBeInTheDocument();
    expect(screen.queryByText(/daily_map_info/)).not.toBeInTheDocument();
  });

  it('never authors itinerary success copy when the server reply is empty', async () => {
    const plan = validPlanSnapshot();
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    apiMocks.submitTripForm.mockResolvedValue({
      status: 'success',
      chat_reply: '',
      itinerary: plan.itinerary,
      daily_geojson_maps: plan.maps,
      destination_country_code: plan.destination_country_code,
      total_budget: plan.budget.total,
      currency: plan.budget.currency,
      budget_allocation: plan.budget.allocation,
      session_id: 'empty-reply-session',
    });

    render(<App />);
    fireEvent.click(await screen.findByRole('button', { name: 'New Trip' }));
    fireEvent.change(screen.getByPlaceholderText('e.g. Japan'), { target: { value: 'Japan' } });
    fireEvent.change(screen.getByPlaceholderText('e.g. Kyoto, Osaka'), { target: { value: 'Tokyo' } });
    fireEvent.change(screen.getByLabelText('Start date'), { target: { value: '2026-08-01' } });
    fireEvent.change(screen.getByLabelText('End date'), { target: { value: '2026-08-05' } });
    fireEvent.change(screen.getByPlaceholderText('e.g. 3000'), { target: { value: '5000' } });
    fireEvent.click(screen.getByRole('button', { name: /Generate itinerary/ }));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'The planning service did not return a complete itinerary. Please retry.',
    );
    expect(screen.queryByText('Tokyo Tower')).not.toBeInTheDocument();
    expect(screen.queryByText(/Here's your draft itinerary/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Expand the details below/)).not.toBeInTheDocument();
  });

  it('retries initial planning with the immutable accepted form submission once', async () => {
    const plan = validPlanSnapshot();
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    apiMocks.submitTripForm
      .mockResolvedValueOnce({
        status: 'planning_unavailable',
        reason: 'review_unavailable',
        chat_reply: 'Planning is temporarily unavailable. Please retry.',
        retryable: true,
        itinerary: null,
        daily_geojson_maps: null,
      })
      .mockResolvedValueOnce({
        status: 'success',
        chat_reply: 'Your validated itinerary is ready.',
        itinerary: plan.itinerary,
        daily_geojson_maps: plan.maps,
        destination_country_code: plan.destination_country_code,
        total_budget: plan.budget.total,
        currency: plan.budget.currency,
        budget_allocation: plan.budget.allocation,
        session_id: 'retried-session',
      });

    render(<App />);
    fireEvent.click(await screen.findByRole('button', { name: 'New Trip' }));
    fireEvent.change(screen.getByPlaceholderText('e.g. Japan'), { target: { value: 'Japan' } });
    fireEvent.change(screen.getByPlaceholderText('e.g. Kyoto, Osaka'), { target: { value: 'Tokyo' } });
    fireEvent.change(screen.getByLabelText('Start date'), { target: { value: '2026-08-01' } });
    fireEvent.change(screen.getByLabelText('End date'), { target: { value: '2026-08-05' } });
    fireEvent.change(screen.getByPlaceholderText('e.g. 3000'), { target: { value: '5000' } });
    fireEvent.click(screen.getByRole('button', { name: /Generate itinerary/ }));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Planning is temporarily unavailable. Please retry.',
    );
    fireEvent.change(screen.getByPlaceholderText('e.g. Japan'), { target: { value: 'China' } });
    const retry = screen.getByRole('button', { name: 'Retry planning' });
    fireEvent.click(retry);
    fireEvent.click(retry);

    await waitFor(() => expect(apiMocks.submitTripForm).toHaveBeenCalledTimes(2));
    expect(apiMocks.submitTripForm).toHaveBeenNthCalledWith(2, {
      country: 'Japan',
      city: ['Tokyo'],
      num_people: 1,
      total_budget: 5000,
      start_date: '2026-08-01',
      end_date: '2026-08-05',
    });
    expect(await screen.findByText('Tokyo Tower')).toBeInTheDocument();
  });

  it('submits origin-free trip data and renders the returned itinerary', async () => {
    const plan = validPlanSnapshot();
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    apiMocks.submitTripForm.mockResolvedValue({
      status: 'success',
      chat_reply: 'Your validated Japan itinerary is ready.',
      itinerary: plan.itinerary,
      daily_geojson_maps: plan.maps,
      destination_country_code: plan.destination_country_code,
      total_budget: 160000,
      currency: 'JPY',
      budget_allocation: { accommodation: 160000 },
      session_id: 'server-session',
    });
    render(<App />);
    fireEvent.click(await screen.findByRole('button', { name: 'New Trip' }));
    fireEvent.change(screen.getByPlaceholderText('e.g. Japan'), { target: { value: 'Japan' } });
    fireEvent.change(screen.getByPlaceholderText('e.g. Kyoto, Osaka'), { target: { value: 'Tokyo, Kyoto' } });
    fireEvent.change(screen.getByLabelText('Start date'), { target: { value: '2026-08-01' } });
    fireEvent.change(screen.getByLabelText('End date'), { target: { value: '2026-08-05' } });
    fireEvent.change(screen.getByPlaceholderText('e.g. 3000'), { target: { value: '5000' } });
    fireEvent.click(screen.getByRole('button', { name: /Generate itinerary/ }));
    await waitFor(() => expect(apiMocks.submitTripForm).toHaveBeenCalledWith({
      country: 'Japan',
      city: ['Tokyo', 'Kyoto'],
      num_people: 1,
      total_budget: 5000,
      start_date: '2026-08-01',
      end_date: '2026-08-05',
    }));
    expect(await screen.findByText('Day 1')).toBeInTheDocument();
    expect(screen.getAllByText(/160,000/).length).toBeGreaterThan(0);
    expect(screen.getAllByText('JPY').length).toBeGreaterThan(0);
    expect(JSON.stringify(apiMocks.submitTripForm.mock.calls[0][0])).not.toContain('origin_');

    const canvas = screen.getByRole('complementary', { name: 'Latest itinerary' });
    fireEvent.click(within(canvas).getByRole('button', { name: 'Close itinerary' }));
    expect(screen.queryByRole('complementary', { name: 'Latest itinerary' })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Open latest itinerary' }));
    expect(screen.getByRole('complementary', { name: 'Latest itinerary' })).toBeInTheDocument();
    expect(screen.getByText('Day 1')).toBeInTheDocument();
  });

  it('asks for a grounded minimum without sending an AI-generated budget', async () => {
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    apiMocks.submitTripForm.mockResolvedValue({
      status: 'budget_confirmation_required',
      reason: 'recommendation_requested',
      chat_reply: 'Confirm the grounded minimum before planning.',
      budget_assessment_id: 'assessment-unknown',
      stated_budget: null,
      recommended_minimum_budget: 3200,
      base_currency: 'MYR',
      destination_currency: 'JPY',
      expires_at: '2026-08-16T09:00:00Z',
      evidence: {
        outbound_flight_price: 40000,
        return_flight_price: 38000,
        hotel_price_per_night: 12000,
        hotel_nights: 4,
      },
      itinerary: null,
      daily_geojson_maps: null,
    });

    render(<App />);
    fireEvent.click(await screen.findByRole('button', { name: 'New Trip' }));
    fireEvent.change(screen.getByPlaceholderText('e.g. Japan'), { target: { value: 'Japan' } });
    fireEvent.change(screen.getByPlaceholderText('e.g. Kyoto, Osaka'), { target: { value: 'Tokyo' } });
    fireEvent.change(screen.getByLabelText('Start date'), { target: { value: '2026-08-01' } });
    fireEvent.change(screen.getByLabelText('End date'), { target: { value: '2026-08-05' } });
    fireEvent.click(screen.getByRole('checkbox', { name: /I don't know my budget/i }));
    fireEvent.click(screen.getByRole('button', { name: /Generate itinerary/ }));

    await waitFor(() => expect(apiMocks.submitTripForm).toHaveBeenCalledWith({
      country: 'Japan',
      city: ['Tokyo'],
      num_people: 1,
      request_budget_recommendation: true,
      start_date: '2026-08-01',
      end_date: '2026-08-05',
    }));
    expect(await screen.findByText('Budget confirmation required')).toBeInTheDocument();
    expect(screen.getByText(/MYR\s*3,200/)).toBeInTheDocument();
    expect(screen.queryByText('Day 1')).not.toBeInTheDocument();
    expect(screen.queryByRole('complementary', { name: 'Latest itinerary' })).not.toBeInTheDocument();
  });

  it('does not render an itinerary until the insufficient budget recommendation is confirmed', async () => {
    const plan = validPlanSnapshot();
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    apiMocks.submitTripForm
      .mockResolvedValueOnce({
        status: 'budget_confirmation_required',
        reason: 'insufficient_budget',
        chat_reply: 'The entered budget is below the grounded minimum.',
        budget_assessment_id: 'assessment-low',
        stated_budget: 500,
        recommended_minimum_budget: 3200,
        base_currency: 'MYR',
        destination_currency: 'JPY',
        expires_at: '2026-08-16T09:00:00Z',
        evidence: {
          outbound_flight_price: 40000,
          return_flight_price: 38000,
          hotel_price_per_night: 12000,
          hotel_nights: 4,
        },
        itinerary: null,
        daily_geojson_maps: null,
      })
      .mockResolvedValueOnce({
        status: 'success',
        chat_reply: 'Planned after budget confirmation.',
        itinerary: plan.itinerary,
        daily_geojson_maps: plan.maps,
        destination_country_code: plan.destination_country_code,
        total_budget: 100000,
        currency: 'JPY',
        budget_allocation: { transportation: 100000 },
        session_id: 'confirmed-session',
      });

    render(<App />);
    fireEvent.click(await screen.findByRole('button', { name: 'New Trip' }));
    fireEvent.change(screen.getByPlaceholderText('e.g. Japan'), { target: { value: 'Japan' } });
    fireEvent.change(screen.getByPlaceholderText('e.g. Kyoto, Osaka'), { target: { value: 'Tokyo' } });
    fireEvent.change(screen.getByLabelText('Start date'), { target: { value: '2026-08-01' } });
    fireEvent.change(screen.getByLabelText('End date'), { target: { value: '2026-08-05' } });
    fireEvent.change(screen.getByPlaceholderText('e.g. 3000'), { target: { value: '500' } });
    fireEvent.click(screen.getByRole('button', { name: /Generate itinerary/ }));

    expect(await screen.findByText('Budget confirmation required')).toBeInTheDocument();
    expect(screen.getByText(/Your entered budget.*MYR\s*500/i)).toBeInTheDocument();
    expect(screen.getByText(/25% transportation/i)).toBeInTheDocument();
    expect(screen.getByText(/35% accommodation/i)).toBeInTheDocument();
    expect(screen.getByText(/Outbound flight.*JPY\s*40,000/i)).toBeInTheDocument();
    expect(screen.getByText(/Return flight.*JPY\s*38,000/i)).toBeInTheDocument();
    expect(screen.getByText(/Hotel.*4 nights.*JPY\s*12,000/i)).toBeInTheDocument();
    expect(apiMocks.submitTripForm).toHaveBeenCalledTimes(1);
    expect(screen.queryByText('Day 1')).not.toBeInTheDocument();
    expect(screen.queryByRole('complementary', { name: 'Latest itinerary' })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Use recommended budget/i }));
    await waitFor(() => expect(apiMocks.submitTripForm).toHaveBeenNthCalledWith(2, {
      country: 'Japan',
      city: ['Tokyo'],
      num_people: 1,
      total_budget: 3200,
      budget_assessment_id: 'assessment-low',
      start_date: '2026-08-01',
      end_date: '2026-08-05',
    }));
    expect(await screen.findByText('Day 1')).toBeInTheDocument();
  });

  it('invalidates a pending budget assessment when trip-defining input changes', async () => {
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    apiMocks.submitTripForm.mockResolvedValue({
      status: 'budget_confirmation_required',
      reason: 'insufficient_budget',
      chat_reply: 'Budget too low.',
      budget_assessment_id: 'assessment-old-trip',
      stated_budget: 500,
      recommended_minimum_budget: 3200,
      base_currency: 'MYR',
      destination_currency: 'JPY',
      expires_at: '2026-08-16T09:00:00Z',
      evidence: {
        outbound_flight_price: 40000,
        return_flight_price: 38000,
        hotel_price_per_night: 12000,
        hotel_nights: 4,
      },
      itinerary: null,
      daily_geojson_maps: null,
    });

    render(<App />);
    fireEvent.click(await screen.findByRole('button', { name: 'New Trip' }));
    const destination = screen.getByPlaceholderText('e.g. Japan');
    fireEvent.change(destination, { target: { value: 'Japan' } });
    fireEvent.change(screen.getByPlaceholderText('e.g. Kyoto, Osaka'), { target: { value: 'Tokyo' } });
    fireEvent.change(screen.getByLabelText('Start date'), { target: { value: '2026-08-01' } });
    fireEvent.change(screen.getByLabelText('End date'), { target: { value: '2026-08-05' } });
    fireEvent.change(screen.getByPlaceholderText('e.g. 3000'), { target: { value: '500' } });
    fireEvent.click(screen.getByRole('button', { name: /Generate itinerary/ }));
    expect(await screen.findByText('Budget confirmation required')).toBeInTheDocument();

    fireEvent.change(destination, { target: { value: 'Thailand' } });
    expect(screen.queryByText('Budget confirmation required')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Use recommended budget/i })).not.toBeInTheDocument();
  });

  it('submits an edited amount as a fresh user budget without confirmation metadata', async () => {
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    const confirmation = {
      status: 'budget_confirmation_required',
      reason: 'insufficient_budget',
      chat_reply: 'Budget too low.',
      budget_assessment_id: 'assessment-old-budget',
      stated_budget: 500,
      recommended_minimum_budget: 3200,
      base_currency: 'MYR',
      destination_currency: 'JPY',
      expires_at: '2026-08-16T09:00:00Z',
      evidence: {
        outbound_flight_price: 40000,
        return_flight_price: 38000,
        hotel_price_per_night: 12000,
        hotel_nights: 4,
      },
      itinerary: null,
      daily_geojson_maps: null,
    };
    apiMocks.submitTripForm.mockResolvedValue(confirmation);

    render(<App />);
    fireEvent.click(await screen.findByRole('button', { name: 'New Trip' }));
    fireEvent.change(screen.getByPlaceholderText('e.g. Japan'), { target: { value: 'Japan' } });
    fireEvent.change(screen.getByPlaceholderText('e.g. Kyoto, Osaka'), { target: { value: 'Tokyo' } });
    fireEvent.change(screen.getByLabelText('Start date'), { target: { value: '2026-08-01' } });
    fireEvent.change(screen.getByLabelText('End date'), { target: { value: '2026-08-05' } });
    const amount = screen.getByPlaceholderText('e.g. 3000');
    fireEvent.change(amount, { target: { value: '500' } });
    fireEvent.click(screen.getByRole('button', { name: /Generate itinerary/ }));
    expect(await screen.findByText('Budget confirmation required')).toBeInTheDocument();

    fireEvent.change(amount, { target: { value: '7000' } });
    fireEvent.click(screen.getByRole('button', { name: /Generate itinerary/ }));
    await waitFor(() => expect(apiMocks.submitTripForm).toHaveBeenNthCalledWith(2, {
      country: 'Japan',
      city: ['Tokyo'],
      num_people: 1,
      total_budget: 7000,
      start_date: '2026-08-01',
      end_date: '2026-08-05',
    }));
    expect(apiMocks.submitTripForm.mock.calls[1][0]).not.toHaveProperty('budget_assessment_id');
  });

  it('fails closed when the grounded budget check is unavailable', async () => {
    const plan = validPlanSnapshot();
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    apiMocks.submitTripForm
      .mockResolvedValueOnce({
        status: 'budget_check_unavailable',
        reason: 'provider_data_unavailable',
        chat_reply: 'Current price data is unavailable. Retry the budget check before planning.',
        itinerary: null,
        daily_geojson_maps: null,
      })
      .mockResolvedValueOnce({
        status: 'success',
        chat_reply: 'Planned after a successful retry.',
        itinerary: plan.itinerary,
        daily_geojson_maps: plan.maps,
        destination_country_code: plan.destination_country_code,
        total_budget: 160000,
        currency: 'JPY',
        budget_allocation: { transportation: 160000 },
        session_id: 'retry-session',
      });

    render(<App />);
    fireEvent.click(await screen.findByRole('button', { name: 'New Trip' }));
    fireEvent.change(screen.getByPlaceholderText('e.g. Japan'), { target: { value: 'Japan' } });
    fireEvent.change(screen.getByPlaceholderText('e.g. Kyoto, Osaka'), { target: { value: 'Tokyo' } });
    fireEvent.change(screen.getByLabelText('Start date'), { target: { value: '2026-08-01' } });
    fireEvent.change(screen.getByLabelText('End date'), { target: { value: '2026-08-05' } });
    fireEvent.change(screen.getByPlaceholderText('e.g. 3000'), { target: { value: '5000' } });
    fireEvent.click(screen.getByRole('button', { name: /Generate itinerary/ }));

    expect(await screen.findByText(/Current price data is unavailable/)).toBeInTheDocument();
    expect(screen.queryByText('Day 1')).not.toBeInTheDocument();
    expect(screen.queryByRole('complementary', { name: 'Latest itinerary' })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Retry budget check/i }));
    await waitFor(() => expect(apiMocks.submitTripForm).toHaveBeenCalledTimes(2));
    expect(await screen.findByText('Day 1')).toBeInTheDocument();
  });

  it('reopens a closed canvas only when a newer itinerary revision arrives', async () => {
    const firstPlan = validPlanSnapshot('First activity', 'JP', { hotel_name: 'First Hotel' });
    const revisedPlan = validPlanSnapshot('Revised activity', 'JP', { hotel_name: 'Revised Hotel' });
    apiMocks.getProfile.mockResolvedValue({
      status: 'success',
      onboarded: true,
      profile: { home_country: 'Malaysia' },
    });
    apiMocks.getChatHistory.mockResolvedValue({
      status: 'success',
      sessions: [{
        id: 'revision-session',
        title: 'Osaka, Japan',
        destination: 'Osaka, Japan',
        updated_at: '2026-08-16T08:00:00+00:00',
        messages: [{
          role: 'ai',
          content: 'Created the Osaka itinerary.',
          ...firstPlan,
        }],
      }],
    });
    apiMocks.sendChatMessage
      .mockResolvedValueOnce({
        status: 'success',
        chat_reply: 'Changed Day 1 accommodation from First Hotel to Revised Hotel.',
        draft_itinerary: revisedPlan.itinerary,
        daily_map_info: revisedPlan.maps,
        destination_country_code: revisedPlan.destination_country_code,
        itinerary_modified: true,
        total_budget: 1000,
        currency: 'JPY',
        budget_allocation: { transportation: 1000 },
        budget_confirmation: null,
      })
      .mockResolvedValueOnce({
        status: 'success',
        chat_reply: 'September is usually warm in Osaka.',
        draft_itinerary: revisedPlan.itinerary,
        daily_map_info: revisedPlan.maps,
        destination_country_code: revisedPlan.destination_country_code,
        itinerary_modified: false,
        total_budget: 1000,
        currency: 'JPY',
        budget_allocation: { transportation: 1000 },
        budget_confirmation: null,
      });

    render(<App />);

    let canvas = await screen.findByRole('complementary', { name: 'Latest itinerary' });
    fireEvent.click(within(canvas).getByRole('button', { name: 'Close itinerary' }));

    const input = screen.getByPlaceholderText('Ask about destinations, flights, or itineraries...');
    fireEvent.change(input, { target: { value: 'Replace my hotel' } });
    fireEvent.submit(input.closest('form')!);

    canvas = await screen.findByRole('complementary', { name: 'Latest itinerary' });
    expect(within(canvas).getByText('Revised Hotel')).toBeInTheDocument();
    expect(screen.getByText(
      'Changed Day 1 accommodation from First Hotel to Revised Hotel.',
    )).toBeInTheDocument();

    fireEvent.click(within(canvas).getByRole('button', { name: 'Close itinerary' }));
    fireEvent.change(input, { target: { value: 'What is the September weather?' } });
    fireEvent.submit(input.closest('form')!);
    expect(await screen.findByText('September is usually warm in Osaka.')).toBeInTheDocument();
    expect(screen.queryByRole('complementary', { name: 'Latest itinerary' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Open latest itinerary' })).toBeInTheDocument();
  });

  it('keeps the accepted itinerary while a chat budget confirmation is pending, then replaces it after acceptance', async () => {
    const acceptedPlan = validPlanSnapshot(
      'Accepted activity', 'CN', { hotel_name: 'Accepted Hotel' },
    );
    const confirmedPlan = validPlanSnapshot(
      'Confirmed activity', 'CN', { hotel_name: 'Confirmed Budget Hotel' },
    );
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    apiMocks.getChatHistory.mockResolvedValue({
      status: 'success',
      sessions: [{
        id: 'budget-session',
        title: 'Beijing',
        destination: 'Beijing, China',
        updated_at: '2026-08-16T08:00:00+00:00',
        messages: [{
          role: 'ai',
          content: 'Accepted itinerary.',
          ...acceptedPlan,
        }],
      }],
    });
    apiMocks.sendChatMessage
      .mockResolvedValueOnce({
        status: 'budget_confirmation_required',
        chat_reply: 'RM 500 is below the grounded minimum. Confirm RM 3,500 first.',
        draft_itinerary: [],
        daily_map_info: {},
        itinerary_modified: false,
        total_budget: 0,
        currency: 'CNY',
        budget_allocation: { transportation: 3500 },
        budget_confirmation: {
          status: 'budget_confirmation_required',
          reason: 'insufficient_budget',
          chat_reply: 'Confirm the grounded minimum.',
          budget_assessment_id: 'assessment-1',
          stated_budget: 500,
          recommended_minimum_budget: 3500,
          base_currency: 'MYR',
          destination_currency: 'CNY',
          expires_at: '2099-08-16T13:00:00+00:00',
          evidence: {
            outbound_flight_price: 1200,
            return_flight_price: 1100,
            hotel_price_per_night: 180,
            hotel_nights: 3,
          },
          itinerary: null,
          daily_geojson_maps: null,
        },
      })
      .mockResolvedValueOnce({
        status: 'success',
        chat_reply: 'Confirmed the grounded budget.',
        draft_itinerary: confirmedPlan.itinerary,
        daily_map_info: confirmedPlan.maps,
        destination_country_code: confirmedPlan.destination_country_code,
        itinerary_modified: true,
        total_budget: 3500,
        currency: 'CNY',
        budget_allocation: { transportation: 3500 },
        budget_confirmation: null,
      });

    render(<App />);
    const input = await screen.findByPlaceholderText('Ask about destinations, flights, or itineraries...');
    fireEvent.change(input, { target: { value: 'Change my budget to RM 500' } });
    fireEvent.submit(input.closest('form')!);

    const card = await screen.findByRole('region', { name: 'Chat budget confirmation' });
    expect(within(card).getByText(/RM\s*500/)).toBeInTheDocument();
    expect(within(card).getByText(/RM\s*3,500/)).toBeInTheDocument();
    expect(screen.queryByText(/Raw response \(itinerary \/ map data\)/)).not.toBeInTheDocument();
    const oldCanvas = screen.getByRole('complementary', { name: 'Latest itinerary' });
    expect(within(oldCanvas).getByText('Accepted Hotel')).toBeInTheDocument();

    fireEvent.click(within(card).getByRole('button', { name: 'Use recommended budget' }));
    await waitFor(() => expect(apiMocks.sendChatMessage).toHaveBeenNthCalledWith(
      2,
      'budget-session',
      'Use the provider-grounded recommended budget currently pending for this trip.',
      {
        budget_action: 'accept_recommended',
        budget_assessment_id: 'assessment-1',
      },
    ));
    const newCanvas = await screen.findByRole('complementary', { name: 'Latest itinerary' });
    expect(within(newCanvas).getByText('Confirmed Budget Hotel')).toBeInTheDocument();
    expect(within(newCanvas).queryByText('Accepted Hotel')).not.toBeInTheDocument();
    fireEvent.click(within(card).getByRole('button', { name: 'Use recommended budget' }));
    expect(apiMocks.sendChatMessage).toHaveBeenCalledTimes(2);
  });

  it('renders a restored unknown-budget confirmation without requesting a new chat response', async () => {
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    apiMocks.getChatHistory.mockResolvedValue({
      status: 'success',
      sessions: [{
        id: 'restored-budget-session',
        title: 'Tokyo',
        destination: 'Tokyo, Japan',
        updated_at: '2026-08-16T08:00:00+00:00',
        messages: [{
          role: 'ai',
          content: 'Please confirm.',
          budget_confirmation: {
            status: 'budget_confirmation_required',
            reason: 'recommendation_requested',
            chat_reply: 'Confirm the grounded minimum.',
            budget_assessment_id: 'assessment-restored',
            stated_budget: null,
            recommended_minimum_budget: 3200,
            base_currency: 'MYR',
            destination_currency: 'JPY',
            expires_at: '2026-08-16T13:00:00+00:00',
            evidence: {
              outbound_flight_price: 40000,
              return_flight_price: 38000,
              hotel_price_per_night: 12000,
              hotel_nights: 4,
            },
            itinerary: null,
            daily_geojson_maps: null,
          },
        }],
      }],
    });

    render(<App />);

    const card = await screen.findByRole('region', { name: 'Chat budget confirmation' });
    expect(within(card).getByText(/RM\s*3,200/)).toBeInTheDocument();
    expect(within(card).getByRole('button', { name: 'Use recommended budget' })).toBeDisabled();
    expect(apiMocks.sendChatMessage).not.toHaveBeenCalled();
  });

  it('consumes a restored confirmation after an unavailable acceptance response', async () => {
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    apiMocks.getChatHistory.mockResolvedValue({
      status: 'success',
      sessions: [{
        id: 'consumed-session', title: 'Beijing', destination: 'Beijing', updated_at: '2026-08-16T08:00:00Z',
        messages: [{ role: 'ai', content: 'Confirm budget.', budget_confirmation: validChatConfirmation() }],
      }],
    });
    apiMocks.sendChatMessage.mockResolvedValue({
      status: 'budget_check_unavailable',
      chat_reply: 'Provider data is temporarily unavailable.',
      draft_itinerary: [], daily_map_info: {}, itinerary_modified: false,
      total_budget: 0, currency: 'CNY', budget_allocation: {},
    });
    render(<App />);
    const card = await screen.findByRole('region', { name: 'Chat budget confirmation' });
    const button = within(card).getByRole('button', { name: 'Use recommended budget' });
    fireEvent.click(button);
    await waitFor(() => expect(apiMocks.sendChatMessage).toHaveBeenCalledTimes(1));
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(apiMocks.sendChatMessage).toHaveBeenCalledTimes(1);
  });

  it('allows only the newest restored confirmation to be accepted', async () => {
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    apiMocks.getChatHistory.mockResolvedValue({
      status: 'success',
      sessions: [{
        id: 'superseded-session', title: 'Beijing', destination: 'Beijing', updated_at: '2026-08-16T08:00:00Z',
        messages: [
          { role: 'ai', content: 'Old confirmation.', budget_confirmation: validChatConfirmation({ budget_assessment_id: 'assessment-old' }) },
          { role: 'ai', content: 'New confirmation.', budget_confirmation: validChatConfirmation({ budget_assessment_id: 'assessment-new' }) },
        ],
      }],
    });
    render(<App />);
    const buttons = await screen.findAllByRole('button', { name: 'Use recommended budget' });
    expect(buttons[0]).toBeDisabled();
    expect(buttons[1]).toBeEnabled();
  });

  it('ignores malformed restored confirmation data without crashing the chat', async () => {
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    apiMocks.getChatHistory.mockResolvedValue({
      status: 'success',
      sessions: [{
        id: 'malformed-history-session', title: 'Beijing', destination: 'Beijing', updated_at: '2026-08-16T08:00:00Z',
        messages: [{
          role: 'ai', content: 'Confirmation could not be loaded.',
          budget_confirmation: validChatConfirmation({ evidence: null, budget_assessment_id: '   ' }),
        }],
      }],
    });
    render(<App />);
    expect(await screen.findByText('Confirmation could not be loaded.')).toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Chat budget confirmation' })).not.toBeInTheDocument();
  });

  it('preserves the accepted itinerary and maps when a success response lacks a valid live map snapshot', async () => {
    const acceptedPlan = validPlanSnapshot(
      'Accepted activity', 'CN', { hotel_name: 'Accepted Hotel' },
    );
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    apiMocks.getChatHistory.mockResolvedValue({
      status: 'success',
      sessions: [{
        id: 'map-session', title: 'Beijing', destination: 'Beijing', updated_at: '2026-08-16T08:00:00Z',
        messages: [{
          role: 'ai', content: 'Accepted itinerary.',
          ...acceptedPlan,
        }],
      }],
    });
    apiMocks.sendChatMessage.mockResolvedValue({
      status: 'success', chat_reply: 'Changed itinerary.', itinerary_modified: true,
      draft_itinerary: [{ day: 1, hotel: { hotel_name: 'Invalid Replacement Hotel' }, activities: [] }],
      daily_map_info: null, total_budget: 1200, currency: 'CNY', budget_allocation: {},
    });
    render(<App />);
    const input = await screen.findByPlaceholderText('Ask about destinations, flights, or itineraries...');
    fireEvent.change(input, { target: { value: 'Change my hotel' } });
    fireEvent.submit(input.closest('form')!);
    const canvas = await screen.findByRole('complementary', { name: 'Latest itinerary' });
    expect(within(canvas).getByText('Accepted Hotel')).toBeInTheDocument();
    expect(within(canvas).queryByText('Invalid Replacement Hotel')).not.toBeInTheDocument();
  });

  it('does not append an accepted response after switching from its originating session', async () => {
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    const deferred = createDeferred<any>();
    apiMocks.getChatHistory.mockResolvedValue({
      status: 'success',
      sessions: [{
        id: 'session-a', title: 'Beijing', destination: 'Beijing', updated_at: '2026-08-16T08:00:00Z',
        messages: [{ role: 'ai', content: 'Confirm budget.', budget_confirmation: validChatConfirmation() }],
      }],
    });
    apiMocks.sendChatMessage.mockReturnValue(deferred.promise);
    render(<App />);
    const card = await screen.findByRole('region', { name: 'Chat budget confirmation' });
    fireEvent.click(within(card).getByRole('button', { name: 'Use recommended budget' }));
    fireEvent.click(screen.getByRole('button', { name: 'New Trip' }));
    await act(async () => deferred.resolve({
      status: 'success', chat_reply: 'A response', itinerary_modified: true,
      draft_itinerary: [{ day: 1, hotel: { hotel_name: 'Session A Hotel' }, activities: [] }],
      daily_map_info: {}, total_budget: 3500, currency: 'CNY', budget_allocation: {},
    }));
    expect(screen.getByPlaceholderText('e.g. Japan')).toBeInTheDocument();
    expect(screen.queryByText('A response')).not.toBeInTheDocument();
    expect(screen.queryByText('Session A Hotel')).not.toBeInTheDocument();
  });

  it('keeps the accepted itinerary when an unavailable budget check includes malformed itinerary data', async () => {
    const acceptedPlan = validPlanSnapshot(
      'Accepted activity', 'KR', { hotel_name: 'Accepted Hotel' },
    );
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    apiMocks.getChatHistory.mockResolvedValue({
      status: 'success',
      sessions: [{
        id: 'unavailable-budget-session',
        title: 'Seoul',
        destination: 'Seoul, South Korea',
        updated_at: '2026-08-16T08:00:00+00:00',
        messages: [{
          role: 'ai',
          content: 'Accepted itinerary.',
          ...acceptedPlan,
        }],
      }],
    });
    apiMocks.sendChatMessage.mockResolvedValue({
      status: 'budget_check_unavailable',
      chat_reply: 'Provider data is unavailable. Please try again later.',
      draft_itinerary: [{ day: 1, hotel: { hotel_name: 'Do Not Replace Hotel' }, activities: [] }],
      daily_map_info: {},
      itinerary_modified: true,
      total_budget: 0,
      currency: 'KRW',
      budget_allocation: {},
    });

    render(<App />);
    const input = await screen.findByPlaceholderText('Ask about destinations, flights, or itineraries...');
    fireEvent.change(input, { target: { value: 'Use a new budget' } });
    fireEvent.submit(input.closest('form')!);

    expect((await screen.findAllByText(/Provider data is unavailable/)).length).toBeGreaterThan(0);
    const canvas = screen.getByRole('complementary', { name: 'Latest itinerary' });
    expect(within(canvas).getByText('Accepted Hotel')).toBeInTheDocument();
    expect(within(canvas).queryByText('Do Not Replace Hotel')).not.toBeInTheDocument();
  });

  it('preserves a closed accepted canvas after chat planning becomes unavailable', async () => {
    const accepted = validPlanSnapshot('Accepted Tokyo Tower');
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    apiMocks.getChatHistory.mockResolvedValue({
      status: 'success',
      sessions: [{
        id: 'planning-unavailable-session',
        title: 'Tokyo',
        destination: 'Tokyo, Japan',
        updated_at: '2026-08-16T08:00:00Z',
        messages: [{
          role: 'ai',
          content: 'Accepted itinerary.',
          itinerary: accepted.itinerary,
          maps: accepted.maps,
          budget: accepted.budget,
          destination_country_code: accepted.destination_country_code,
        }],
      }],
    });
    apiMocks.sendChatMessage.mockResolvedValue({
      status: 'planning_unavailable',
      reason: 'validation_failed',
      chat_reply: 'Planning is temporarily unavailable. Your accepted plan is unchanged.',
      retryable: true,
      draft_itinerary: [],
      daily_map_info: {},
      itinerary_modified: false,
      total_budget: 0,
      currency: '',
      budget_allocation: {},
      budget_confirmation: null,
    });

    render(<App />);
    const canvas = await screen.findByRole('complementary', { name: 'Latest itinerary' });
    fireEvent.click(within(canvas).getByRole('button', { name: 'Close itinerary' }));
    expect(screen.queryByRole('complementary', { name: 'Latest itinerary' })).not.toBeInTheDocument();

    const input = screen.getByPlaceholderText('Ask about destinations, flights, or itineraries...');
    fireEvent.change(input, { target: { value: 'Replace the attraction' } });
    fireEvent.submit(input.closest('form')!);

    expect(await screen.findByText(
      'Planning is temporarily unavailable. Your accepted plan is unchanged.',
    )).toBeInTheDocument();
    expect(screen.queryByRole('complementary', { name: 'Latest itinerary' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Open latest itinerary' }));
    const reopened = screen.getByRole('complementary', { name: 'Latest itinerary' });
    expect(within(reopened).getByText('Accepted Tokyo Tower')).toBeInTheDocument();
  });

  it('preserves a closed accepted canvas after a real-parser unavailable response', async () => {
    const accepted = validPlanSnapshot('Parser Accepted Tower');
    const actualApi = await vi.importActual<typeof import('../../fyp_frontend/src/lib/api')>(
      '../../fyp_frontend/src/lib/api',
    );
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    apiMocks.getChatHistory.mockResolvedValue({
      status: 'success',
      sessions: [{
        id: 'parser-session', title: 'Tokyo', destination: 'Tokyo, Japan',
        updated_at: '2026-08-16T08:00:00Z',
        messages: [{ role: 'ai', content: 'Accepted itinerary.', ...accepted }],
      }],
    });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      status: 'planning_unavailable', reason: 'review_unavailable',
      chat_reply: 'Planning is temporarily unavailable. Your accepted plan is unchanged.',
      retryable: true, draft_itinerary: [], daily_map_info: {}, itinerary_modified: false,
      total_budget: 0, currency: '', budget_allocation: {}, budget_confirmation: null,
    }), { status: 200 })));
    apiMocks.sendChatMessage.mockImplementation(actualApi.sendChatMessage);

    render(<App />);
    const canvas = await screen.findByRole('complementary', { name: 'Latest itinerary' });
    fireEvent.click(within(canvas).getByRole('button', { name: 'Close itinerary' }));
    const input = screen.getByPlaceholderText('Ask about destinations, flights, or itineraries...');
    fireEvent.change(input, { target: { value: 'Replace the attraction' } });
    fireEvent.submit(input.closest('form')!);

    expect(await screen.findByText(
      'Planning is temporarily unavailable. Your accepted plan is unchanged.',
    )).toBeInTheDocument();
    expect(screen.queryByRole('complementary', { name: 'Latest itinerary' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Open latest itinerary' }));
    expect(within(screen.getByRole('complementary', { name: 'Latest itinerary' }))
      .getByText('Parser Accepted Tower')).toBeInTheDocument();
  });

  it.each(['success', 'budget_check_unavailable', 'error'])(
    'keeps a consumed %s confirmation disabled after leaving and reopening its session',
    async (outcome) => {
      apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
      apiMocks.getChatHistory.mockResolvedValue({ status: 'success', sessions: [{
        id: 'round-trip-session', title: 'Round trip', destination: 'Round trip', updated_at: '2026-08-16T08:00:00Z',
        messages: [{ role: 'ai', content: 'Confirm.', budget_confirmation: validChatConfirmation({ budget_assessment_id: 'same-id' }) }],
      }] });
      const reply = {
        status: outcome === 'success' ? 'success' : 'budget_check_unavailable',
        chat_reply: 'Resolved.', draft_itinerary: [], daily_map_info: {}, itinerary_modified: false,
        total_budget: 0, currency: 'CNY', budget_allocation: {},
      };
      if (outcome === 'error') apiMocks.sendChatMessage.mockRejectedValue(new Error('offline'));
      else apiMocks.sendChatMessage.mockResolvedValue(reply);
      render(<App />);
      const card = await screen.findByRole('region', { name: 'Chat budget confirmation' });
      fireEvent.click(within(card).getByRole('button', { name: 'Use recommended budget' }));
      await waitFor(() => expect(apiMocks.sendChatMessage).toHaveBeenCalledTimes(1));
      if (outcome === 'error') {
        expect(await screen.findByText(/Request failed: offline/)).toBeInTheDocument();
      } else {
        expect((await screen.findAllByText('Resolved.')).length).toBeGreaterThan(0);
      }
      await waitFor(() => expect(screen.queryByText('Planning your trip…')).not.toBeInTheDocument());
      fireEvent.click(screen.getByRole('button', { name: 'New Trip' }));
      fireEvent.click(screen.getByRole('button', { name: /Round trip/ }));
      const reopened = await screen.findByRole('region', { name: 'Chat budget confirmation' });
      const button = within(reopened).getByRole('button', { name: 'Use recommended budget' });
      expect(button).toBeDisabled();
      fireEvent.click(button);
      expect(apiMocks.sendChatMessage).toHaveBeenCalledTimes(1);
    },
  );

  it('makes only the newest repeated same-id live confirmation actionable', async () => {
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    apiMocks.getChatHistory.mockResolvedValue({ status: 'success', sessions: [{
      id: 'same-id-session', title: 'Same id', destination: 'Same id', updated_at: '2026-08-16T08:00:00Z',
      messages: [{ role: 'ai', content: 'Confirm.', budget_confirmation: validChatConfirmation({ budget_assessment_id: 'same-id-repeat' }) }],
    }] });
    apiMocks.sendChatMessage.mockResolvedValue({
      status: 'budget_confirmation_required', chat_reply: 'Confirm again.',
      budget_confirmation: validChatConfirmation({ budget_assessment_id: 'same-id-repeat' }),
      draft_itinerary: [], daily_map_info: {}, itinerary_modified: false, total_budget: 0, currency: 'CNY', budget_allocation: {},
    });
    render(<App />);
    const card = await screen.findByRole('region', { name: 'Chat budget confirmation' });
    fireEvent.click(within(card).getByRole('button', { name: 'Use recommended budget' }));
    expect(await screen.findByText('Confirm again.')).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByText('Planning your trip…')).not.toBeInTheDocument());
    const buttons = await screen.findAllByRole('button', { name: 'Use recommended budget' });
    expect(buttons).toHaveLength(2);
    expect(buttons[0]).toBeDisabled();
    expect(buttons[1]).toBeEnabled();
    fireEvent.click(buttons[0]);
    expect(apiMocks.sendChatMessage).toHaveBeenCalledTimes(1);
    fireEvent.click(buttons[1]);
    await waitFor(() => expect(apiMocks.sendChatMessage).toHaveBeenCalledTimes(2));
  });

  it('clears an older live confirmation after a sufficient alternative succeeds', async () => {
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    apiMocks.getChatHistory.mockResolvedValue({ status: 'success', sessions: [{
      id: 'alternative-session', title: 'Alternative', destination: 'Alternative', updated_at: '2026-08-16T08:00:00Z', messages: [],
    }] });
    apiMocks.sendChatMessage
      .mockResolvedValueOnce({
        status: 'budget_confirmation_required', chat_reply: 'Confirm recommendation.',
        budget_confirmation: validChatConfirmation({ budget_assessment_id: 'alternative-id' }),
        draft_itinerary: [], daily_map_info: {}, itinerary_modified: false,
        total_budget: 0, currency: 'CNY', budget_allocation: {},
      })
      .mockResolvedValueOnce({
        status: 'success', chat_reply: 'Alternative accepted.',
        draft_itinerary: [{ day: 1, hotel: { hotel_name: 'Alternative Hotel' }, activities: [] }],
        daily_map_info: {}, itinerary_modified: true,
        total_budget: 4000, currency: 'CNY', budget_allocation: {},
      });

    render(<App />);
    const input = await screen.findByPlaceholderText('Ask about destinations, flights, or itineraries...');
    fireEvent.change(input, { target: { value: 'Use RM 500' } });
    fireEvent.submit(input.closest('form')!);
    const card = await screen.findByRole('region', { name: 'Chat budget confirmation' });
    const staleButton = within(card).getByRole('button', { name: 'Use recommended budget' });
    expect(staleButton).toBeEnabled();

    fireEvent.change(input, { target: { value: 'Use RM 4,000 instead' } });
    fireEvent.submit(input.closest('form')!);
    expect(await screen.findByText('Alternative accepted.')).toBeInTheDocument();
    expect(staleButton).toBeDisabled();
    fireEvent.click(staleButton);
    expect(apiMocks.sendChatMessage).toHaveBeenCalledTimes(2);
  });

  it.each([
    ['a later success', [{ role: 'ai', content: 'Confirm.', budget_confirmation: validChatConfirmation({ budget_assessment_id: 'resolved-success' }) }, { role: 'ai', content: 'Plan accepted.' }]],
    ['a later user turn', [{ role: 'ai', content: 'Confirm.', budget_confirmation: validChatConfirmation({ budget_assessment_id: 'resolved-user' }) }, { role: 'user', content: 'Try something else.' }]],
    ['a later error turn', [{ role: 'ai', content: 'Confirm.', budget_confirmation: validChatConfirmation({ budget_assessment_id: 'resolved-error' }) }, { role: 'ai', content: 'Unable to continue.' }]],
  ])('keeps restored confirmation cards disabled after %s', async (_label, messages) => {
    apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
    apiMocks.getChatHistory.mockResolvedValue({ status: 'success', sessions: [{
      id: 'resolved-history', title: 'Resolved', destination: 'Resolved', updated_at: '2026-08-16T08:00:00Z', messages,
    }] });

    render(<App />);
    const card = await screen.findByRole('region', { name: 'Chat budget confirmation' });
    const button = within(card).getByRole('button', { name: 'Use recommended budget' });
    expect(button).toBeDisabled();
    fireEvent.click(button);
    expect(apiMocks.sendChatMessage).not.toHaveBeenCalled();
  });

  it.each(['success', 'rejection'])(
    'does not let a deferred form %s overwrite a newer session request',
    async (outcome) => {
      const savedPlan = validPlanSnapshot(
        'Saved activity', 'CN', { hotel_name: 'Saved Hotel' },
      );
      const deferredForm = createDeferred<any>();
      const deferredChat = createDeferred<any>();
      apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
      apiMocks.getChatHistory.mockResolvedValue({ status: 'success', sessions: [{
        id: 'saved-session', title: 'Saved trip', destination: 'Saved trip', updated_at: '2026-08-16T08:00:00Z',
        messages: [{ role: 'ai', content: 'Saved itinerary.', ...savedPlan }],
      }] });
      apiMocks.submitTripForm.mockReturnValue(deferredForm.promise);
      apiMocks.sendChatMessage
        .mockReturnValueOnce(deferredChat.promise)
        .mockResolvedValueOnce({ status: 'success', chat_reply: 'Follow-up reply.' });
      render(<App />);

      fireEvent.click(await screen.findByRole('button', { name: 'New Trip' }));
      fireEvent.change(screen.getByPlaceholderText('e.g. Japan'), { target: { value: 'Japan' } });
      fireEvent.change(screen.getByPlaceholderText('e.g. Kyoto, Osaka'), { target: { value: 'Tokyo' } });
      fireEvent.change(screen.getByLabelText('Start date'), { target: { value: '2026-08-01' } });
      fireEvent.change(screen.getByLabelText('End date'), { target: { value: '2026-08-05' } });
      fireEvent.change(screen.getByPlaceholderText('e.g. 3000'), { target: { value: '5000' } });
      fireEvent.click(screen.getByRole('button', { name: /Generate itinerary/ }));
      await waitFor(() => expect(apiMocks.submitTripForm).toHaveBeenCalledTimes(1));

      fireEvent.click(screen.getByRole('button', { name: /Saved trip/ }));
      const input = await screen.findByPlaceholderText('Ask about destinations, flights, or itineraries...');
      fireEvent.change(input, { target: { value: 'Newer chat request' } });
      fireEvent.submit(input.closest('form')!);
      await waitFor(() => expect(apiMocks.sendChatMessage).toHaveBeenCalledTimes(1));
      expect(screen.getByText('Planning your trip…')).toBeInTheDocument();

      await act(async () => {
        if (outcome === 'success') deferredForm.resolve({
          status: 'success', chat_reply: 'Stale form reply', itinerary: [{ day: 1, activities: [] }],
          daily_geojson_maps: {}, total_budget: 5000, currency: 'JPY', budget_allocation: {}, session_id: 'stale-session',
        });
        else deferredForm.reject(new Error('stale form failure'));
      });

      expect(screen.getByText('Saved itinerary.')).toBeInTheDocument();
      expect(screen.getByText('Newer chat request')).toBeInTheDocument();
      expect(screen.queryByText('Stale form reply')).not.toBeInTheDocument();
      expect(screen.queryByText('stale form failure')).not.toBeInTheDocument();
      expect(screen.getByText('Planning your trip…')).toBeInTheDocument();
      expect(within(screen.getByRole('complementary', { name: 'Latest itinerary' })).getByText('Saved Hotel')).toBeInTheDocument();

      await act(async () => deferredChat.resolve({ status: 'success', chat_reply: 'Newer reply.' }));
      expect(await screen.findByText('Newer reply.')).toBeInTheDocument();
      await waitFor(() => expect(screen.queryByText('Planning your trip…')).not.toBeInTheDocument());
      expect(within(screen.getByRole('complementary', { name: 'Latest itinerary' })).getByText('Saved Hotel')).toBeInTheDocument();
      fireEvent.change(input, { target: { value: 'Follow-up request' } });
      fireEvent.submit(input.closest('form')!);
      await waitFor(() => expect(apiMocks.sendChatMessage).toHaveBeenCalledTimes(2));
      expect(await screen.findByText('Follow-up reply.')).toBeInTheDocument();
    },
  );

  it('keeps a mounted card disabled after its expiry passes before it is clicked', async () => {
    vi.useFakeTimers();
    try {
      vi.setSystemTime(new Date('2026-08-16T12:00:00Z'));
      apiMocks.getProfile.mockResolvedValue({ status: 'success', onboarded: true, profile: {} });
      apiMocks.getChatHistory.mockResolvedValue({ status: 'success', sessions: [{
        id: 'expiry-session', title: 'Expiry', destination: 'Expiry', updated_at: '2026-08-16T08:00:00Z',
        messages: [{ role: 'ai', content: 'Confirm.', budget_confirmation: validChatConfirmation({ expires_at: '2026-08-16T12:00:01Z' }) }],
      }] });
      render(<App />);
      await act(async () => { await Promise.resolve(); await Promise.resolve(); });
      expect(screen.getByRole('button', { name: 'Use recommended budget' })).not.toBeDisabled();
      vi.setSystemTime(new Date('2026-08-16T12:00:01.100Z'));
      const button = screen.getByRole('button', { name: 'Use recommended budget' });
      expect(button).not.toBeDisabled();
      fireEvent.click(button);
      expect(apiMocks.sendChatMessage).not.toHaveBeenCalled();
      await act(async () => vi.advanceTimersByTimeAsync(1_100));
      expect(screen.getByRole('button', { name: 'Use recommended budget' })).toBeDisabled();
    } finally {
      vi.useRealTimers();
    }
  });

  it('opens emergency actions with the home-country embassy context', async () => {
    apiMocks.getProfile.mockResolvedValue({
      status: 'success',
      onboarded: true,
      profile: { home_country: 'Malaysia' },
    });
    render(<App />);
    fireEvent.click(await screen.findByRole('button', { name: /Emergency/ }));
    expect(screen.getByText('Emergency Assistance')).toBeInTheDocument();
    expect(screen.getByText('Embassy of Malaysia')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Get All Emergency Numbers' })).toBeInTheDocument();
  });
});
