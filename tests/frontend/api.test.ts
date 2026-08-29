import type { TripFormRequest } from '../../fyp_frontend/src/lib/api';

async function loadApi(baseUrl = 'http://api.test/', userId = 'user-123') {
  vi.resetModules();
  vi.stubEnv('VITE_API_BASE_URL', baseUrl);
  vi.stubEnv('VITE_USER_ID', userId);
  vi.stubEnv('VITE_MAPBOX_TOKEN', 'pk.test');
  return import('../../fyp_frontend/src/lib/api');
}


describe('frontend API client', () => {
  beforeEach(() => {
    vi.useRealTimers();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
    vi.useRealTimers();
  });

  it('normalizes configuration and exposes non-secret diagnostics', async () => {
    const api = await loadApi('http://api.test///', 'device-1');
    expect(api.apiConfig).toEqual({
      baseUrl: 'http://api.test//',
      userId: 'device-1',
      hasMapbox: true,
    });
  });

  it('ApiError preserves status, detail, name, and message', async () => {
    const { ApiError } = await loadApi();
    const error = new ApiError(409, { reason: 'conflict' }, 'Conflict');
    expect(error).toBeInstanceOf(Error);
    expect(error.name).toBe('ApiError');
    expect(error.status).toBe(409);
    expect(error.detail).toEqual({ reason: 'conflict' });
    expect(error.message).toBe('Conflict');
  });

  it('submits onboarding with JSON and the user header', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ status: 'success', user_id: 'user-123' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const api = await loadApi();
    const result = await api.submitProfile({
      name: 'Alex',
      origin_country: 'Malaysia',
      origin_state: 'Selangor',
    });
    expect(result.user_id).toBe('user-123');
    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://api.test/api/profile/');
    expect(init.method).toBe('POST');
    expect(init.headers).toEqual({
      'Content-Type': 'application/json',
      'X-User-ID': 'user-123',
    });
    expect(JSON.parse(String(init.body))).toEqual({
      name: 'Alex',
      origin_country: 'Malaysia',
      origin_state: 'Selangor',
    });
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });

  it('gets the onboarding profile with GET', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ status: 'success', onboarded: true, profile: {} }), {
        status: 200,
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const api = await loadApi();
    await api.getProfile();
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://api.test/api/profile/');
    expect(init.method).toBe('GET');
    expect(init.headers).toEqual({ 'X-User-ID': 'user-123' });
    expect(init.cache).toBe('no-store');
  });

  it('loads persisted chat history with the user header', async () => {
    const payload = {
      status: 'success',
      sessions: [{
        id: 'session-1', title: 'Tokyo', destination: 'Tokyo, Japan',
        updated_at: '2026-08-01T10:00:00Z', messages: [],
      }],
    };
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(payload), { status: 200 }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const api = await loadApi();
    await expect(api.getChatHistory(5)).resolves.toEqual(payload);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://api.test/api/chat/history?limit=5');
    expect(init.method).toBe('GET');
    expect(init.headers).toEqual({ 'X-User-ID': 'user-123' });
    expect(init.cache).toBe('no-store');
  });

  it.each([
    '{"draft_itinerary":[{"day":1}]}',
    'Before {"candidate_plan":{"country":"Malaysia"}} after',
  ])('rejects unsafe assistant history text before reload rendering: %s', async (content) => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      status: 'success',
      sessions: [{
        id: 'unsafe-history',
        title: 'Unsafe',
        destination: 'Singapore',
        updated_at: '2026-08-16T12:01:00Z',
        messages: [
          { role: 'user', content: 'Keep my request' },
          { role: 'ai', content },
        ],
      }],
    }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const api = await loadApi();

    await expect(api.getChatHistory()).rejects.toThrow(
      'Saved trips returned an invalid response.',
    );
  });

  it('posts the exact chat contract', async () => {
    const response = {
      status: 'budget_check_unavailable',
      chat_reply: 'Hello',
      draft_itinerary: [],
      daily_map_info: {},
      itinerary_modified: false,
      total_budget: 0,
      currency: '',
      budget_allocation: {},
      budget_confirmation: null,
    };
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(response), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const api = await loadApi();
    await expect(api.sendChatMessage('session-1', 'hello')).resolves.toEqual(response);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toEqual({
      session_id: 'session-1',
      user_message: 'hello',
    });
  });

  it('posts an explicit recommended-budget acceptance with its opaque assessment id', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      status: 'budget_check_unavailable',
      chat_reply: 'Budget confirmed.',
      draft_itinerary: [],
      daily_map_info: {},
      itinerary_modified: false,
      total_budget: 0,
      currency: '',
      budget_allocation: {},
      budget_confirmation: null,
    }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const api = await loadApi();

    await api.sendChatMessage('session-1', 'Use recommended budget', {
      budget_action: 'accept_recommended',
      budget_assessment_id: 'assessment-1',
    });

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toEqual({
      session_id: 'session-1',
      user_message: 'Use recommended budget',
      budget_action: 'accept_recommended',
      budget_assessment_id: 'assessment-1',
    });
  });

  it.each([
    ['missing evidence', { evidence: undefined }],
    ['null evidence', { evidence: null }],
    ['blank assessment id', { budget_assessment_id: '  ' }],
    ['non-finite amount', { recommended_minimum_budget: Infinity }],
    ['invalid currency', { base_currency: 'NOT-A-CURRENCY' }],
    ['invalid expiry', { expires_at: 'not-a-date' }],
    ['invalid reason', { reason: 'other' }],
    ['invalid status', { status: 'success' }],
  ])('rejects malformed chat confirmation data: %s', async (_label, override) => {
    const api = await loadApi();
    const valid = {
      status: 'budget_confirmation_required', reason: 'insufficient_budget', chat_reply: 'Confirm the grounded minimum.',
      budget_assessment_id: 'assessment-1', stated_budget: 500, recommended_minimum_budget: 3500,
      base_currency: 'MYR', destination_currency: 'CNY', expires_at: '2099-08-16T13:00:00Z',
      evidence: { outbound_flight_price: 1200, return_flight_price: 1100, hotel_price_per_night: 180, hotel_nights: 3 },
      itinerary: null, daily_geojson_maps: null,
    };
    expect(api.isChatBudgetConfirmation({ ...valid, ...override })).toBe(false);
  });

  it.each([
    ['zero stated budget', { stated_budget: 0 }], ['negative stated budget', { stated_budget: -1 }],
    ['zero recommended budget', { recommended_minimum_budget: 0 }], ['negative recommended budget', { recommended_minimum_budget: -1 }],
    ['zero outbound price', { evidence: { outbound_flight_price: 0, return_flight_price: 1100, hotel_price_per_night: 180, hotel_nights: 3 } }],
    ['negative return price', { evidence: { outbound_flight_price: 1200, return_flight_price: -1, hotel_price_per_night: 180, hotel_nights: 3 } }],
    ['zero hotel price', { evidence: { outbound_flight_price: 1200, return_flight_price: 1100, hotel_price_per_night: 0, hotel_nights: 3 } }],
    ['zero hotel nights', { evidence: { outbound_flight_price: 1200, return_flight_price: 1100, hotel_price_per_night: 180, hotel_nights: 0 } }],
    ['negative hotel nights', { evidence: { outbound_flight_price: 1200, return_flight_price: 1100, hotel_price_per_night: 180, hotel_nights: -1 } }],
  ])('rejects nonpositive backend confirmation values: %s', async (_label, override) => {
    const api = await loadApi();
    const valid = {
      status: 'budget_confirmation_required', reason: 'insufficient_budget', chat_reply: 'Confirm the grounded minimum.',
      budget_assessment_id: 'assessment-1', stated_budget: 500, recommended_minimum_budget: 3500,
      base_currency: 'MYR', destination_currency: 'CNY', expires_at: '2099-08-16T13:00:00Z',
      evidence: { outbound_flight_price: 1200, return_flight_price: 1100, hotel_price_per_night: 180, hotel_nights: 3 },
      itinerary: null, daily_geojson_maps: null,
    };
    expect(api.isChatBudgetConfirmation({ ...valid, ...override })).toBe(false);
  });

  it('accepts canonical zero-night day-trip budget evidence', async () => {
    const api = await loadApi();
    expect(api.isChatBudgetConfirmation({
      status: 'budget_confirmation_required', reason: 'insufficient_budget', chat_reply: 'Confirm the grounded minimum.',
      budget_assessment_id: 'assessment-day-trip', stated_budget: 199,
      recommended_minimum_budget: 200, base_currency: 'MYR', destination_currency: 'JPY',
      expires_at: '2099-08-16T13:00:00Z', itinerary: null, daily_geojson_maps: null,
      evidence: { outbound_flight_price: 25, return_flight_price: 25,
        hotel_price_per_night: 0, hotel_nights: 0 },
    })).toBe(true);
  });

  it.each([
    ['embedded primitive array', 'Activities: ["Merlion Park"]', false],
    ['embedded empty array', 'Activities: []', false],
    ['embedded nested array', 'Activities: [["Merlion Park"]]', false],
    ['whole primitive array', '["Merlion Park"]', false],
    ['ordinary punctuation', 'Meet at [Gate A] {near arrivals}.', true],
    ['single-integer citation', 'See the official guide [1].', true],
  ])('applies backend JSON-container semantics: %s', async (_label, content, expected) => {
    const api = await loadApi();
    expect(api.isSafeAssistantContent(content)).toBe(expected);
  });

  it.each([
    ['missing maps', { daily_map_info: undefined }],
    ['null maps', { daily_map_info: null }],
    ['empty itinerary', { draft_itinerary: [] }],
    ['invalid itinerary', { draft_itinerary: [{}] }],
    ['non-finite total', { total_budget: Infinity }],
    ['missing allocation', { budget_allocation: undefined }],
  ])('rejects incomplete live success snapshots: %s', async (_label, override) => {
    const api = await loadApi();
    const valid = {
      status: 'success', itinerary_modified: true, chat_reply: 'Updated.',
      draft_itinerary: [{ day: 1, activities: [] }], daily_map_info: {},
      total_budget: 3500, currency: 'CNY', budget_allocation: {},
    };
    expect(api.hasCompleteLiveSuccessSnapshot({ ...valid, ...override })).toBe(false);
  });

  const completeDay = {
    day: 1,
    date: '2026-08-01',
    flight: null,
    hotel: null,
    activities: [{
      name: 'Gardens by the Bay',
      type: 'attraction',
      address: '18 Marina Gardens Drive, Singapore',
      estimated_cost: 10,
      order: 1,
      location: {
        place_name: 'Gardens by the Bay',
        latitude: 1.2816,
        longitude: 103.8636,
        country_code: 'SG',
        requested_city: 'Singapore',
        verified_locality: 'Singapore',
      },
    }],
    route: null,
    day_total_cost: 10,
  };
  const completeMap = {
    type: 'FeatureCollection',
    features: [{
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [103.8636, 1.2816] },
      properties: { name: 'Gardens by the Bay', type: 'attraction', order: 1 },
    }],
  };
  const completeLiveSuccess = {
    status: 'success',
    chat_reply: 'Updated.',
    draft_itinerary: [completeDay],
    daily_map_info: { '1': completeMap },
    destination_country_code: 'SG',
    itinerary_modified: true,
    total_budget: 3500,
    currency: 'SGD',
    budget_allocation: { transportation: 3500 },
    budget_confirmation: null,
  };

  it.each([
    ['embedded JSON', 'Your itinerary is ready: {"days":[{"day":1}]}'],
    ['private planning marker', 'The candidate_plan is now ready.'],
  ])('rejects unsafe assistant copy at every success boundary: %s', async (
    _label,
    chatReply,
  ) => {
    const api = await loadApi();
    const liveResponse = { ...completeLiveSuccess, chat_reply: chatReply };
    const initialResponse = {
      status: 'success',
      chat_reply: chatReply,
      itinerary: [completeDay],
      daily_geojson_maps: { '1': completeMap },
      destination_country_code: 'SG',
      total_budget: 3500,
      currency: 'SGD',
      budget_allocation: { transportation: 3500 },
      session_id: 'session-1',
    };

    expect(api.hasCompleteLiveSuccessSnapshot(liveResponse)).toBe(false);
    expect(api.isChatResponse(liveResponse)).toBe(false);
    expect(api.hasCompleteInitialSuccessSnapshot(initialResponse)).toBe(false);
    expect(api.isTripSubmissionResponse(initialResponse)).toBe(false);
  });

  it.each([
    ['embedded JSON', 'Retry details: {"candidate_plan":{"day":1}}'],
    ['private planning marker', 'The planning_outcome is unavailable.'],
  ])('rejects unsafe assistant copy from non-success response variants: %s', async (
    _label,
    chatReply,
  ) => {
    const api = await loadApi();
    const initialUnavailable = {
      status: 'planning_unavailable', reason: 'review_unavailable',
      chat_reply: chatReply, retryable: true,
      itinerary: null, daily_geojson_maps: null,
    };
    const chatUnavailable = {
      status: 'planning_unavailable', reason: 'review_unavailable',
      chat_reply: chatReply, retryable: true,
      draft_itinerary: [], daily_map_info: {}, itinerary_modified: false,
      total_budget: 0, currency: '', budget_allocation: {}, budget_confirmation: null,
    };
    const confirmation = {
      status: 'budget_confirmation_required', reason: 'insufficient_budget',
      chat_reply: chatReply, budget_assessment_id: 'assessment-1',
      stated_budget: 500, recommended_minimum_budget: 3500,
      base_currency: 'MYR', destination_currency: 'SGD',
      expires_at: '2099-08-16T13:00:00Z',
      evidence: {
        outbound_flight_price: 1200, return_flight_price: 1100,
        hotel_price_per_night: 180, hotel_nights: 3,
      },
      itinerary: null, daily_geojson_maps: null,
    };

    expect(api.isTripSubmissionResponse(initialUnavailable)).toBe(false);
    expect(api.isChatResponse(chatUnavailable)).toBe(false);
    expect(api.isChatBudgetConfirmation(confirmation)).toBe(false);
  });

  it.each([
    ['empty activity day', [{ ...completeDay, activities: [] }]],
    ['non-sequential day', [{ ...completeDay, day: 2 }]],
    ['absent activity address', [{
      ...completeDay,
      activities: [{ ...completeDay.activities[0], address: undefined }],
    }]],
    ['non-finite coordinate', [{
      ...completeDay,
      activities: [{
        ...completeDay.activities[0],
        location: { ...completeDay.activities[0].location, latitude: Number.NaN },
      }],
    }]],
    ['private nested activity field', [{
      ...completeDay,
      activities: [{ ...completeDay.activities[0], provider_payload: { secret: true } }],
    }]],
    ['provider locality does not match requested city', [{
      ...completeDay,
      activities: [{
        ...completeDay.activities[0],
        location: {
          ...completeDay.activities[0].location,
          verified_locality: 'Johor Bahru',
        },
      }],
    }]],
  ])('rejects incomplete public itineraries: %s', async (_label, itinerary) => {
    const api = await loadApi();
    expect(api.isCompleteItinerary(itinerary)).toBe(false);
  });

  it('accepts a complete closed public itinerary', async () => {
    const api = await loadApi();
    expect(api.isCompleteItinerary([completeDay])).toBe(true);
  });

  it('accepts the unmutated destination/map live-success baseline', async () => {
    const api = await loadApi();
    expect(api.hasCompleteLiveSuccessSnapshot(completeLiveSuccess)).toBe(true);
  });

  it.each([
    ['wrong country', {
      draft_itinerary: [{
        ...completeDay,
        activities: [{
          ...completeDay.activities[0],
          location: { ...completeDay.activities[0].location, country_code: 'MY' },
        }],
      }],
    }],
    ['missing map day', { daily_map_info: {} }],
    ['coercible map day key', { daily_map_info: { '01': completeMap } }],
    ['mismatched map point name', {
      daily_map_info: {
        '1': {
          ...completeMap,
          features: [{
            ...completeMap.features[0],
            properties: { ...completeMap.features[0].properties, name: 'Different place' },
          }],
        },
      },
    }],
    ['mismatched map point coordinate', {
      daily_map_info: {
        '1': {
          ...completeMap,
          features: [{
            ...completeMap.features[0],
            geometry: { type: 'Point', coordinates: [103.8, 1.2] },
          }],
        },
      },
    }],
    ['unknown map property', {
      daily_map_info: {
        '1': {
          ...completeMap,
          features: [{
            ...completeMap.features[0],
            properties: { ...completeMap.features[0].properties, provider_payload: true },
          }],
        },
      },
    }],
    ['mismatched map point type', {
      daily_map_info: {
        '1': {
          ...completeMap,
          features: [{
            ...completeMap.features[0],
            properties: { ...completeMap.features[0].properties, type: 'hotel' },
          }],
        },
      },
    }],
    ['mismatched map point order', {
      daily_map_info: {
        '1': {
          ...completeMap,
          features: [{
            ...completeMap.features[0],
            properties: { ...completeMap.features[0].properties, order: 99 },
          }],
        },
      },
    }],
    ['point with route-only property', {
      daily_map_info: {
        '1': {
          ...completeMap,
          features: [{
            ...completeMap.features[0],
            properties: { ...completeMap.features[0].properties, profile: 'walking' },
          }],
        },
      },
    }],
    ['unsupported polygon feature', {
      daily_map_info: {
        '1': {
          ...completeMap,
          features: [
            ...completeMap.features,
            {
              type: 'Feature',
              geometry: { type: 'Polygon', coordinates: [] },
              properties: { name: 'Injected', type: 'attraction', order: 2 },
            },
          ],
        },
      },
    }],
    ['route line without matching route metadata', {
      daily_map_info: {
        '1': {
          ...completeMap,
          features: [
            ...completeMap.features,
            {
              type: 'Feature',
              geometry: {
                type: 'LineString',
                coordinates: [[103.8636, 1.2816], [103.87, 1.29]],
              },
              properties: {
                type: 'route',
                profile: 'driving',
                distance_km: 2,
                duration_mins: 5,
              },
            },
          ],
        },
      },
    }],
  ])('rejects destination-inconsistent live success: %s', async (_label, override) => {
    const api = await loadApi();
    expect(api.hasCompleteLiveSuccessSnapshot({ ...completeLiveSuccess, ...override })).toBe(false);
  });

  it.each([
    ['null', null],
    ['empty', ''],
    ['wrong type', 42],
  ])('rejects an explicitly present %s snapshot country', async (_label, country) => {
    const api = await loadApi();
    expect(api.isCompleteItinerarySnapshot({
      itinerary: [completeDay], maps: { '1': completeMap },
      budget: { total: 10, currency: 'SGD', allocation: { activity: 10 } },
      destination_country_code: country,
    })).toBe(false);
  });

  it('infers the country only when a legacy snapshot omits the property', async () => {
    const api = await loadApi();
    expect(api.isCompleteItinerarySnapshot({
      itinerary: [completeDay], maps: { '1': completeMap },
      budget: { total: 10, currency: 'SGD', allocation: { activity: 10 } },
    })).toBe(true);
  });

  const nextDay = (date: string) => ({
    ...completeDay,
    day: 2,
    date,
    activities: [{
      ...completeDay.activities[0],
      name: 'Merlion Park',
      address: '1 Fullerton Road, Singapore',
      location: {
        ...completeDay.activities[0].location,
        place_name: 'Merlion Park',
        latitude: 1.2868,
        longitude: 103.8545,
      },
    }],
  });

  it('rejects non-consecutive itinerary dates', async () => {
    const api = await loadApi();
    expect(api.isCompleteItinerary([
      completeDay,
      nextDay('2026-08-03'),
    ])).toBe(false);
  });

  it.each([
    ['month rollover', { ...completeDay, date: '2026-08-31' }, nextDay('2026-09-01')],
    ['year rollover', { ...completeDay, date: '2026-12-31' }, nextDay('2027-01-01')],
  ])('accepts consecutive itinerary dates across a %s', async (_label, first, second) => {
    const api = await loadApi();
    expect(api.isCompleteItinerary([first, second])).toBe(true);
  });

  it.each([
    ['under allocation', { total: 100, currency: 'SGD', allocation: { transportation: 99.94 } }, false],
    ['over allocation', { total: 100, currency: 'SGD', allocation: { transportation: 100.06 } }, false],
    ['negative allocation', { total: 100, currency: 'SGD', allocation: { transportation: -1 } }, false],
    ['non-finite allocation', { total: 100, currency: 'SGD', allocation: { transportation: Infinity } }, false],
    ['unknown allocation key', { total: 100, currency: 'SGD', allocation: { private: 100 } }, false],
    ['rounded within tolerance', { total: 100, currency: 'SGD', allocation: { transportation: 99.96 } }, true],
  ])('validates budget allocation totals: %s', async (_label, value, expected) => {
    const api = await loadApi();
    expect(api.isBudgetInfo(value)).toBe(expected);
  });

  it.each([
    ['missing route distance', { duration_mins: 5 }],
    ['missing route duration', { distance_km: 2 }],
    ['mismatched route distance', { distance_km: 3, duration_mins: 5 }],
  ])('rejects incomplete or mismatched route line metrics: %s', async (_label, properties) => {
    const api = await loadApi();
    const routedDay = {
      ...completeDay,
      route: {
        ordered_stops: ['Gardens by the Bay'],
        profiles: { driving: { distance_km: 2, duration_mins: 5 } },
      },
    };
    const routeFeature = {
      type: 'Feature',
      geometry: {
        type: 'LineString',
        coordinates: [[103.8636, 1.2816], [103.87, 1.29]],
      },
      properties: { type: 'route', profile: 'driving', ...properties },
    };
    expect(api.hasCompleteLiveSuccessSnapshot({
      status: 'success', chat_reply: 'Updated.', itinerary_modified: true,
      draft_itinerary: [routedDay],
      daily_map_info: { '1': { ...completeMap, features: [...completeMap.features, routeFeature] } },
      destination_country_code: 'SG', total_budget: 3500, currency: 'SGD',
      budget_allocation: { transportation: 3500 }, budget_confirmation: null,
    })).toBe(false);
  });

  it('rejects duplicate route profiles in a day map', async () => {
    const api = await loadApi();
    const routedDay = {
      ...completeDay,
      route: {
        ordered_stops: ['Gardens by the Bay'],
        profiles: { driving: { distance_km: 2, duration_mins: 5 } },
      },
    };
    const routeFeature = {
      type: 'Feature',
      geometry: {
        type: 'LineString',
        coordinates: [[103.8636, 1.2816], [103.87, 1.29]],
      },
      properties: {
        type: 'route', profile: 'driving', distance_km: 2, duration_mins: 5,
      },
    };
    expect(api.hasCompleteLiveSuccessSnapshot({
      status: 'success', chat_reply: 'Updated.', itinerary_modified: true,
      draft_itinerary: [routedDay],
      daily_map_info: {
        '1': { ...completeMap, features: [...completeMap.features, routeFeature, routeFeature] },
      },
      destination_country_code: 'SG', total_budget: 3500, currency: 'SGD',
      budget_allocation: { transportation: 3500 }, budget_confirmation: null,
    })).toBe(false);
  });

  it.each([
    [
      'geometry endpoints',
      ['Gardens by the Bay'],
      [[103.8636, 1.2816], [103.87, 1.29]],
    ],
    [
      'ordered stops',
      ['Wrong stop'],
      [[103.8636, 1.2816], [103.8636, 1.2816]],
    ],
  ])('rejects route %s that disagree with ordered map points', async (
    _label,
    orderedStops,
    coordinates,
  ) => {
    const api = await loadApi();
    const routedDay = {
      ...completeDay,
      route: {
        ordered_stops: orderedStops,
        profiles: { driving: { distance_km: 2, duration_mins: 5 } },
      },
    };
    const routeFeature = {
      type: 'Feature',
      geometry: { type: 'LineString', coordinates },
      properties: {
        type: 'route', profile: 'driving', distance_km: 2, duration_mins: 5,
      },
    };
    expect(api.hasCompleteLiveSuccessSnapshot({
      ...completeLiveSuccess,
      draft_itinerary: [routedDay],
      daily_map_info: {
        '1': { ...completeMap, features: [...completeMap.features, routeFeature] },
      },
    })).toBe(false);
  });

  it.each([
    ['duplicate map day', 'map-day'],
    ['duplicate nested activity field', 'nested-field'],
  ])('rejects raw JSON with an exact %s key before ordinary parsing', async (_label, duplicate) => {
    const mapJson = JSON.stringify(completeMap);
    const dayJson = JSON.stringify(completeDay);
    const raw = duplicate === 'map-day'
      ? `{"status":"success","chat_reply":"Ready","itinerary":[${dayJson}],` +
        `"daily_geojson_maps":{"1":${mapJson},"1":${mapJson}},` +
        `"destination_country_code":"SG","total_budget":10,"currency":"SGD",` +
        `"budget_allocation":{"activity":10},"session_id":"session-1"}`
      : `{"status":"success","chat_reply":"Ready","itinerary":[${dayJson.replace(
        '"name":"Gardens by the Bay"',
        '"name":"Gardens by the Bay","name":"Private duplicate"',
      )}],"daily_geojson_maps":{"1":${mapJson}},"destination_country_code":"SG",` +
        `"total_budget":10,"currency":"SGD","budget_allocation":{"activity":10},` +
        `"session_id":"session-1"}`;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(raw, { status: 200 })));
    const api = await loadApi();
    await expect(api.submitTripForm({
      country: 'Singapore', city: [], num_people: 1, total_budget: 10,
      start_date: '2026-08-01', end_date: '2026-08-02',
    })).rejects.toThrow('duplicate');
  });

  it('rejects recursively malformed itinerary snapshots in chat history', async () => {
    const payload = {
      status: 'success',
      sessions: [{
        id: 'history-1', title: 'Singapore', destination: 'Singapore',
        updated_at: '2026-08-01T10:00:00Z',
        messages: [{
          role: 'ai', content: 'Saved plan.',
          itinerary: [{ ...completeDay, private_candidate: true }],
          maps: { '1': completeMap },
          budget: { total: 10, currency: 'SGD', allocation: { activity: 10 } },
          destination_country_code: 'SG',
        }],
      }],
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify(payload), { status: 200 }),
    ));
    const api = await loadApi();
    await expect(api.getChatHistory()).rejects.toThrow('invalid response');
  });

  it.each([
    ['null', null],
    ['empty', ''],
    ['wrong type', 42],
  ])('rejects history with an explicitly present %s snapshot country', async (_label, country) => {
    const payload = {
      status: 'success',
      sessions: [{
        id: 'history-country', title: 'Singapore', destination: 'Singapore',
        updated_at: '2026-08-01T10:00:00Z',
        messages: [{
          role: 'ai', content: 'Saved plan.', itinerary: [completeDay],
          maps: { '1': completeMap },
          budget: { total: 10, currency: 'SGD', allocation: { activity: 10 } },
          destination_country_code: country,
        }],
      }],
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify(payload), { status: 200 }),
    ));
    const api = await loadApi();
    await expect(api.getChatHistory()).rejects.toThrow('invalid response');
  });

  it('accepts a legacy history snapshot only when the country property is absent', async () => {
    const payload = {
      status: 'success',
      sessions: [{
        id: 'legacy-history', title: 'Singapore', destination: 'Singapore',
        updated_at: '2026-08-01T10:00:00Z',
        messages: [{
          role: 'ai', content: 'Saved plan.', itinerary: [completeDay],
          maps: { '1': completeMap },
          budget: { total: 10, currency: 'SGD', allocation: { activity: 10 } },
        }],
      }],
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify(payload), { status: 200 }),
    ));
    const api = await loadApi();
    await expect(api.getChatHistory()).resolves.toEqual(payload);
  });

  it('rejects initial success without a non-empty server-reviewed reply', async () => {
    const response = {
      status: 'success',
      chat_reply: '   ',
      itinerary: [completeDay],
      daily_geojson_maps: { '1': completeMap },
      destination_country_code: 'SG',
      total_budget: 10,
      currency: 'SGD',
      budget_allocation: { activity: 10 },
      session_id: 'session-1',
    };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify(response), { status: 200 }),
    ));
    const api = await loadApi();

    await expect(api.submitTripForm({
      country: 'Singapore',
      city: ['Singapore'],
      num_people: 1,
      total_budget: 10,
      start_date: '2026-08-01',
      end_date: '2026-08-01',
    })).rejects.toThrow('The planning service returned an invalid response.');
  });

  it('posts the trip form without obsolete origin fields', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        status: 'planning_unavailable', reason: 'review_unavailable',
        chat_reply: 'Please retry.', retryable: true,
        itinerary: null, daily_geojson_maps: null,
      }), { status: 200 }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const api = await loadApi();
    const form = {
      country: 'Japan',
      city: ['Tokyo'],
      num_people: 2,
      total_budget: 5000,
      start_date: '2026-08-01',
      end_date: '2026-08-05',
    };
    await api.submitTripForm(form);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toEqual(form);
    expect(String(init.body)).not.toContain('origin_country');
  });

  it('posts an explicit recommendation request without inventing a budget', async () => {
    const response = {
      status: 'budget_confirmation_required',
      reason: 'recommendation_requested',
      chat_reply: 'Confirm the grounded minimum.',
      budget_assessment_id: 'assessment-1',
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
    };
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(response), { status: 200 }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const api = await loadApi();
    const form: TripFormRequest = {
      country: 'Japan',
      city: ['Tokyo'],
      num_people: 1,
      request_budget_recommendation: true,
      start_date: '2026-08-01',
      end_date: '2026-08-05',
    };

    await expect(api.submitTripForm(form)).resolves.toEqual(response);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toEqual(form);
    expect(JSON.parse(String(init.body))).not.toHaveProperty('total_budget');
  });

  it('posts the exact cached assessment id with a confirmed minimum', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        status: 'planning_unavailable', reason: 'review_unavailable',
        chat_reply: 'Please retry.', retryable: true,
        itinerary: null, daily_geojson_maps: null,
      }), { status: 200 }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const api = await loadApi();
    const form: TripFormRequest = {
      country: 'Japan',
      city: ['Tokyo'],
      num_people: 1,
      total_budget: 3200,
      budget_assessment_id: 'assessment-1',
      start_date: '2026-08-01',
      end_date: '2026-08-05',
    };

    await api.submitTripForm(form);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toEqual(form);
  });

  it.each([
    ['Kuala Lumpur', undefined, undefined, '/api/geocode/?q=Kuala%20Lumpur'],
    ['KLCC & Park', 3.1, 101.7, '/api/geocode/?q=KLCC%20%26%20Park&lat=3.1&lng=101.7'],
    ['Tokyo', 3.1, undefined, '/api/geocode/?q=Tokyo'],
  ])('builds geocode query safely for %s', async (query, lat, lng, expectedPath) => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ status: 'success', found: false }), { status: 200 }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const api = await loadApi();
    await api.geocodePlace(query, lat, lng);
    expect(fetchMock.mock.calls[0][0]).toBe(`http://api.test${expectedPath}`);
  });

  it('turns string backend details into ApiError messages', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: 'Onboarding incomplete' }), { status: 409 }),
      ),
    );
    const api = await loadApi();
    await expect(api.getProfile()).rejects.toMatchObject({
      name: 'ApiError',
      status: 409,
      detail: 'Onboarding incomplete',
      message: 'Onboarding incomplete',
    });
  });

  it('retains structured validation details and creates a safe summary', async () => {
    const detail = [{ loc: ['body', 'country'], msg: 'Field required' }];
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail }), { status: 422, statusText: 'Unprocessable' }),
      ),
    );
    const api = await loadApi();
    await expect(api.getProfile()).rejects.toMatchObject({
      status: 422,
      detail,
      message: 'Request to /api/profile/ failed with 422',
    });
  });

  it('handles non-JSON error bodies', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response('Gateway offline', { status: 502 })),
    );
    const api = await loadApi();
    await expect(api.getProfile()).rejects.toMatchObject({
      status: 502,
      detail: 'Gateway offline',
      message: 'Gateway offline',
    });
  });

  it('wraps network failures without leaking low-level messages to the UI', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('socket secret')));
    const api = await loadApi('http://offline.test');
    await expect(api.getProfile()).rejects.toMatchObject({
      status: 0,
      message: expect.stringContaining('Cannot reach backend at http://offline.test'),
    });
  });

  it('aborts a chat request at its configured timeout', async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn((_url: string, init: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        init.signal?.addEventListener('abort', () =>
          reject(new DOMException('aborted', 'AbortError')),
        );
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    const api = await loadApi();
    const request = api.sendChatMessage('s', 'hello');
    const rejection = expect(request).rejects.toMatchObject({
      status: 0,
      message: expect.stringContaining('timed out after 250s'),
    });
    await vi.advanceTimersByTimeAsync(250_001);
    await rejection;
  });

  it.each([
    [true, 200],
    [false, 503],
  ])('health returns %s for HTTP %i', async (expected, status) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status })));
    const api = await loadApi();
    await expect(api.checkHealth()).resolves.toBe(expected);
  });

  it('health returns false for network errors', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')));
    const api = await loadApi();
    await expect(api.checkHealth()).resolves.toBe(false);
  });

  it('generates unique non-empty session IDs', async () => {
    const api = await loadApi();
    const first = api.newSessionId();
    const second = api.newSessionId();
    expect(first).toBeTruthy();
    expect(second).toBeTruthy();
    expect(first).not.toBe(second);
  });

  it('uses the deterministic fallback shape when randomUUID is unavailable', async () => {
    vi.stubGlobal('crypto', {});
    vi.spyOn(Date, 'now').mockReturnValue(123456);
    vi.spyOn(Math, 'random').mockReturnValue(0.5);
    const api = await loadApi();
    expect(api.newSessionId()).toBe('sess-123456-8');
  });
});
