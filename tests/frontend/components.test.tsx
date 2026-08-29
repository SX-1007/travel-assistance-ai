import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';

const componentMocks = vi.hoisted(() => ({
  geocodePlace: vi.fn(),
  buildMarkersMapUrl: vi.fn(() => 'https://maps.test/markers.png'),
  buildStaticMapUrl: vi.fn(
    (_fc?: unknown, _options?: unknown): string | null => 'https://maps.test/route.png',
  ),
}));

const { geocodePlace, buildMarkersMapUrl, buildStaticMapUrl } = componentMocks;

vi.mock('../../fyp_frontend/src/lib/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../fyp_frontend/src/lib/api')>();
  return {
    ...actual,
    apiConfig: { baseUrl: 'http://api.test', userId: 'user', hasMapbox: true },
    geocodePlace: componentMocks.geocodePlace,
  };
});

vi.mock('../../fyp_frontend/src/lib/mapStatic', () => ({
  buildMarkersMapUrl: componentMocks.buildMarkersMapUrl,
  buildStaticMapUrl: componentMocks.buildStaticMapUrl,
}));

import ItineraryView from '../../fyp_frontend/src/app/components/ItineraryView';
import ItineraryCanvas from '../../fyp_frontend/src/app/components/ItineraryCanvas';
import MessageContent from '../../fyp_frontend/src/app/components/MessageContent';


describe('MessageContent structured renderer', () => {
  beforeEach(() => {
    geocodePlace.mockReset();
    geocodePlace.mockResolvedValue({ status: 'success', found: false });
    buildMarkersMapUrl.mockClear();
  });

  it('renders an empty message without crashing', () => {
    const { container } = render(<MessageContent content="" />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders inline bold, italic, code and strike formatting', () => {
    render(<MessageContent content="Use **bold**, *italics*, `code`, and ~~old~~." />);
    expect(screen.getByText('bold').tagName).toBe('STRONG');
    expect(screen.getByText('italics').tagName).toBe('EM');
    expect(screen.getByText('code').tagName).toBe('CODE');
    expect(screen.getByText('old').tagName).toBe('S');
  });

  it('turns bracketed inline code into an action chip', () => {
    render(<MessageContent content={'`[ Book Now ]`'} />);
    expect(screen.getByRole('button', { name: /Book Now/i })).toBeInTheDocument();
  });

  it('renders markdown headings and timeline bullets', () => {
    render(<MessageContent content={'## Plan\n\n* Morning: Museum\n- Evening: Dinner'} />);
    expect(screen.getByText('Plan')).toHaveClass('text-lg');
    expect(screen.getByText(/Museum/)).toBeInTheDocument();
    expect(screen.getByText(/Dinner/)).toBeInTheDocument();
  });

  it('renders markdown tables and removes the separator row', () => {
    render(
      <MessageContent
        content={'| Item | Price |\n| :--- | ---: |\n| Ticket | 20 |'}
      />,
    );
    const table = screen.getByRole('table');
    expect(within(table).getByText('Item')).toBeInTheDocument();
    expect(within(table).getByText('Ticket')).toBeInTheDocument();
    expect(within(table).queryByText(':---')).not.toBeInTheDocument();
  });

  it('renders ordinary blockquotes separately from live API cards', () => {
    const { rerender } = render(<MessageContent content={'> A normal note'} />);
    expect(screen.getByText('A normal note')).toBeInTheDocument();

    rerender(
      <MessageContent
        content={'> 🏨 **LIVE API RESULT: ACCOMMODATION**\n> **Sakura Inn**\n> Address: Tokyo'}
      />,
    );
    expect(screen.getByText('Live API Result')).toBeInTheDocument();
    expect(screen.getByText('ACCOMMODATION')).toBeInTheDocument();
    expect(screen.getByText('Sakura Inn')).toBeInTheDocument();
  });

  it('parses Sequence and Step wrappers into ordered day cards', () => {
    const content = `
      {/* renderer note */}
      Intro to your trip.
      <Sequence>
        <Step subtitle="Arrival" title="Day 1: Tokyo">
          * Morning: Check in
        </Step>
        <Step title="Day 2: Kyoto" subtitle="Temples">
          * Afternoon: Shrine
        </Step>
      </Sequence>`;
    render(<MessageContent content={content} />);
    expect(screen.getByText('Intro to your trip.')).toBeInTheDocument();
    expect(screen.getByText('Day 1: Tokyo')).toBeInTheDocument();
    expect(screen.getByText('Arrival')).toBeInTheDocument();
    expect(screen.getByText('Day 2: Kyoto')).toBeInTheDocument();
    expect(screen.getByText('Temples')).toBeInTheDocument();
    expect(screen.queryByText(/renderer note/)).not.toBeInTheDocument();
  });

  it('merges consecutive map lines, re-geocodes labels and renders one map', async () => {
    geocodePlace
      .mockResolvedValueOnce({ status: 'success', found: true, lat: 3.2, lng: 101.8 })
      .mockResolvedValueOnce({ status: 'success', found: false });
    render(
      <MessageContent
        content={'[MAP: 3.10, 101.70 | Police | Main Street]\n\n[MAP: 3.11, 101.71 | Hospital]'}
      />,
    );
    await waitFor(() => expect(screen.getByRole('img', { name: /Map of Police, Hospital/ })).toBeInTheDocument());
    expect(geocodePlace).toHaveBeenCalledTimes(2);
    expect(geocodePlace).toHaveBeenCalledWith('Police, Main Street', 3.1, 101.7);
    expect(buildMarkersMapUrl).toHaveBeenLastCalledWith([
      { lat: 3.2, lng: 101.8, label: 'Police', address: 'Main Street' },
      { lat: 3.11, lng: 101.71, label: 'Hospital', address: undefined },
    ]);
  });

  it('falls back to plain paragraph rendering for a single pipe line', () => {
    render(<MessageContent content="Path | Tokyo | Kyoto" />);
    expect(screen.getByText('Path | Tokyo | Kyoto')).toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });

  it('upgrades legacy zero-based activity summaries to one-based labels', () => {
    render(
      <MessageContent
        content={'* [0] attraction: Osaka Castle\n* [1] restaurant: Kani Doraku'}
      />,
    );

    expect(screen.getByText(/\[1\] attraction: Osaka Castle/)).toBeInTheDocument();
    expect(screen.getByText(/\[2\] restaurant: Kani Doraku/)).toBeInTheDocument();
    expect(screen.queryByText(/\[0\] attraction/)).not.toBeInTheDocument();
  });

  it('does not shift summaries that are already one-based', () => {
    render(<MessageContent content={'* [1] attraction: Osaka Castle'} />);

    expect(screen.getByText(/\[1\] attraction: Osaka Castle/)).toBeInTheDocument();
    expect(screen.queryByText(/\[2\] attraction/)).not.toBeInTheDocument();
  });
});


describe('ItineraryView structured renderer', () => {
  beforeEach(() => {
    buildStaticMapUrl.mockReset();
    buildStaticMapUrl.mockReturnValue('https://maps.test/route.png');
  });

  const completeSnapshot = (days: any[], total: number = 1000, countryCode = 'JP') => {
    const completeDays = days.map((day, dayIndex) => {
      const sourceActivities = day.activities?.length
        ? day.activities
        : [{ name: `Day ${dayIndex + 1} anchor`, type: 'attraction' }];
      return {
        ...day,
        day: dayIndex + 1,
        date: day.date ?? `2026-08-${String(dayIndex + 1).padStart(2, '0')}`,
        day_total_cost: day.day_total_cost ?? 0,
        activities: sourceActivities.map((activity: any, activityIndex: number) => ({
          ...activity,
          type: activity.type ?? 'attraction',
          address: activity.address ?? `${activity.name} address`,
          estimated_cost: activity.estimated_cost ?? 0,
          order: activityIndex + 1,
          location: {
            place_name: activity.name,
            country_code: countryCode,
            requested_city: countryCode === 'SG' ? 'Singapore' : 'Tokyo',
            verified_locality: countryCode === 'SG' ? 'Singapore' : 'Tokyo',
            latitude: 35 + dayIndex + activityIndex / 100,
            longitude: 139 + dayIndex + activityIndex / 100,
            ...(activity.location ?? {}),
          },
        })),
      };
    });
    const completeMaps = Object.fromEntries(completeDays.map((day) => {
      const points = day.activities.map((activity: any) => ({
        type: 'Feature' as const,
        geometry: {
          type: 'Point' as const,
          coordinates: [activity.location.longitude, activity.location.latitude],
        },
        properties: { name: activity.name, type: activity.type, order: activity.order },
      }));
      const routes = Object.entries(day.route?.profiles ?? {})
        .filter(([, metric]) => metric != null)
        .map(([profile, metric]: [string, any]) => ({
          type: 'Feature' as const,
          geometry: {
            type: 'LineString' as const,
            coordinates: [[139, 35], [139.1, 35.1]],
          },
          properties: {
            type: 'route', profile,
            distance_km: metric.distance_km, duration_mins: metric.duration_mins,
          },
        }));
      return [String(day.day), { type: 'FeatureCollection' as const, features: [...points, ...routes] }];
    }));
    return {
      itinerary: completeDays,
      maps: completeMaps,
      budget: { total, currency: 'JPY', allocation: { transportation: total } },
      destination_country_code: countryCode,
    };
  };

  const renderComplete = (days: any[], total?: number, countryCode?: string) => {
    const snapshot = completeSnapshot(days, total, countryCode);
    return {
      snapshot,
      view: render(<ItineraryView
        itinerary={snapshot.itinerary}
        maps={snapshot.maps}
        budget={snapshot.budget}
        destinationCountryCode={snapshot.destination_country_code}
      />),
    };
  };

  const itinerary = [{
    day: 1,
    date: '2026-08-01',
    day_total_cost: 850,
    flight: [{
      airline: 'Test Air',
      flight_number: 'TA100',
      departure_airport: { id: 'KUL', name: 'KLIA' },
      arrival_airport: { id: 'NRT', name: 'Narita' },
      departure_time: '2026-08-01 08:00',
      arrival_time: '2026-08-01 16:00',
      duration: 480,
      stops: 0,
      price: 500,
      booking_url: 'https://book.test/flight',
      over_budget: true,
    }],
    hotel: {
      hotel_name: 'Sakura Inn',
      hotel_class: 4,
      overall_rating: 4.6,
      reviews: 321,
      price_per_night: 300,
      amenities: ['WiFi', 'Pool'],
      location: { lat: 34.9, lng: 138.9, country_code: 'JP' },
      booking_url: 'https://book.test/hotel',
      over_budget: true,
    },
    activities: [{
      name: 'Tokyo Tower',
      type: 'attraction',
      rating: 4.5,
      address: 'Minato City',
      suggested_time: 'Evening',
      estimated_cost: 50,
      is_estimated: true,
      order: 1,
      location: {
        place_name: 'Tokyo Tower', country_code: 'JP', requested_city: 'Tokyo',
        verified_locality: 'Tokyo',
        latitude: 35, longitude: 139,
      },
    }],
    route: {
      ordered_stops: ['Sakura Inn', 'Tokyo Tower'],
      profiles: {
        driving: { distance_km: 10, duration_mins: 20 },
        walking: { distance_km: 8, duration_mins: 100 },
      },
    },
  }];

  const maps = {
    '1': { type: 'FeatureCollection' as const, features: [
      {
        type: 'Feature' as const,
        geometry: { type: 'Point' as const, coordinates: [138.9, 34.9] },
        properties: { name: 'Sakura Inn', type: 'hotel', order: 0 },
      },
      {
        type: 'Feature' as const,
        geometry: { type: 'Point' as const, coordinates: [139, 35] },
        properties: { name: 'Tokyo Tower', type: 'attraction', order: 1 },
      },
      {
        type: 'Feature' as const,
        geometry: { type: 'LineString' as const, coordinates: [[138.9, 34.9], [139, 35]] },
        properties: { type: 'route', profile: 'driving', distance_km: 10, duration_mins: 20 },
      },
      {
        type: 'Feature' as const,
        geometry: { type: 'LineString' as const, coordinates: [[138.9, 34.9], [139, 35]] },
        properties: { type: 'route', profile: 'walking', distance_km: 8, duration_mins: 100 },
      },
    ] },
  };

  const guardedSnapshot = (countryCode: string = 'JP') => ({
    itinerary: [{
      day: 1,
      date: '2026-08-01',
      flight: null,
      hotel: null,
      activities: [{
        name: 'Tokyo Tower',
        type: 'attraction',
        address: 'Minato City, Tokyo',
        estimated_cost: 50,
        order: 1,
        location: {
          place_name: 'Tokyo Tower',
          latitude: 35.6586,
          longitude: 139.7454,
          country_code: countryCode,
          requested_city: 'Tokyo',
          verified_locality: 'Tokyo',
        },
      }],
      route: null,
      day_total_cost: 50,
    }],
    maps: {
      '1': {
        type: 'FeatureCollection' as const,
        features: [{
          type: 'Feature' as const,
          geometry: { type: 'Point' as const, coordinates: [139.7454, 35.6586] },
          properties: { name: 'Tokyo Tower', type: 'attraction', order: 1 },
        }],
      },
    },
    budget: { total: 50, currency: 'JPY', allocation: { activity: 50 } },
    destination_country_code: 'JP',
  });

  it('does not render an unsafe snapshot when ItineraryCanvas is invoked directly', () => {
    const unsafe = guardedSnapshot('MY');
    const { container } = render(
      <ItineraryCanvas snapshot={unsafe} onClose={() => undefined} isMobileModal={false} />,
    );
    expect(container).toBeEmptyDOMElement();
    expect(screen.queryByText('Tokyo Tower')).not.toBeInTheDocument();
  });

  it.each([
    ['null', null],
    ['array', []],
    ['primitive', 'unsafe'],
  ])('does not throw or render when ItineraryCanvas receives a %s runtime snapshot', (
    _label,
    runtimeSnapshot,
  ) => {
    let container: HTMLElement | undefined;
    expect(() => {
      container = render(
        <ItineraryCanvas
          snapshot={runtimeSnapshot as never}
          onClose={() => undefined}
          isMobileModal={false}
        />,
      ).container;
    }).not.toThrow();
    expect(container!).toBeEmptyDOMElement();
  });

  it('renders a valid snapshot when ItineraryCanvas is invoked directly', () => {
    render(
      <ItineraryCanvas
        snapshot={guardedSnapshot()}
        onClose={() => undefined}
        isMobileModal={false}
      />,
    );
    expect(screen.getByRole('complementary', { name: 'Latest itinerary' })).toBeInTheDocument();
    expect(screen.getByText('Tokyo Tower')).toBeInTheDocument();
  });

  it('does not render wrong-country runtime props when ItineraryView is invoked directly', () => {
    const unsafe = guardedSnapshot('MY');
    const { container } = render(
      <ItineraryView
        itinerary={unsafe.itinerary}
        maps={unsafe.maps}
        budget={unsafe.budget}
        destinationCountryCode="JP"
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('does not infer over an explicitly null direct-view country prop', () => {
    const snapshot = guardedSnapshot();
    const { container } = render(
      <ItineraryView
        itinerary={snapshot.itinerary}
        maps={snapshot.maps}
        budget={snapshot.budget}
        destinationCountryCode={null as never}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('supports an absent destination country for a legacy direct-view snapshot', () => {
    const snapshot = guardedSnapshot();
    render(
      <ItineraryView
        itinerary={snapshot.itinerary}
        maps={snapshot.maps}
        budget={snapshot.budget}
      />,
    );
    expect(screen.getByText('Tokyo Tower')).toBeInTheDocument();
  });

  it('renders a complete destination-correct snapshot when invoked directly', () => {
    const valid = guardedSnapshot();
    render(
      <ItineraryView
        itinerary={valid.itinerary}
        maps={valid.maps}
        budget={valid.budget}
        destinationCountryCode={valid.destination_country_code}
      />,
    );
    expect(screen.getByText('Tokyo Tower')).toBeInTheDocument();
  });

  it('returns no markup for an empty itinerary', () => {
    const { container } = render(<ItineraryView itinerary={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders budget, day, flight, hotel, activity and route sections', () => {
    render(
      <ItineraryView
        itinerary={itinerary}
        maps={maps}
        budget={{
          total: 5000,
          currency: 'JPY',
          allocation: { transportation: 1000, accommodation: 2000, food: 1000, activity: 1000 },
        }}
        destinationCountryCode="JP"
      />,
    );
    expect(screen.getByText('Budget allocation')).toBeInTheDocument();
    expect(screen.getByText(/5,000/)).toBeInTheDocument();
    expect(screen.getAllByText('JPY').length).toBeGreaterThan(0);
    expect(screen.getByText('Day 1')).toBeInTheDocument();
    expect(screen.getByText('Test Air')).toBeInTheDocument();
    expect(screen.getByText('TA100')).toBeInTheDocument();
    expect(screen.getByText('Sakura Inn')).toBeInTheDocument();
    expect(screen.getByText('Tokyo Tower')).toBeInTheDocument();
    expect(screen.getByText(/Daily Route & Map/)).toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'Day route map' })).toHaveAttribute(
      'src',
      'https://maps.test/route.png',
    );
    expect(screen.getAllByText('Over budget')).toHaveLength(2);
  });

  it('keeps long activity names and addresses fully available without truncation', () => {
    const name = 'Sentosa Nature Discovery Boardwalk and Coastal Heritage Trail';
    const address = '50 Siloso Beach Walk, Sentosa Island, Singapore 098982';
    const longItinerary = [{
      ...itinerary[0],
      activities: [{
        ...itinerary[0].activities[0],
        name,
        address,
        location: { ...itinerary[0].activities[0].location, place_name: name },
      }],
      route: { ...itinerary[0].route, ordered_stops: ['Sakura Inn', name] },
    }];
    const longMaps = {
      '1': {
        ...maps['1'],
        features: maps['1'].features.map((feature) => (
          feature.geometry.type === 'Point' && feature.properties.type !== 'hotel'
            ? { ...feature, properties: { ...feature.properties, name } }
            : feature
        )),
      },
    };

    render(
      <ItineraryView
        itinerary={longItinerary}
        maps={longMaps}
        budget={{
          total: 5000,
          currency: 'JPY',
          allocation: { transportation: 1000, accommodation: 2000, food: 1000, activity: 1000 },
        }}
        destinationCountryCode="JP"
      />,
    );

    const heading = screen.getByRole('heading', { level: 5, name });
    const addressText = screen.getByText(address);
    expect(heading).toHaveTextContent(name);
    expect(addressText).toHaveTextContent(address);
    expect(heading).not.toHaveClass('truncate');
    expect(addressText).not.toHaveClass('truncate');
    expect(heading).toHaveClass('break-words');
    expect(addressText).toHaveClass('break-words');
  });

  it('labels and links outbound and return flights on the boundary days', () => {
    renderComplete([
      {
        day: 1,
        date: '2026-08-01',
        flight: [{
          airline: 'Outbound Air',
          flight_number: 'OA100',
          departure_airport: { id: 'KUL', name: 'KLIA' },
          arrival_airport: { id: 'NRT', name: 'Narita' },
          departure_time: '2026-08-01 08:00',
          arrival_time: '2026-08-01 16:00',
          price: 500,
          booking_url: 'https://book.test/outbound',
        }],
        activities: [],
        day_total_cost: 500,
      },
      { day: 2, date: '2026-08-02', activities: [], day_total_cost: 0 },
      {
        day: 3,
        date: '2026-08-03',
        flight: [{
          airline: 'Return Air',
          flight_number: 'RA200',
          departure_airport: { id: 'NRT', name: 'Narita' },
          arrival_airport: { id: 'KUL', name: 'KLIA' },
          departure_time: '2026-08-03 10:00',
          arrival_time: '2026-08-03 18:00',
          price: 350,
          booking_url: 'https://book.test/return',
        }],
        activities: [],
        day_total_cost: 350,
      },
    ], 850);

    const outboundCard = screen.getByRole('group', { name: 'Outbound flight ticket' });
    expect(within(outboundCard).getByText('Outbound flight')).toBeInTheDocument();
    expect(within(outboundCard).getByText('Outbound Air')).toBeInTheDocument();
    expect(within(outboundCard).getByText('OA100')).toBeInTheDocument();
    expect(within(outboundCard).getByText('KUL')).toBeInTheDocument();
    expect(within(outboundCard).getByText('NRT')).toBeInTheDocument();
    expect(within(outboundCard).getByRole('link', { name: 'Book outbound flight' }))
      .toHaveAttribute('href', 'https://book.test/outbound');

    const returnCard = screen.getByRole('group', { name: 'Return flight ticket' });
    expect(within(returnCard).getByText('Return flight')).toBeInTheDocument();
    expect(within(returnCard).getByText('Return Air')).toBeInTheDocument();
    expect(within(returnCard).getByText('RA200')).toBeInTheDocument();
    expect(within(returnCard).getByText('NRT')).toBeInTheDocument();
    expect(within(returnCard).getByText('KUL')).toBeInTheDocument();
    expect(within(returnCard).getByRole('link', { name: 'Book return flight' }))
      .toHaveAttribute('href', 'https://book.test/return');
  });

  it('keeps one-day and middle-day legacy flight cards generic', () => {
    const legacyFlight = {
      airline: 'Legacy Air',
      departure_airport: { id: 'KUL', name: 'KLIA' },
      arrival_airport: { id: 'SIN', name: 'Changi' },
      price: 100,
      booking_url: 'https://book.test/legacy',
    };
    const first = completeSnapshot([
      { day: 1, flight: [legacyFlight], activities: [], day_total_cost: 100 },
    ], 100);
    const { rerender } = render(<ItineraryView
      itinerary={first.itinerary} maps={first.maps} budget={first.budget}
      destinationCountryCode={first.destination_country_code}
    />);

    let legacyCard = screen.getByRole('group', { name: 'Flight ticket' });
    expect(within(legacyCard).getByText('Flight')).toBeInTheDocument();
    expect(within(legacyCard).getByRole('link', { name: 'Book flight' }))
      .toHaveAttribute('href', 'https://book.test/legacy');

    const middle = completeSnapshot([
      { day: 1, activities: [], day_total_cost: 0 },
      { day: 2, flight: [legacyFlight], activities: [], day_total_cost: 100 },
      { day: 3, activities: [], day_total_cost: 0 },
    ], 100);
    rerender(<ItineraryView
      itinerary={middle.itinerary} maps={middle.maps} budget={middle.budget}
      destinationCountryCode={middle.destination_country_code}
    />);

    legacyCard = screen.getByRole('group', { name: 'Flight ticket' });
    expect(within(legacyCard).getByText('Legacy Air')).toBeInTheDocument();
    expect(within(legacyCard).getByText('KUL')).toBeInTheDocument();
    expect(within(legacyCard).getByText('SIN')).toBeInTheDocument();
    expect(within(legacyCard).getByRole('link', { name: 'Book flight' }))
      .toHaveAttribute('href', 'https://book.test/legacy');
  });

  it('changes active route metrics and rebuilds the map when profile changes', () => {
    render(<ItineraryView
      itinerary={itinerary} maps={maps}
      budget={{ total: 850, currency: 'JPY', allocation: { transportation: 850 } }}
      destinationCountryCode="JP"
    />);
    expect(screen.getByText('20 min')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Walking 8km/ }));
    expect(screen.getByText('100 min')).toBeInTheDocument();
    expect(buildStaticMapUrl).toHaveBeenLastCalledWith(maps['1'], { profile: 'walking' });
  });

  it('shows a configuration fallback when a route has no image URL', () => {
    buildStaticMapUrl.mockReturnValue(null);
    render(<ItineraryView
      itinerary={itinerary} maps={maps}
      budget={{ total: 850, currency: 'JPY', allocation: { transportation: 850 } }}
      destinationCountryCode="JP"
    />);
    expect(screen.getByText('No mappable locations for this day yet.')).toBeInTheDocument();
  });

  it('hides optional flight and hotel sections when data is absent', () => {
    renderComplete([{ day: 1, activities: [], day_total_cost: 0 }], 0);
    expect(screen.getByText('Day 1')).toBeInTheDocument();
    expect(screen.queryByText(/Flight/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Hotel/)).not.toBeInTheDocument();
    expect(screen.getByText(/Daily Route/)).toBeInTheDocument();
  });

  it('hides malformed legacy place edits that were never verified', () => {
    render(<ItineraryView itinerary={[{
      day: 1,
      activities: [{
        name: 'Nearby Attraction to Umeda Sky Building',
        rating: 4.2,
        address: 'Ikeda, Osaka',
      }],
    }]} />);

    expect(screen.queryByText('Nearby Attraction to Umeda Sky Building')).not.toBeInTheDocument();
  });

  it('shows the place icon fallback when a remote thumbnail fails', () => {
    renderComplete([{
      day: 1,
      activities: [{
        name: 'Umeda Sky Building',
        type: 'attraction',
        thumbnail: 'https://images.test/broken.jpg',
      }],
    }]);

    fireEvent.error(screen.getByRole('img', { name: 'Umeda Sky Building' }));

    expect(screen.queryByRole('img', { name: 'Umeda Sky Building' })).not.toBeInTheDocument();
    expect(screen.getByLabelText('No photo available for Umeda Sky Building')).toBeInTheDocument();
  });
});
