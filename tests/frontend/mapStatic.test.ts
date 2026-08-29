async function loadMaps(token = 'pk.test') {
  vi.resetModules();
  vi.stubEnv('VITE_MAPBOX_TOKEN', token);
  return import('../../fyp_frontend/src/lib/mapStatic');
}

const point = (lng: number, lat: number, type = 'activity') => ({
  type: 'Feature' as const,
  geometry: { type: 'Point' as const, coordinates: [lng, lat] },
  properties: { type },
});

const route = (profile: string, coordinates: number[][]) => ({
  type: 'Feature' as const,
  geometry: { type: 'LineString' as const, coordinates },
  properties: { type: 'route', profile },
});


describe('Mapbox static URL builders', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it('returns null when no public token is configured', async () => {
    const maps = await loadMaps('');
    expect(maps.buildMarkersMapUrl([{ lat: 1, lng: 2 }])).toBeNull();
    expect(maps.buildStaticMapUrl({ type: 'FeatureCollection', features: [point(2, 1)] })).toBeNull();
  });

  it.each([null, undefined, [], [null]])('rejects absent or invalid marker collections', async (value) => {
    const maps = await loadMaps();
    expect(maps.buildMarkersMapUrl(value as never)).toBeNull();
  });

  it('builds a street-level single-pin URL without incompatible padding', async () => {
    const maps = await loadMaps();
    const url = maps.buildMarkersMapUrl([{ lat: 3.139, lng: 101.6869 }]);
    expect(url).toContain('pin-l-marker+ef4444(101.68690,3.13900)');
    expect(url).toContain('/101.68690,3.13900,14/640x320@2x?access_token=pk.test');
    expect(url).not.toContain('padding=');
  });

  it('labels multi-pin maps in caption order and enables auto padding', async () => {
    const maps = await loadMaps();
    const url = maps.buildMarkersMapUrl(
      [{ lat: 1, lng: 2 }, { lat: 3, lng: 4 }],
      { width: 800, height: 400, color: '00ff00' },
    );
    expect(url).toContain('pin-l-a+00ff00(2.00000,1.00000)');
    expect(url).toContain('pin-l-b+00ff00(4.00000,3.00000)');
    expect(url).toContain('/auto/800x400@2x?padding=60&');
  });

  it('limits marker overlays to the Mapbox-supported ten labels', async () => {
    const maps = await loadMaps();
    const points = Array.from({ length: 12 }, (_, index) => ({ lat: index, lng: index }));
    const url = maps.buildMarkersMapUrl(points)!;
    expect((url.match(/pin-l-/g) ?? []).length).toBe(10);
    expect(url).toContain('pin-l-j+');
  });

  it.each([
    [[{ lat: Number.NaN, lng: 1 }]],
    [[{ lat: 1, lng: Number.POSITIVE_INFINITY }]],
    [[{ lat: 91, lng: 1 }]],
    [[{ lat: 1, lng: 181 }]],
  ])('rejects non-finite or out-of-range marker coordinates', async (points) => {
    const maps = await loadMaps();
    expect(maps.buildMarkersMapUrl(points)).toBeNull();
  });

  it.each([null, undefined, { type: 'FeatureCollection', features: [] }])(
    'returns null for empty feature collections',
    async (fc) => {
      const maps = await loadMaps();
      expect(maps.buildStaticMapUrl(fc as never)).toBeNull();
    },
  );

  it('requires at least one valid point even if a route exists', async () => {
    const maps = await loadMaps();
    const fc = {
      type: 'FeatureCollection' as const,
      features: [route('driving', [[1, 2], [3, 4]])],
    };
    expect(maps.buildStaticMapUrl(fc)).toBeNull();
  });

  it('uses semantic colors for hotel, airport, restaurant, attraction and default pins', async () => {
    const maps = await loadMaps();
    const fc = {
      type: 'FeatureCollection' as const,
      features: [
        point(1, 1, 'hotel'),
        point(2, 2, 'airport'),
        point(3, 3, 'restaurant'),
        point(4, 4, 'attraction'),
        point(5, 5, 'unknown'),
      ],
    };
    const url = maps.buildStaticMapUrl(fc)!;
    for (const color of ['2563eb', '64748b', 'f59e0b', '10b981', '6366f1']) {
      expect(url).toContain(`pin-s+${color}`);
    }
  });

  it('draws the selected route profile before pins', async () => {
    const maps = await loadMaps();
    const features = [
      point(139.7, 35.6, 'hotel'),
      point(139.8, 35.7, 'attraction'),
      route('driving', [[139.7, 35.6], [139.8, 35.7]]),
      route('walking', [[139.7, 35.6], [139.75, 35.65], [139.8, 35.7]]),
    ];
    const fc = { type: 'FeatureCollection' as const, features };
    const driving = maps.buildStaticMapUrl(fc, { profile: 'driving' })!;
    const walking = maps.buildStaticMapUrl(fc, { profile: 'walking' })!;
    expect(driving).toContain('path-4+2563eb-0.75');
    expect(walking).toContain('path-4+2563eb-0.75');
    expect(driving).not.toBe(walking);
    expect(driving.indexOf('path-4')).toBeLessThan(driving.indexOf('pin-s'));
  });

  it('falls back to the first route when the requested profile is absent', async () => {
    const maps = await loadMaps();
    const fc = {
      type: 'FeatureCollection' as const,
      features: [point(1, 1), point(2, 2), route('walking', [[1, 1], [2, 2]])],
    };
    expect(maps.buildStaticMapUrl(fc, { profile: 'cycling' })).toContain('path-4');
  });

  it('keeps pins but drops a route that would exceed safe URL length', async () => {
    const maps = await loadMaps();
    const coordinates = Array.from({ length: 3000 }, (_, i) => [
      ((i * 37) % 300) - 150,
      ((i * 53) % 140) - 70,
    ]);
    const fc = {
      type: 'FeatureCollection' as const,
      features: [point(1, 1), point(2, 2), route('driving', coordinates)],
    };
    const url = maps.buildStaticMapUrl(fc)!;
    expect(url).not.toContain('path-4');
    expect((url.match(/pin-s/g) ?? []).length).toBe(2);
    expect(url.length).toBeLessThan(7800);
  });

  it('rejects NaN and out-of-range GeoJSON point coordinates', async () => {
    const maps = await loadMaps();
    const fc = {
      type: 'FeatureCollection' as const,
      features: [point(Number.NaN, 1), point(1, 91)],
    };
    expect(maps.buildStaticMapUrl(fc)).toBeNull();
  });
});
