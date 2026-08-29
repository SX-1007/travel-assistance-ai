/**
 * MessageContent.tsx — Rich renderer for the AI assistant's replies.
 *
 * The planner agent replies in a structured, markdown-flavoured format
 * (see docs/audits/archive/original-itinerary-ui-requirement-2026-07-05.txt).
 * A reply is either:
 *
 *   1. A STRUCTURED ITINERARY — wrapped in <Sequence> with one or more
 *      <Step title="Day 1: …" subtitle="…"> blocks. Inside each step:
 *        • timeline bullets  ->  `*   **2:00 PM:** Take a Grab to …`
 *        • "LIVE API RESULT" cards  ->  markdown blockquotes (`> …`) holding
 *          a header, a name, a 📍 meta line, a markdown table, and
 *          `[ Book Now ]`-style action buttons.
 *
 *   2. A PLAIN ANSWER — the agent just answers a question. Rendered with a
 *      lightweight markdown pass (headings, lists, tables, blockquotes, bold).
 *
 * This module parses that format and renders it as first-class UI instead of
 * dumping the raw markdown as paragraphs. It intentionally has no external
 * markdown dependency: the grammar is small and well-defined, so a focused
 * parser gives us full control over the card styling.
 */
import React from 'react';
import {
  Hotel, Utensils, Route, MapPin, Star, Clock, Navigation,
  ExternalLink, Ticket, Sparkles, Info, ChevronRight,
} from 'lucide-react';
import { buildMarkersMapUrl, type MarkerPoint } from '../../lib/mapStatic';
import { geocodePlace } from '../../lib/api';

// ─────────────────────────────────────────────────────────────────────
// [MAP: <lat>, <lng> | <label> | <address?>] — location marker lines
// emitted by the agent for emergency / place lookups (see prompts.py §5).
// The LLM-transcribed coordinates are only a rough hint: each pin is
// re-geocoded by name + address through /api/geocode (Google Maps data)
// and only falls back to the line's coordinates if the lookup misses.
// Rendered as a static map with pins + "open in Google Maps" links.
// ─────────────────────────────────────────────────────────────────────

const MAP_LINE_RE =
  /^\[MAP:\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\|\s*([^\]]+)\]$/i;

/** Shared across messages/re-renders so each place geocodes exactly once. */
const geoCache = new Map<string, Promise<{ lat: number; lng: number } | null>>();

/** Re-geocode one pin by "name, address" (LLM coords bias the search). */
function resolvePoint(p: MarkerPoint): Promise<MarkerPoint> {
  const query = [p.label, p.address].filter(Boolean).join(', ').trim();
  if (query.length < 3) return Promise.resolve(p);
  let hit = geoCache.get(query);
  if (!hit) {
    hit = geocodePlace(query, p.lat, p.lng)
      .then((r) =>
        r.found && Number.isFinite(r.lat) && Number.isFinite(r.lng)
          ? { lat: r.lat as number, lng: r.lng as number }
          : null,
      )
      .catch(() => {
        geoCache.delete(query); // transient failure — allow a retry later
        return null;
      });
    geoCache.set(query, hit);
  }
  return hit.then((coords) => (coords ? { ...p, ...coords } : p));
}

const LocationMap = ({ points }: { points: MarkerPoint[] }) => {
  // Render the LLM's rough pins immediately, then snap to geocoded ones.
  const [pins, setPins] = React.useState<MarkerPoint[]>(points);
  const pointsKey = points
    .map((p) => `${p.lat},${p.lng}|${p.label}|${p.address ?? ''}`)
    .join(';');
  React.useEffect(() => {
    let alive = true;
    Promise.all(points.map(resolvePoint)).then((resolved) => {
      if (alive) setPins(resolved);
    });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pointsKey]);

  const url = buildMarkersMapUrl(pins);
  // Prefer a text search (name + address) — it lands on the actual Google
  // Maps place page; raw coordinates only when there is nothing to search.
  const gmaps = (p: MarkerPoint) => {
    const query = [p.label, p.address].filter(Boolean).join(', ').trim();
    return query.length >= 3
      ? `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(query)}`
      : `https://www.google.com/maps/search/?api=1&query=${p.lat},${p.lng}`;
  };
  const letters = 'abcdefghij';

  return (
    <div className="my-3 overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
      {url && (
        <a href={gmaps(pins[0])} target="_blank" rel="noreferrer" title="Open in Google Maps">
          <img
            src={url}
            alt={`Map of ${pins.map((p) => p.label).join(', ')}`}
            className="w-full object-cover"
            loading="lazy"
          />
        </a>
      )}
      <div className="space-y-1 px-3 py-2">
        {pins.map((p, i) => (
          <a
            key={i}
            href={gmaps(p)}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 text-xs font-medium text-blue-600 hover:underline"
          >
            <MapPin size={12} className="shrink-0 text-red-500" />
            <span className="truncate">
              {pins.length > 1 ? `(${letters[i] ?? '•'}) ` : ''}
              {p.label}
              {p.address ? (
                <span className="ml-1 font-normal text-slate-400">· {p.address}</span>
              ) : null}
            </span>
            <ExternalLink size={10} className="shrink-0 opacity-60" />
          </a>
        ))}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────
// Inline markdown  (**bold**, *italic*, `code`, `[ Button ]` chips)
// ─────────────────────────────────────────────────────────────────────

/** A ``[ … ]`` chip rendered as a pill button (booking / map actions). */
const ActionChip = ({ label }: { label: string }) => {
  const lower = label.toLowerCase();
  const isPrimary = /book|reserve|order/.test(lower);
  const isMap = /map|route|direction|expand|menu/.test(lower);
  const Icon = isMap ? (/menu/.test(lower) ? Ticket : Navigation) : null;
  return (
    <button
      type="button"
      className={
        'inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ' +
        'transition-colors align-middle ' +
        (isPrimary
          ? 'bg-blue-600 text-white hover:bg-blue-700 shadow-sm'
          : 'bg-white text-slate-600 border border-slate-300 hover:border-blue-400 hover:text-blue-600')
      }
    >
      {Icon && <Icon size={13} />}
      {label}
    </button>
  );
};

/**
 * Tokenise a single line of inline markdown into React nodes.
 * Handles nested emphasis loosely (good enough for the agent's output).
 */
function renderInline(text: string, keyBase: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  // Order matters: bold before italic so ** wins over *.
  const pattern = /(\*\*[^*]+\*\*|\*[^*\n]+\*|`[^`]+`|~~[^~]+~~)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;

  while ((m = pattern.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    const tok = m[0];
    const key = `${keyBase}-${i++}`;

    if (tok.startsWith('**')) {
      nodes.push(
        <strong key={key} className="font-semibold text-slate-900">
          {tok.slice(2, -2)}
        </strong>,
      );
    } else if (tok.startsWith('~~')) {
      nodes.push(
        <s key={key} className="text-slate-400">
          {tok.slice(2, -2)}
        </s>,
      );
    } else if (tok.startsWith('`')) {
      const inner = tok.slice(1, -1).trim();
      const chip = inner.match(/^\[\s*(.*?)\s*\]$/); // `[ Book Now ]`
      if (chip) {
        nodes.push(<ActionChip key={key} label={chip[1]} />);
      } else {
        nodes.push(
          <code
            key={key}
            className="rounded bg-slate-100 px-1.5 py-0.5 text-[0.85em] font-mono text-slate-700"
          >
            {inner}
          </code>,
        );
      }
    } else {
      // single-asterisk italic
      nodes.push(
        <em key={key} className="italic text-slate-600">
          {tok.slice(1, -1)}
        </em>,
      );
    }
    last = m.index + tok.length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

// ─────────────────────────────────────────────────────────────────────
// Markdown table
// ─────────────────────────────────────────────────────────────────────

const splitRow = (row: string): string[] =>
  row
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((c) => c.trim());

const isSeparatorRow = (cells: string[]): boolean =>
  cells.every((c) => /^:?-{2,}:?$/.test(c.replace(/\s/g, '')));

const MdTable = ({ lines, keyBase }: { lines: string[]; keyBase: string }) => {
  const rows = lines.map(splitRow);
  let header: string[] | null = null;
  let body = rows;
  // First row is a header when the second row is a `---` separator.
  if (rows.length >= 2 && isSeparatorRow(rows[1])) {
    header = rows[0];
    body = rows.slice(2);
  } else if (rows.length && isSeparatorRow(rows[0])) {
    body = rows.slice(1);
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-slate-200 my-2">
      <table className="w-full text-left text-sm">
        {header && (
          <thead className="bg-slate-100/80 text-xs uppercase tracking-wide text-slate-500">
            <tr>
              {header.map((c, i) => (
                <th key={i} className="px-3 py-2 font-semibold">
                  {renderInline(c, `${keyBase}-h${i}`)}
                </th>
              ))}
            </tr>
          </thead>
        )}
        <tbody className="divide-y divide-slate-100">
          {body.map((cells, r) => (
            <tr key={r} className="hover:bg-slate-50/60">
              {cells.map((c, i) => (
                <td key={i} className="px-3 py-2 text-slate-700 align-middle">
                  {renderInline(c, `${keyBase}-r${r}c${i}`)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────
// "LIVE API RESULT" card  (parsed from a markdown blockquote)
// ─────────────────────────────────────────────────────────────────────

type CardKind = 'hotel' | 'food' | 'route' | 'info';

const CARD_THEME: Record<
  CardKind,
  { icon: React.ElementType; ring: string; head: string; chipText: string; label: string }
> = {
  hotel: {
    icon: Hotel,
    ring: 'border-indigo-200',
    head: 'bg-indigo-50 text-indigo-700',
    chipText: 'text-indigo-600',
    label: 'Accommodation',
  },
  food: {
    icon: Utensils,
    ring: 'border-amber-200',
    head: 'bg-amber-50 text-amber-700',
    chipText: 'text-amber-600',
    label: 'Restaurant & Places',
  },
  route: {
    icon: Route,
    ring: 'border-emerald-200',
    head: 'bg-emerald-50 text-emerald-700',
    chipText: 'text-emerald-600',
    label: 'Daily Routing Map',
  },
  info: {
    icon: Info,
    ring: 'border-slate-200',
    head: 'bg-slate-100 text-slate-600',
    chipText: 'text-slate-500',
    label: 'Details',
  },
};

function detectCardKind(headerText: string): CardKind {
  const u = headerText.toUpperCase();
  if (/ACCOMMODATION|HOTEL|STAY|LODG/.test(u)) return 'hotel';
  if (/RESTAURANT|PLACE|DINING|FOOD|CAFE|EAT/.test(u)) return 'food';
  if (/ROUTING|ROUTE|MAP|SPATIAL|TRANSIT|OVERVIEW/.test(u)) return 'route';
  return 'info';
}

/** True when a line is only made of `[ … ]` action chips (and separators). */
const isActionLine = (line: string): boolean => {
  const stripped = line.replace(/`\[[^\]]*\]`/g, '').replace(/[|\s]/g, '');
  return stripped.length === 0 && /`\[[^\]]*\]`/.test(line);
};

/** A line like `📍 *Path: A → B*` or `⭐ 7.2/10` used as sub-header meta. */
const isMetaLine = (line: string): boolean =>
  /^[\s>]*(📍|⭐|🕒|🟢|🟡|🔴|🏨|🍽️|🗺️)/.test(line) === false &&
  /(📍|⭐|🕒)/.test(line);

/**
 * A blockquote is a "LIVE API RESULT" card only when it announces itself as
 * one. Everything else is a normal quote (tips, callouts, disclaimers).
 */
const isApiResultQuote = (lines: string[]): boolean =>
  lines.some((l) => /LIVE\s*API|API\s*RESULT/i.test(l));

/** Plain markdown blockquote — a subtle left-bordered callout. */
const Blockquote = ({ lines, keyBase }: { lines: string[]; keyBase: string }) => {
  const inner = lines.map((l) => l.replace(/^\s*>\s?/, '')).join('\n');
  return (
    <blockquote
      className="my-3 rounded-r-lg border-l-4 border-blue-200 bg-blue-50/50 px-4 py-2 text-sm text-slate-600"
    >
      <Blocks segments={segmentBody(inner)} keyBase={keyBase} />
    </blockquote>
  );
};

const ApiResultCard = ({ lines, keyBase }: { lines: string[]; keyBase: string }) => {
  // Strip the leading "> " blockquote marker from every line.
  const clean = lines.map((l) => l.replace(/^\s*>\s?/, '').replace(/\s+$/, ''));
  // Drop leading/trailing blank lines.
  while (clean.length && clean[0].trim() === '') clean.shift();
  while (clean.length && clean[clean.length - 1].trim() === '') clean.pop();

  const headerRaw = clean[0] ?? '';
  const kind = detectCardKind(headerRaw);
  const theme = CARD_THEME[kind];
  const Icon = theme.icon;

  // Header label: text after a "RESULT:" colon, else the theme default.
  const headerLabel =
    headerRaw
      .replace(/\*\*/g, '')
      .replace(/^[^:]*RESULT:\s*/i, '')
      .replace(/^[^A-Za-z]*/, '')
      .trim() || theme.label;

  // ── Parse the remaining lines into a small block list ──
  type Block =
    | { t: 'name'; text: string }
    | { t: 'meta'; text: string }
    | { t: 'note'; text: string }
    | { t: 'text'; text: string }
    | { t: 'actions'; text: string }
    | { t: 'table'; lines: string[] };

  const blocks: Block[] = [];
  let firstBoldTaken = false;

  for (let i = 1; i < clean.length; i++) {
    const line = clean[i];
    if (line.trim() === '') continue;

    // Accumulate consecutive table rows.
    if (line.includes('|')) {
      const tbl: string[] = [];
      while (i < clean.length && clean[i].includes('|')) tbl.push(clean[i++]);
      i--;
      blocks.push({ t: 'table', lines: tbl });
      continue;
    }
    if (isActionLine(line)) {
      blocks.push({ t: 'actions', text: line });
      continue;
    }
    // The name is the first stand-alone **bold** line.
    if (!firstBoldTaken && /^\*\*.+\*\*/.test(line.trim())) {
      firstBoldTaken = true;
      blocks.push({ t: 'name', text: line.trim() });
      continue;
    }
    if (/(📍|⭐|🕒|🟢|🟡|🔴)/.test(line)) {
      blocks.push({ t: 'meta', text: line.trim() });
      continue;
    }
    if (/^\*[^*].*\*$/.test(line.trim())) {
      blocks.push({ t: 'note', text: line.trim() });
      continue;
    }
    blocks.push({ t: 'text', text: line.trim() });
  }

  return (
    <div className={`my-3 overflow-hidden rounded-xl border ${theme.ring} bg-white shadow-sm`}>
      {/* Header strip */}
      <div className={`flex items-center gap-2 px-4 py-2 ${theme.head}`}>
        <Icon size={16} />
        <span className="text-[11px] font-bold uppercase tracking-wider">Live API Result</span>
        <ChevronRight size={12} className="opacity-50" />
        <span className="text-xs font-semibold">{headerLabel}</span>
      </div>

      <div className="px-4 py-3 space-y-2">
        {blocks.map((b, i) => {
          const k = `${keyBase}-b${i}`;
          switch (b.t) {
            case 'name':
              return (
                <div key={k} className="text-base font-bold text-slate-900 leading-snug">
                  {renderInline(b.text, k)}
                </div>
              );
            case 'meta':
              return (
                <div
                  key={k}
                  className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-slate-500"
                >
                  {renderInline(b.text, k)}
                </div>
              );
            case 'note':
              return (
                <p key={k} className={`text-xs ${theme.chipText}`}>
                  {renderInline(b.text, k)}
                </p>
              );
            case 'table':
              return <MdTable key={k} lines={b.lines} keyBase={k} />;
            case 'actions':
              return (
                <div key={k} className="flex flex-wrap gap-2 pt-1">
                  {renderInline(b.text.replace(/\|/g, ' '), k)}
                </div>
              );
            default:
              return (
                <p key={k} className="text-sm text-slate-700 leading-relaxed">
                  {renderInline(b.text, k)}
                </p>
              );
          }
        })}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────
// Generic block renderer (shared by itinerary step bodies and plain replies)
// ─────────────────────────────────────────────────────────────────────

/** Timeline bullet: `*   **2:00 PM:** Take a Grab …` */
const TimelineItem = ({ text, keyBase }: { text: string; keyBase: string }) => {
  // Split an optional leading `**Time:**` badge from the rest of the line.
  const badge = text.match(/^\*\*(.+?):\*\*\s*(.*)$/);
  return (
    <li className="relative flex gap-3 pl-1">
      <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full bg-blue-500 ring-4 ring-blue-100" />
      <div className="flex-1 text-sm leading-relaxed text-slate-700">
        {badge ? (
          <>
            <span className="mr-2 inline-block rounded-md bg-blue-50 px-2 py-0.5 text-xs font-semibold text-blue-700">
              {badge[1]}
            </span>
            {renderInline(badge[2], keyBase)}
          </>
        ) : (
          renderInline(text, keyBase)
        )}
      </div>
    </li>
  );
};

type Segment =
  | { t: 'bullets'; items: string[] }
  | { t: 'quote'; lines: string[] }
  | { t: 'table'; lines: string[] }
  | { t: 'heading'; level: number; text: string }
  | { t: 'map'; points: MarkerPoint[] }
  | { t: 'para'; text: string };

/** Split a markdown body into ordered segments. */
function segmentBody(body: string): Segment[] {
  const lines = body.replace(/\r/g, '').split('\n');
  const segs: Segment[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (trimmed === '') {
      i++;
      continue;
    }

    // [MAP: lat, lng | label] marker lines — consecutive ones (blank lines
    // in between allowed) merge into a single multi-pin map.
    if (MAP_LINE_RE.test(trimmed)) {
      const points: MarkerPoint[] = [];
      while (i < lines.length) {
        const t = lines[i].trim();
        if (t === '') {
          // Only skip the blank if another map line follows directly.
          if (lines[i + 1] !== undefined && MAP_LINE_RE.test(lines[i + 1].trim())) {
            i++;
            continue;
          }
          break;
        }
        const mm = t.match(MAP_LINE_RE);
        if (!mm) break;
        // Group 3 is "name" or "name | address" — the address is optional.
        const [label, ...rest] = mm[3].split('|').map((s) => s.trim());
        points.push({
          lat: parseFloat(mm[1]),
          lng: parseFloat(mm[2]),
          label,
          address: rest.filter(Boolean).join(', ') || undefined,
        });
        i++;
      }
      if (points.length) segs.push({ t: 'map', points });
      continue;
    }

    // Blockquote (may contain a LIVE API card).
    if (/^\s*>/.test(line)) {
      const buf: string[] = [];
      while (i < lines.length && (/^\s*>/.test(lines[i]) || lines[i].trim() === '')) {
        // Stop the quote at a blank line that is followed by a non-quote line.
        if (lines[i].trim() === '') {
          const next = lines[i + 1];
          if (next === undefined || !/^\s*>/.test(next)) break;
        }
        buf.push(lines[i]);
        i++;
      }
      segs.push({ t: 'quote', lines: buf });
      continue;
    }

    // Heading.
    const h = trimmed.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      segs.push({ t: 'heading', level: h[1].length, text: h[2] });
      i++;
      continue;
    }

    // Table (a run of pipe lines).
    if (trimmed.includes('|') && trimmed.replace(/[^|]/g, '').length >= 1) {
      const buf: string[] = [];
      while (i < lines.length && lines[i].includes('|') && lines[i].trim() !== '') {
        buf.push(lines[i]);
        i++;
      }
      if (buf.length >= 2) {
        segs.push({ t: 'table', lines: buf });
        continue;
      }
      // Not really a table — fall through as paragraph text.
      segs.push({ t: 'para', text: buf.join(' ') });
      continue;
    }

    // Bullet list (`* `, `- `, `• `).
    if (/^\s*([*\-•])\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*([*\-•])\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*([*\-•])\s+/, '').trim());
        i++;
      }
      segs.push({ t: 'bullets', items });
      continue;
    }

    // Plain paragraph (collect consecutive non-empty, non-special lines).
    const buf: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== '' &&
      !/^\s*>/.test(lines[i]) &&
      !/^\s*([*\-•])\s+/.test(lines[i]) &&
      !/^#{1,4}\s+/.test(lines[i].trim()) &&
      !MAP_LINE_RE.test(lines[i].trim()) &&
      !lines[i].includes('|')
    ) {
      buf.push(lines[i].trim());
      i++;
    }
    segs.push({ t: 'para', text: buf.join(' ') });
  }

  return segs;
}

const HEADING_CLASS: Record<number, string> = {
  1: 'text-xl font-bold text-slate-900 mt-4 mb-2',
  2: 'text-lg font-bold text-slate-900 mt-4 mb-2',
  3: 'text-base font-semibold text-slate-900 mt-3 mb-1.5',
  4: 'text-sm font-semibold text-slate-800 mt-3 mb-1',
};

const Blocks = ({ segments, keyBase }: { segments: Segment[]; keyBase: string }) => (
  <>
    {segments.map((seg, i) => {
      const k = `${keyBase}-s${i}`;
      switch (seg.t) {
        case 'heading':
          return (
            <div key={k} className={HEADING_CLASS[seg.level] ?? HEADING_CLASS[4]}>
              {renderInline(seg.text, k)}
            </div>
          );
        case 'bullets':
          return (
            <ul key={k} className="my-2 space-y-2">
              {seg.items.map((it, j) => (
                <TimelineItem key={`${k}-${j}`} text={it} keyBase={`${k}-${j}`} />
              ))}
            </ul>
          );
        case 'table':
          return <MdTable key={k} lines={seg.lines} keyBase={k} />;
        case 'quote':
          return isApiResultQuote(seg.lines) ? (
            <ApiResultCard key={k} lines={seg.lines} keyBase={k} />
          ) : (
            <Blockquote key={k} lines={seg.lines} keyBase={k} />
          );
        case 'map':
          return <LocationMap key={k} points={seg.points} />;
        default:
          return (
            <p key={k} className="my-2 text-sm leading-relaxed text-slate-700">
              {renderInline(seg.text, k)}
            </p>
          );
      }
    })}
  </>
);

// ─────────────────────────────────────────────────────────────────────
// Itinerary <Step> cards
// ─────────────────────────────────────────────────────────────────────

interface Step {
  title: string;
  subtitle: string;
  body: string;
}

/** Pull the ``title`` / ``subtitle`` attributes off a <Step …> open tag. */
function parseStepAttrs(attrs: string): { title: string; subtitle: string } {
  // \btitle avoids matching the "title" inside "subtitle".
  const title = attrs.match(/\btitle\s*=\s*"([^"]*)"/i)?.[1] ?? '';
  const subtitle = attrs.match(/\bsubtitle\s*=\s*"([^"]*)"/i)?.[1] ?? '';
  return { title, subtitle };
}

/** Extract <Step> blocks. Returns null when the reply isn't an itinerary. */
function parseSteps(content: string): Step[] | null {
  const stepRe = /<Step\b([^>]*)>([\s\S]*?)<\/Step>/gi;
  const steps: Step[] = [];
  let m: RegExpExecArray | null;
  while ((m = stepRe.exec(content)) !== null) {
    const { title, subtitle } = parseStepAttrs(m[1]);
    steps.push({ title, subtitle, body: m[2] });
  }
  return steps.length ? steps : null;
}

const StepCard = ({ step, index, keyBase }: { step: Step; index: number; keyBase: string }) => {
  const segments = segmentBody(step.body);
  return (
    <div className="relative rounded-2xl border border-slate-200 bg-white shadow-sm">
      {/* Day header */}
      <div className="flex items-start gap-3 border-b border-slate-100 px-5 py-4">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-blue-600 text-sm font-bold text-white shadow-sm">
          {index + 1}
        </div>
        <div className="min-w-0">
          <h3 className="text-base font-bold leading-snug text-slate-900">
            {renderInline(step.title, `${keyBase}-t`)}
          </h3>
          {step.subtitle && (
            <div className="mt-0.5 flex items-center gap-1.5 text-xs font-medium text-blue-600">
              <Sparkles size={12} />
              {renderInline(step.subtitle, `${keyBase}-st`)}
            </div>
          )}
        </div>
      </div>
      <div className="px-5 py-4">
        <Blocks segments={segments} keyBase={keyBase} />
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────
// Public component
// ─────────────────────────────────────────────────────────────────────

/** Remove wrapper/JSX noise the agent may emit around the itinerary. */
function preclean(content: string): string {
  // Saved replies created before activity numbering became one-based contain
  // lines such as "[0] attraction". If a reply has that legacy marker, shift
  // every activity label in the same reply once; new [1]-based replies remain
  // untouched.
  const hasLegacyActivityIndex =
    /^\s*(?:[*\-•]\s*)?\[0\]\s+(?:attraction|restaurant|activity):/im.test(content);
  const indexedContent = hasLegacyActivityIndex
    ? content.replace(
        /^(\s*(?:[*\-•]\s*)?)\[(\d+)\](\s+(?:attraction|restaurant|activity):)/gim,
        (_line, prefix: string, rawIndex: string, suffix: string) =>
          `${prefix}[${Number(rawIndex) + 1}]${suffix}`,
      )
    : content;

  return indexedContent
    .replace(/\{\/\*[\s\S]*?\*\/\}/g, '') // {/* JSX comments */}
    .replace(/<\/?Sequence>/gi, '') // <Sequence> wrapper
    // Normalise [MAP: …] lines the agent wrapped in a bullet or bold so the
    // map parser sees them as stand-alone lines.
    .replace(/^\s*[*\-•]\s*(?=\[MAP:)/gim, '')
    .replace(/^\s*\*\*(\[MAP:[^\]]+\])\*\*\s*$/gim, '$1')
    .trim();
}

export default function MessageContent({ content }: { content: string }) {
  const cleaned = preclean(content);
  const steps = parseSteps(cleaned);

  if (steps) {
    // Any prose that appears before the first <Step> (an intro line).
    const intro = cleaned.split(/<Step\b/i)[0].trim();
    return (
      <div className="space-y-4">
        {intro && (
          <div className="text-sm leading-relaxed text-slate-700">
            <Blocks segments={segmentBody(intro)} keyBase="intro" />
          </div>
        )}
        {steps.map((step, i) => (
          <StepCard key={i} step={step} index={i} keyBase={`step${i}`} />
        ))}
      </div>
    );
  }

  // Plain answer → markdown pass.
  return <Blocks segments={segmentBody(cleaned)} keyBase="msg" />;
}
