import React, { useEffect, useRef } from 'react';
import { X } from 'lucide-react';
import type {
  BudgetInfo, DayItinerary, GeoJSONFeatureCollection,
} from '../../lib/api';
import { isCompleteItinerarySnapshot } from '../../lib/api';
import ItineraryView from './ItineraryView';

export interface ItinerarySnapshot {
  itinerary: DayItinerary[];
  maps?: Record<string, GeoJSONFeatureCollection> | null;
  budget?: BudgetInfo | null;
  destination_country_code?: string;
}

export interface LatestItinerarySnapshot extends ItinerarySnapshot {
  messageIndex: number;
}

interface ItineraryMessage {
  itinerary?: unknown;
  maps?: unknown;
  budget?: unknown;
  destination_country_code?: unknown;
}

export function findLatestItinerarySnapshot(
  messages: readonly ItineraryMessage[],
): LatestItinerarySnapshot | null {
  for (let messageIndex = messages.length - 1; messageIndex >= 0; messageIndex -= 1) {
    const message = messages[messageIndex];
    const candidate = {
      itinerary: message?.itinerary,
      maps: message?.maps,
      budget: message?.budget,
      ...(Object.prototype.hasOwnProperty.call(message ?? {}, 'destination_country_code')
        ? { destination_country_code: message?.destination_country_code }
        : {}),
    };
    if (!isCompleteItinerarySnapshot(candidate)) continue;
    return {
      messageIndex,
      itinerary: candidate.itinerary,
      maps: candidate.maps,
      budget: candidate.budget,
      ...(Object.prototype.hasOwnProperty.call(candidate, 'destination_country_code')
        ? { destination_country_code: candidate.destination_country_code }
        : {}),
    };
  }
  return null;
}

export default function ItineraryCanvas({
  snapshot,
  onClose,
  isMobileModal,
}: {
  snapshot: unknown;
  onClose: () => void;
  isMobileModal: boolean;
}) {
  const panelRef = useRef<HTMLElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (isMobileModal) closeButtonRef.current?.focus();
  }, [isMobileModal]);

  const snapshotIsSafe = isCompleteItinerarySnapshot(snapshot);

  const handleKeyDown = (event: React.KeyboardEvent<HTMLElement>) => {
    if (!isMobileModal) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      onClose();
      return;
    }
    if (event.key !== 'Tab') return;

    const focusable = Array.from(panelRef.current?.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), ' +
      'textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    ) ?? []);
    if (focusable.length === 0) return;

    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  if (!snapshotIsSafe) return null;

  return (
    <aside
      ref={panelRef}
      role={isMobileModal ? 'dialog' : undefined}
      aria-label="Latest itinerary"
      aria-modal={isMobileModal ? true : undefined}
      onKeyDown={handleKeyDown}
      className="fixed inset-0 z-40 flex min-h-0 min-w-0 flex-col bg-white lg:static lg:z-auto lg:w-[58%] lg:flex-none lg:border-l lg:border-slate-200"
    >
      <div className="flex h-14 shrink-0 items-center justify-between gap-3 border-b border-slate-200 px-4 lg:px-6">
        <div>
          <h2 className="text-sm font-bold text-slate-800">Latest itinerary</h2>
          <p className="text-xs text-slate-500">Updates automatically as you refine your trip.</p>
        </div>
        <button
          ref={closeButtonRef}
          type="button"
          onClick={onClose}
          aria-label="Close itinerary"
          className="rounded-full p-2 text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-800"
        >
          <X size={18} />
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto bg-slate-50/60 p-4 lg:p-6">
        <div className="mx-auto max-w-4xl">
          <ItineraryView
            itinerary={snapshot.itinerary}
            maps={snapshot.maps}
            budget={snapshot.budget}
            {...(Object.prototype.hasOwnProperty.call(snapshot, 'destination_country_code')
              ? { destinationCountryCode: snapshot.destination_country_code }
              : {})}
          />
        </div>
      </div>
    </aside>
  );
}
