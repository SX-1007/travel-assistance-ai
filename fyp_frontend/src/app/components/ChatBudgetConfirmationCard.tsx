import type { ChatBudgetConfirmation } from '../../lib/api';

const formatAmount = (amount: number, currency: string): string => {
  if (!Number.isFinite(amount)) return 'Unavailable';
  try {
    return new Intl.NumberFormat('en-MY', {
      style: 'currency',
      currency,
      currencyDisplay: 'narrowSymbol',
      maximumFractionDigits: 0,
    }).format(amount);
  } catch {
    return `${typeof currency === 'string' ? currency : 'Amount'} ${amount.toLocaleString('en-MY')}`;
  }
};

export default function ChatBudgetConfirmationCard({
  decision,
  disabled,
  onAccept,
}: {
  decision: ChatBudgetConfirmation;
  disabled: boolean;
  onAccept: () => void;
}) {
  const statedBudget = decision.stated_budget == null
    ? 'No budget entered'
    : formatAmount(decision.stated_budget, decision.base_currency);

  return (
    <section
      role="region"
      aria-label="Chat budget confirmation"
      className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-slate-800"
    >
      <h3 className="font-semibold">Budget confirmation required</h3>
      <p className="mt-1 text-sm text-slate-700">
        Your budget: <strong>{statedBudget}</strong>
      </p>
      <p className="mt-1 text-sm text-slate-700">
        Provider-grounded minimum: <strong>{formatAmount(
          decision.recommended_minimum_budget,
          decision.base_currency,
        )}</strong>
      </p>
      <dl className="mt-3 grid gap-1 text-xs text-slate-600 sm:grid-cols-2">
        <div><dt className="inline font-medium">Outbound flight: </dt><dd className="inline">{formatAmount(decision.evidence.outbound_flight_price, decision.destination_currency)}</dd></div>
        <div><dt className="inline font-medium">Return flight: </dt><dd className="inline">{formatAmount(decision.evidence.return_flight_price, decision.destination_currency)}</dd></div>
        <div><dt className="inline font-medium">Hotel: </dt><dd className="inline">{decision.evidence.hotel_nights === 0 ? 'No overnight stay required' : <>{formatAmount(decision.evidence.hotel_price_per_night, decision.destination_currency)} × {decision.evidence.hotel_nights} nights</>}</dd></div>
        <div><dt className="inline font-medium">Expires: </dt><dd className="inline">{Number.isFinite(Date.parse(decision.expires_at)) ? new Date(decision.expires_at).toLocaleString() : 'Unavailable'}</dd></div>
      </dl>
      <button
        type="button"
        onClick={onAccept}
        disabled={disabled}
        className="mt-4 rounded-lg bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
      >
        Use recommended budget
      </button>
    </section>
  );
}
