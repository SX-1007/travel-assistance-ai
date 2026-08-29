export interface GateProfileResponse {
  status: string;
  onboarded: boolean;
  profile: Record<string, unknown>;
}

export interface ProfileGateOptions {
  retryDelaysMs?: number[];
  missingConfirmationDelayMs?: number;
}

const sleep = (delayMs: number) =>
  new Promise<void>((resolve) => globalThis.setTimeout(resolve, delayMs));

/**
 * A single false profile read must never send an existing user to onboarding.
 * Confirm the absence with a second successful read before returning false.
 */
export async function loadProfileForGate(
  loadProfile: () => Promise<GateProfileResponse>,
  options: ProfileGateOptions = {},
): Promise<GateProfileResponse> {
  const retryDelaysMs = options.retryDelaysMs ?? [300, 900];

  const readWithRetry = async (): Promise<GateProfileResponse> => {
    for (let attempt = 0; ; attempt += 1) {
      try {
        return await loadProfile();
      } catch (error) {
        const status = (error as { status?: unknown } | null)?.status;
        const retryable = typeof status !== 'number' || status === 0 || status >= 500;
        if (!retryable || attempt >= retryDelaysMs.length) throw error;
        await sleep(retryDelaysMs[attempt]);
      }
    }
  };

  const first = await readWithRetry();
  if (first.onboarded) return first;

  await sleep(options.missingConfirmationDelayMs ?? 250);
  return readWithRetry();
}