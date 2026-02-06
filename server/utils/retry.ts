/**
 * API Retry Utility with Exponential Backoff
 *
 * Provides resilient API calling with automatic retry on failures.
 * Used to prevent single network blips from causing scan cycle failures.
 */

export interface RetryOptions {
  maxRetries?: number;
  baseDelay?: number;
  maxDelay?: number;
  onRetry?: (error: Error, attempt: number) => void;
}

/**
 * Sleep utility for retry delays
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Executes an async function with exponential backoff retry logic
 *
 * @param fn - The async function to execute
 * @param options - Retry configuration options
 * @returns The result of the async function
 * @throws The last error if all retries fail
 *
 * @example
 * const data = await withRetry(
 *   () => alpaca.getClock(),
 *   {
 *     maxRetries: 3,
 *     baseDelay: 1000,
 *     onRetry: (err, attempt) => console.log(`Retry ${attempt}: ${err.message}`)
 *   }
 * );
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  options: RetryOptions = {}
): Promise<T> {
  const {
    maxRetries = 3,
    baseDelay = 1000,
    maxDelay = 10000,
    onRetry,
  } = options;

  let lastError: Error;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));

      // Don't retry on last attempt
      if (attempt === maxRetries - 1) {
        break;
      }

      // Calculate exponential backoff delay: baseDelay * 2^attempt
      const delay = Math.min(baseDelay * Math.pow(2, attempt), maxDelay);

      // Log retry attempt
      if (onRetry) {
        onRetry(lastError, attempt + 1);
      } else {
        console.log(
          `[Retry] Attempt ${attempt + 1}/${maxRetries} failed: ${lastError.message}. Retrying in ${delay}ms...`
        );
      }

      // Wait before retrying
      await sleep(delay);
    }
  }

  // All retries exhausted
  console.error(
    `[Retry] All ${maxRetries} attempts failed. Last error: ${lastError!.message}`
  );
  throw lastError!;
}

/**
 * Creates a wrapped version of an async function with automatic retry logic
 *
 * @param fn - The async function to wrap
 * @param options - Retry configuration options
 * @returns A new function with retry logic built-in
 *
 * @example
 * const resilientGetClock = withRetryWrapped(
 *   () => alpaca.getClock(),
 *   { maxRetries: 3, baseDelay: 1000 }
 * );
 * const clock = await resilientGetClock();
 */
export function withRetryWrapped<T extends any[], R>(
  fn: (...args: T) => Promise<R>,
  options: RetryOptions = {}
): (...args: T) => Promise<R> {
  return (...args: T) => withRetry(() => fn(...args), options);
}

/**
 * Retry-specific error class for better error handling
 */
export class RetryExhaustedError extends Error {
  constructor(
    message: string,
    public readonly attempts: number,
    public readonly lastError: Error
  ) {
    super(message);
    this.name = "RetryExhaustedError";
  }
}
