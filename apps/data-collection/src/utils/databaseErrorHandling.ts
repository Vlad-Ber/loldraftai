import { sleep } from "./index";

const INITIAL_BACKOFF_MS = 1000; // Start with 1 second
const MAX_BACKOFF_MS = 60000; // Max 1 minute

// Define a more specific logger type
export type LoggerFunction = (
  level: "INFO" | "ERROR" | "DEBUG",
  message: string
) => void;

export class DatabaseBackoff {
  private currentBackoffMs = INITIAL_BACKOFF_MS;
  private consecutiveDbErrors = 0;

  async handleDatabaseError(error: any, logger: LoggerFunction = console.log) {
    if (error?.message?.includes("Can't reach database server")) {
      this.consecutiveDbErrors++;
      this.currentBackoffMs = Math.min(
        this.currentBackoffMs * 2,
        MAX_BACKOFF_MS
      );
      logger(
        "ERROR",
        `Database connection error. Backing off for ${
          this.currentBackoffMs / 1000
        } seconds. Consecutive errors: ${this.consecutiveDbErrors}`
      );
      await sleep(this.currentBackoffMs);
      return true;
    }
    return false;
  }

  resetBackoff(logger: LoggerFunction = console.log) {
    if (this.consecutiveDbErrors > 0) {
      logger("INFO", "Database connection restored. Resetting backoff.");
      this.consecutiveDbErrors = 0;
      this.currentBackoffMs = INITIAL_BACKOFF_MS;
    }
  }

  async withRetry<T>(
    operation: () => Promise<T>,
    logger: LoggerFunction = console.log
  ): Promise<T> {
    while (true) {
      try {
        const result = await operation();
        this.resetBackoff(logger);
        return result;
      } catch (error) {
        const isDbError = await this.handleDatabaseError(error, logger);
        if (!isDbError) {
          throw error;
        }
      }
    }
  }
}
