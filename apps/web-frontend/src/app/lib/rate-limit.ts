import { kv } from "@vercel/kv";

export class RateLimit {
  private static readonly WINDOW_MS = 60 * 1000; // 1 minute window
  private static readonly MAX_REQUESTS = 60; // Max requests per window

  static async checkRateLimit(ip: string): Promise<boolean> {
    const now = Date.now();
    const key = `ratelimit:${ip}`;

    try {
      // Get the current requests array or create new one
      const requests = (await kv.get<number[]>(key)) || [];

      // Filter out old requests
      const windowStart = now - this.WINDOW_MS;
      const recentRequests = requests.filter((time) => time > windowStart);

      // Check if rate limit is exceeded
      if (recentRequests.length >= this.MAX_REQUESTS) {
        return false;
      }

      // Add current request and update KV store
      recentRequests.push(now);
      await kv.set(key, recentRequests, { ex: 60 }); // Expire after 60 seconds

      return true;
    } catch (error) {
      console.error("Rate limit error:", error);
      return true; // Fail open if KV store is unavailable
    }
  }
}
