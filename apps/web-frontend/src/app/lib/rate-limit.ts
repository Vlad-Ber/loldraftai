import { kv } from "@vercel/kv";
import { headers } from "next/headers";

interface RateLimitConfig {
  windowMs: number;
  maxRequests: number;
}

export class RateLimit {
  private static readonly DEFAULT_CONFIG: RateLimitConfig = {
    windowMs: 60 * 1000, // 1 minute window
    maxRequests: 60, // Max requests per window
  };

  private static readonly ELEVATED_CONFIG: RateLimitConfig = {
    windowMs: 60 * 1000, // 1 minute window
    maxRequests: 1000, // Higher rate limit (1000 requests per minute)
  };

  private static readonly ELEVATED_ACCESS_KEY =
    process.env.ELEVATED_RATE_LIMIT_KEY;

  static async checkRateLimit(): Promise<boolean> {
    const headersList = await headers();
    const ip = headersList.get("x-forwarded-for") || "unknown";
    const elevatedAccessHeader = headersList.get("x-rate-limit-key");

    const now = Date.now();
    const key = `ratelimit:${ip}`;

    // Determine which config to use based on the header
    const config =
      this.ELEVATED_ACCESS_KEY &&
      elevatedAccessHeader === this.ELEVATED_ACCESS_KEY
        ? this.ELEVATED_CONFIG
        : this.DEFAULT_CONFIG;

    try {
      // Get the current requests array or create new one
      const requests = (await kv.get<number[]>(key)) || [];

      // Filter out old requests
      const windowStart = now - config.windowMs;
      const recentRequests = requests.filter((time) => time > windowStart);

      // Check if rate limit is exceeded
      if (recentRequests.length >= config.maxRequests) {
        // Check if this IP's rate limit violation has been logged before
        const logKey = `ratelimit-logged:${ip}`;
        const hasLogged = await kv.get(logKey);

        if (!hasLogged) {
          console.warn(`Rate limit exceeded for IP: ${ip}`);
          // Mark this IP as logged with 24h expiration
          await kv.set(logKey, true, { ex: 24 * 60 * 60 });
        }
        return false;
      }

      // Add current request and update KV store
      recentRequests.push(now);

      // Calculate remaining time in current window and set expiration
      const remainingMs = config.windowMs - (now % config.windowMs);
      await kv.set(key, recentRequests, { ex: Math.ceil(remainingMs / 1000) });

      return true;
    } catch (error) {
      console.error("Rate limit error:", error);
      return true; // Fail open if KV store is unavailable
    }
  }
}
