import Bottleneck from "bottleneck";

// API Key Rate Limits
export const API_RATE_LIMITS = {
  DEVELOPMENT: {
    REQUESTS_PER_SECOND: 20,
    REQUESTS_PER_2_MINUTES: 100,
  },
  PRODUCTION: {
    REQUESTS_PER_SECOND: 500,
    REQUESTS_PER_10_MINUTES: 30000,
  },
} as const;

// Determine which API key type we're using
// You can set this environment variable to "PRODUCTION" if using a production API key
const API_KEY_TYPE = (process.env.API_KEY_TYPE || "DEVELOPMENT") as keyof typeof API_RATE_LIMITS;
const RATE_LIMITS = API_RATE_LIMITS[API_KEY_TYPE];

// Calculate safe limits for multiple services
// We have 7 total services running concurrently
const TOTAL_SERVICES = 7;
const SAFETY_MARGIN = 0.8; // Use 80% of available capacity to be safe

// Calculate per-service limits
const REQUESTS_PER_SECOND_PER_SERVICE = Math.floor(
  (RATE_LIMITS.REQUESTS_PER_SECOND * SAFETY_MARGIN) / TOTAL_SERVICES
);

const REQUESTS_PER_2_MINUTES_PER_SERVICE = Math.floor(
  (RATE_LIMITS.REQUESTS_PER_2_MINUTES * SAFETY_MARGIN) / TOTAL_SERVICES
);

console.log(`Using ${API_KEY_TYPE} API key limits:`);
console.log(`- Total services: ${TOTAL_SERVICES}`);
console.log(`- Per-service requests/second: ${REQUESTS_PER_SECOND_PER_SERVICE}`);
console.log(`- Per-service requests/2-minutes: ${REQUESTS_PER_2_MINUTES_PER_SERVICE}`);

// Rate limiter configurations for different service types
export const RATE_LIMITER_CONFIGS = {
  // For services that make frequent API calls (collectMatchIds, processMatches)
  HIGH_FREQUENCY: {
    minTime: Math.ceil(1000 / REQUESTS_PER_SECOND_PER_SERVICE), // Convert to milliseconds
    reservoir: Math.floor(REQUESTS_PER_2_MINUTES_PER_SERVICE * 0.6), // Use 60% of 2-minute limit
    reservoirRefreshAmount: Math.floor(REQUESTS_PER_2_MINUTES_PER_SERVICE * 0.6),
    reservoirRefreshInterval: 2 * 60 * 1000, // 2 minutes
    maxConcurrent: 2,
  },
  
  // For services that make less frequent API calls (updateLadder)
  LOW_FREQUENCY: {
    minTime: Math.ceil(1000 / (REQUESTS_PER_SECOND_PER_SERVICE * 0.5)), // Even slower
    reservoir: Math.floor(REQUESTS_PER_2_MINUTES_PER_SERVICE * 0.4), // Use 40% of 2-minute limit
    reservoirRefreshAmount: Math.floor(REQUESTS_PER_2_MINUTES_PER_SERVICE * 0.4),
    reservoirRefreshInterval: 2 * 60 * 1000, // 2 minutes
    maxConcurrent: 1,
  },
  
  // Database rate limiter (separate from API rate limits)
  DATABASE: {
    minTime: 100,
    maxConcurrent: 5,
  },
} as const;

// Helper function to create rate limiters
export function createRateLimiter(type: keyof typeof RATE_LIMITER_CONFIGS): Bottleneck {
  return new Bottleneck(RATE_LIMITER_CONFIGS[type]);
}

// Helper function to get current rate limit info
export function getRateLimitInfo() {
  return {
    apiKeyType: API_KEY_TYPE,
    totalServices: TOTAL_SERVICES,
    requestsPerSecondPerService: REQUESTS_PER_SECOND_PER_SERVICE,
    requestsPer2MinutesPerService: REQUESTS_PER_2_MINUTES_PER_SERVICE,
    safetyMargin: SAFETY_MARGIN,
  };
} 