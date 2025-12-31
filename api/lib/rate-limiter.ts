/**
 * Rate limiter for HoloLoom API
 * Supports both in-memory and Vercel KV backends
 */

export interface RateLimitConfig {
  maxRequests: number;
  windowMs: number; // milliseconds
}

export interface RateLimitResult {
  allowed: boolean;
  limit: number;
  remaining: number;
  reset: number; // Unix timestamp in seconds
  retryAfter?: number; // seconds
}

/**
 * In-memory rate limiter with sliding window
 * Per-IP tracking for serverless environment
 */
export class RateLimiter {
  private requests: Map<string, number[]> = new Map();
  private config: RateLimitConfig;
  private kvClient?: any;

  constructor(config: RateLimitConfig, kvClient?: any) {
    this.config = config;
    this.kvClient = kvClient;
  }

  /**
   * Check rate limit for an IP/identifier
   */
  async checkLimit(identifier: string): Promise<RateLimitResult> {
    const now = Date.now();
    const windowStart = now - this.config.windowMs;

    // Get requests from memory
    let requests = this.requests.get(identifier) || [];

    // Try to get from KV if available (for cross-container persistence)
    if (this.kvClient) {
      try {
        const kvData = await this.kvClient.get(`ratelimit:${identifier}`);
        if (kvData) {
          const parsed = JSON.parse(kvData);
          requests = parsed.timestamps || requests;
        }
      } catch (err) {
        console.warn('KV rate limit fetch failed:', err);
      }
    }

    // Filter out old requests (sliding window)
    requests = requests.filter(timestamp => timestamp > windowStart);

    // Check if limit exceeded
    const allowed = requests.length < this.config.maxRequests;

    // Calculate reset time
    const oldestRequest = requests.length > 0 ? requests[0] : now;
    const resetTime = Math.ceil((oldestRequest + this.config.windowMs) / 1000);

    if (allowed) {
      // Add current request
      requests.push(now);

      // Store in memory
      this.requests.set(identifier, requests);

      // Try to persist to KV
      if (this.kvClient) {
        try {
          await this.kvClient.set(
            `ratelimit:${identifier}`,
            JSON.stringify({ timestamps: requests }),
            { ex: Math.ceil(this.config.windowMs / 1000) }
          );
        } catch (err) {
          console.warn('KV rate limit store failed:', err);
        }
      }
    }

    return {
      allowed,
      limit: this.config.maxRequests,
      remaining: Math.max(0, this.config.maxRequests - requests.length),
      reset: resetTime,
      retryAfter: allowed ? undefined : resetTime - Math.ceil(now / 1000)
    };
  }

  /**
   * Reset rate limit for an identifier
   */
  async reset(identifier: string): Promise<void> {
    this.requests.delete(identifier);
    if (this.kvClient) {
      try {
        await this.kvClient.del(`ratelimit:${identifier}`);
      } catch (err) {
        console.warn('KV rate limit reset failed:', err);
      }
    }
  }

  /**
   * Get current rate limit status (for monitoring)
   */
  getStats(): {
    tracked_identifiers: number;
    total_requests_tracked: number;
  } {
    let total = 0;
    for (const requests of this.requests.values()) {
      total += requests.length;
    }

    return {
      tracked_identifiers: this.requests.size,
      total_requests_tracked: total
    };
  }
}

// Global rate limiter instance
let globalLimiter: RateLimiter | null = null;

export function getRateLimiter(
  config?: RateLimitConfig,
  kvClient?: any
): RateLimiter {
  if (!globalLimiter) {
    const defaultConfig: RateLimitConfig = {
      maxRequests: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS || '100'),
      windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS || '60000')
    };
    globalLimiter = new RateLimiter(config || defaultConfig, kvClient);
  }
  return globalLimiter;
}

/**
 * Extract client IP from request
 */
export function getClientIP(req: any): string {
  const forwarded = req.headers['x-forwarded-for'];
  if (typeof forwarded === 'string') {
    return forwarded.split(',')[0].trim();
  }
  return req.headers['x-real-ip'] || req.socket?.remoteAddress || 'unknown';
}

/**
 * Apply rate limit headers to response
 */
export function applyRateLimitHeaders(
  res: any,
  result: RateLimitResult
): void {
  res.setHeader('X-RateLimit-Limit', result.limit.toString());
  res.setHeader('X-RateLimit-Remaining', result.remaining.toString());
  res.setHeader('X-RateLimit-Reset', result.reset.toString());

  if (!result.allowed && result.retryAfter) {
    res.setHeader('Retry-After', result.retryAfter.toString());
  }
}
