import { VercelRequest, VercelResponse } from '@vercel/node';
import { getStore } from './lib/storage';
import { getRateLimiter, getClientIP, applyRateLimitHeaders } from './lib/rate-limiter';

/**
 * GET /api/recall
 *
 * Retrieve memories matching a query
 *
 * Query parameters:
 * - q: string - Query text (required)
 * - limit: number - Max results to return (default: 10, max: 50)
 *
 * Response:
 * {
 *   "query": "string",
 *   "results": [
 *     {
 *       "id": "string",
 *       "content": "string",
 *       "relevance": "number 0.0-1.0",
 *       "confidence": "number 0.0-1.0",
 *       "source": "string",
 *       "age_seconds": "number"
 *     }
 *   ],
 *   "count": "number",
 *   "stats": {
 *     "total_memories": "number",
 *     "avg_age_seconds": "number"
 *   }
 * }
 */
export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Only allow GET
  if (req.method !== 'GET') {
    return res.status(405).json({
      error: 'Method not allowed',
      allowed_methods: ['GET']
    });
  }

  try {
    // Rate limiting
    const limiter = getRateLimiter();
    const clientIP = getClientIP(req);
    const limitResult = await limiter.checkLimit(clientIP);

    applyRateLimitHeaders(res, limitResult);

    if (!limitResult.allowed) {
      return res.status(429).json({
        error: 'Rate limit exceeded',
        limit: limitResult.limit,
        remaining: limitResult.remaining,
        reset: limitResult.reset,
        retry_after_seconds: limitResult.retryAfter
      });
    }

    // Get query parameters
    const query = req.query.q as string;
    const limitParam = parseInt((req.query.limit as string) || '10', 10);

    // Validate query
    if (!query || typeof query !== 'string') {
      return res.status(400).json({
        error: 'Missing or invalid required query parameter: q'
      });
    }

    if (query.length > 1000) {
      return res.status(400).json({
        error: 'Query too long (max 1000 characters)'
      });
    }

    // Validate limit
    const limit = Math.min(Math.max(1, limitParam), 50); // Clamp to 1-50

    // Retrieve memories
    const store = getStore();
    const memories = await store.recall(query, limit);
    const stats = store.getStats();

    // Format response
    const now = Date.now();
    const results = memories.map(mem => ({
      id: mem.id,
      content: mem.content,
      relevance: mem.relevance ? parseFloat(mem.relevance.toFixed(3)) : 0,
      confidence: mem.metadata.confidence ?? 0.8,
      source: mem.metadata.source || 'unknown',
      age_seconds: Math.floor((now - mem.metadata.timestamp) / 1000),
      timestamp: new Date(mem.metadata.timestamp).toISOString()
    }));

    return res.status(200).json({
      query,
      results,
      count: results.length,
      stats: {
        total_memories: stats.total_memories,
        avg_age_seconds: stats.avg_age_seconds,
        oldest_memory_seconds: stats.oldest_memory_seconds
      }
    });
  } catch (error) {
    console.error('Recall endpoint error:', error);

    return res.status(500).json({
      error: 'Internal server error',
      message:
        error instanceof Error ? error.message : 'Unknown error occurred'
    });
  }
}
