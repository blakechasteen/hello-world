import { VercelRequest, VercelResponse } from '@vercel/node';
import { getStore } from './lib/storage';
import { getRateLimiter, getClientIP, applyRateLimitHeaders } from './lib/rate-limiter';

/**
 * POST /api/experience
 *
 * Store a new memory/experience in HoloLoom
 *
 * Request body:
 * {
 *   "content": "string - The memory content",
 *   "source": "optional string - Source (e.g., user, document, chat)",
 *   "confidence": "optional number 0.0-1.0 - Confidence in this memory",
 *   "metadata": "optional object - Additional metadata"
 * }
 *
 * Response:
 * {
 *   "id": "string - Memory ID",
 *   "cached": "boolean - Whether persisted to KV",
 *   "timestamp": "ISO string",
 *   "stats": {
 *     "total_memories": "number",
 *     "memory_usage_mb": "number"
 *   }
 * }
 */
export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({
      error: 'Method not allowed',
      allowed_methods: ['POST']
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

    // Validate request body
    const { content, source, confidence, metadata } = req.body;

    if (!content || typeof content !== 'string') {
      return res.status(400).json({
        error: 'Missing or invalid required field: content'
      });
    }

    if (content.length > 10000) {
      return res.status(400).json({
        error: 'Content too large (max 10000 characters)'
      });
    }

    if (confidence !== undefined) {
      if (typeof confidence !== 'number' || confidence < 0 || confidence > 1) {
        return res.status(400).json({
          error: 'Invalid confidence: must be number between 0 and 1'
        });
      }
    }

    // Generate memory ID
    const id = `mem_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // Store memory
    const store = getStore();
    const memory = {
      id,
      content,
      metadata: {
        timestamp: Date.now(),
        source: source || 'api',
        confidence: confidence ?? 0.8,
        ...(metadata && typeof metadata === 'object' ? metadata : {})
      }
    };

    const storeResult = await store.store(memory);
    const stats = store.getStats();

    return res.status(201).json({
      id,
      cached: storeResult.cached,
      timestamp: new Date(memory.metadata.timestamp).toISOString(),
      content_length: content.length,
      stats: {
        total_memories: stats.total_memories,
        memory_usage_mb: stats.memory_usage_mb,
        oldest_memory_seconds: stats.oldest_memory_seconds
      }
    });
  } catch (error) {
    console.error('Experience endpoint error:', error);

    return res.status(500).json({
      error: 'Internal server error',
      message:
        error instanceof Error ? error.message : 'Unknown error occurred'
    });
  }
}
