import { VercelRequest, VercelResponse } from '@vercel/node';
import { getStore } from './lib/storage';
import { getRateLimiter, getClientIP, applyRateLimitHeaders } from './lib/rate-limiter';

/**
 * POST /api/query
 *
 * Execute a full query cycle: experience (store) + recall (retrieve)
 * For compound operations like "store this and find related memories"
 *
 * Request body:
 * {
 *   "query": "string - The query text",
 *   "experience": "optional object - If provided, store this first",
 *   "experience.content": "string - Memory content to store",
 *   "experience.source": "optional string",
 *   "experience.confidence": "optional number 0.0-1.0",
 *   "recall_limit": "optional number - Max results (default: 10, max: 50)"
 * }
 *
 * Response:
 * {
 *   "query": "string",
 *   "experience_stored": {
 *     "id": "string",
 *     "cached": "boolean"
 *   } | null,
 *   "recalled_memories": [
 *     {
 *       "id": "string",
 *       "content": "string",
 *       "relevance": "number",
 *       "confidence": "number",
 *       "source": "string",
 *       "age_seconds": "number"
 *     }
 *   ],
 *   "count": "number",
 *   "processing_ms": "number",
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

  const startTime = Date.now();

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

    // Validate request
    const { query, experience, recall_limit } = req.body;

    if (!query || typeof query !== 'string') {
      return res.status(400).json({
        error: 'Missing or invalid required field: query'
      });
    }

    if (query.length > 1000) {
      return res.status(400).json({
        error: 'Query too long (max 1000 characters)'
      });
    }

    const store = getStore();
    let experienceStored = null;

    // Step 1: Store experience if provided
    if (experience && typeof experience === 'object') {
      const { content, source, confidence, metadata } = experience;

      if (!content || typeof content !== 'string') {
        return res.status(400).json({
          error: 'Missing or invalid field: experience.content'
        });
      }

      if (content.length > 10000) {
        return res.status(400).json({
          error: 'Experience content too large (max 10000 characters)'
        });
      }

      const memId = `mem_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      const memory = {
        id: memId,
        content,
        metadata: {
          timestamp: Date.now(),
          source: source || 'api',
          confidence: confidence ?? 0.8,
          ...(metadata && typeof metadata === 'object' ? metadata : {})
        }
      };

      const storeResult = await store.store(memory);
      experienceStored = {
        id: memId,
        cached: storeResult.cached
      };
    }

    // Step 2: Recall memories matching query
    const limitParam = parseInt((recall_limit as any) || '10', 10);
    const limit = Math.min(Math.max(1, limitParam), 50);

    const memories = await store.recall(query, limit);
    const stats = store.getStats();

    // Format results
    const now = Date.now();
    const recalled_memories = memories.map(mem => ({
      id: mem.id,
      content: mem.content,
      relevance: mem.relevance ? parseFloat(mem.relevance.toFixed(3)) : 0,
      confidence: mem.metadata.confidence ?? 0.8,
      source: mem.metadata.source || 'unknown',
      age_seconds: Math.floor((now - mem.metadata.timestamp) / 1000),
      timestamp: new Date(mem.metadata.timestamp).toISOString()
    }));

    const processingMs = Date.now() - startTime;

    return res.status(200).json({
      query,
      experience_stored: experienceStored,
      recalled_memories,
      count: recalled_memories.length,
      processing_ms: processingMs,
      stats: {
        total_memories: stats.total_memories,
        memory_usage_mb: stats.memory_usage_mb,
        avg_age_seconds: stats.avg_age_seconds
      }
    });
  } catch (error) {
    console.error('Query endpoint error:', error);

    const processingMs = Date.now() - startTime;

    return res.status(500).json({
      error: 'Internal server error',
      message:
        error instanceof Error ? error.message : 'Unknown error occurred',
      processing_ms: processingMs
    });
  }
}
