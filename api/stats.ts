import { VercelRequest, VercelResponse } from '@vercel/node';
import { getStore } from './lib/storage';
import { getRateLimiter } from './lib/rate-limiter';

/**
 * GET /api/stats
 *
 * Get server and storage statistics
 *
 * Response:
 * {
 *   "uptime_seconds": "number",
 *   "memory": {
 *     "heap_used_mb": "number",
 *     "heap_total_mb": "number",
 *     "external_mb": "number"
 *   },
 *   "storage": {
 *     "total_memories": "number",
 *     "memory_usage_mb": "number",
 *     "avg_age_seconds": "number",
 *     "oldest_memory_seconds": "number"
 *   },
 *   "rate_limiting": {
 *     "tracked_identifiers": "number",
 *     "total_requests_tracked": "number"
 *   }
 * }
 */
export default function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const uptime = process.uptime();
  const memory = process.memoryUsage();
  const store = getStore();
  const storageStats = store.getStats();
  const limiter = getRateLimiter();
  const rateLimitStats = limiter.getStats();

  return res.status(200).json({
    timestamp: new Date().toISOString(),
    uptime_seconds: Math.floor(uptime),
    memory: {
      heap_used_mb: Math.round(memory.heapUsed / 1024 / 1024),
      heap_total_mb: Math.round(memory.heapTotal / 1024 / 1024),
      external_mb: Math.round(memory.external / 1024 / 1024),
      rss_mb: Math.round(memory.rss / 1024 / 1024)
    },
    storage: {
      total_memories: storageStats.total_memories,
      memory_usage_mb: storageStats.memory_usage_mb,
      avg_age_seconds: storageStats.avg_age_seconds,
      oldest_memory_seconds: storageStats.oldest_memory_seconds
    },
    rate_limiting: {
      tracked_identifiers: rateLimitStats.tracked_identifiers,
      total_requests_tracked: rateLimitStats.total_requests_tracked
    }
  });
}
