import { VercelRequest, VercelResponse } from '@vercel/node';

/**
 * Health check endpoint
 * GET /api/health
 *
 * Returns:
 * - 200: Service is healthy
 * - 503: Service is degraded
 */
export default function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const uptime = process.uptime();
  const memory = process.memoryUsage();

  return res.status(200).json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime_seconds: Math.floor(uptime),
    memory: {
      heap_used_mb: Math.round(memory.heapUsed / 1024 / 1024),
      heap_total_mb: Math.round(memory.heapTotal / 1024 / 1024),
      external_mb: Math.round(memory.external / 1024 / 1024)
    },
    config: {
      memory_backend: process.env.HOLOLOOM_MEMORY_BACKEND || 'INMEMORY',
      max_memories: parseInt(process.env.HOLOLOOM_MAX_MEMORIES || '1000'),
      cache_ttl_seconds: parseInt(process.env.HOLOLOOM_CACHE_TTL || '3600')
    },
    api_endpoints: {
      experience: 'POST /api/experience',
      recall: 'GET /api/recall',
      query: 'POST /api/query',
      stats: 'GET /api/stats'
    }
  });
}
