/**
 * Production HTTP Server for HoloLoom 110/100 Emotional Intelligence
 *
 * Wraps complete_emotional_pipeline.js as a production-ready HTTP API
 *
 * Endpoints:
 *   POST /analyze - Analyze emotion from multimodal input
 *   GET /health - Health check
 *   GET /metrics - Prometheus metrics
 *
 * Built: November 2025
 */

const http = require('http');
const { processEmotionalInput } = require('./complete_emotional_pipeline.js');

// Configuration from environment
const PORT = process.env.PORT || 3000;
const METRICS_PORT = process.env.METRICS_PORT || 9091;
const NODE_ENV = process.env.NODE_ENV || 'production';

// Metrics tracking
const metrics = {
    totalRequests: 0,
    successfulRequests: 0,
    failedRequests: 0,
    totalProcessingTimeMs: 0,
    requestsByEmotion: {},
    requestsByMode: {},
    startTime: Date.now()
};

/**
 * Parse JSON from request body
 */
function parseBody(req) {
    return new Promise((resolve, reject) => {
        let body = '';
        req.on('data', chunk => {
            body += chunk.toString();
        });
        req.on('end', () => {
            try {
                resolve(body ? JSON.parse(body) : {});
            } catch (err) {
                reject(new Error('Invalid JSON'));
            }
        });
        req.on('error', reject);
    });
}

/**
 * Handle /analyze endpoint
 */
async function handleAnalyze(req, res) {
    const startTime = Date.now();
    metrics.totalRequests++;

    try {
        // Parse request body
        const input = await parseBody(req);

        // Validate input
        if (!input.text && !input.video && !input.audio) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({
                error: 'Missing required input (text, video, or audio)'
            }));
            metrics.failedRequests++;
            return;
        }

        // Process through emotional pipeline
        const result = await processEmotionalInput(input);

        // Track metrics
        const processingTime = Date.now() - startTime;
        metrics.successfulRequests++;
        metrics.totalProcessingTimeMs += processingTime;

        if (result.fusedEmotion?.emotion) {
            metrics.requestsByEmotion[result.fusedEmotion.emotion] =
                (metrics.requestsByEmotion[result.fusedEmotion.emotion] || 0) + 1;
        }

        const mode = input.context?.meta_mode || 'standard';
        metrics.requestsByMode[mode] = (metrics.requestsByMode[mode] || 0) + 1;

        // Add processing time to result
        result.processingTime = processingTime;

        // Send response
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));

    } catch (err) {
        console.error('Error processing emotion:', err);
        metrics.failedRequests++;

        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
            error: 'Internal server error',
            message: NODE_ENV === 'development' ? err.message : undefined
        }));
    }
}

/**
 * Handle /health endpoint
 */
function handleHealth(req, res) {
    const uptime = Date.now() - metrics.startTime;
    const health = {
        status: 'healthy',
        uptime: uptime,
        uptimeHuman: `${Math.floor(uptime / 1000 / 60)} minutes`,
        timestamp: new Date().toISOString()
    };

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(health));
}

/**
 * Handle /metrics endpoint (Prometheus format)
 */
function handleMetrics(req, res) {
    const uptime = (Date.now() - metrics.startTime) / 1000;
    const avgProcessingTime = metrics.totalRequests > 0
        ? metrics.totalProcessingTimeMs / metrics.totalRequests
        : 0;

    const lines = [
        '# HELP emotion_requests_total Total number of emotion analysis requests',
        '# TYPE emotion_requests_total counter',
        `emotion_requests_total ${metrics.totalRequests}`,
        '',
        '# HELP emotion_requests_successful Number of successful requests',
        '# TYPE emotion_requests_successful counter',
        `emotion_requests_successful ${metrics.successfulRequests}`,
        '',
        '# HELP emotion_requests_failed Number of failed requests',
        '# TYPE emotion_requests_failed counter',
        `emotion_requests_failed ${metrics.failedRequests}`,
        '',
        '# HELP emotion_processing_time_ms Average processing time in milliseconds',
        '# TYPE emotion_processing_time_ms gauge',
        `emotion_processing_time_ms ${avgProcessingTime}`,
        '',
        '# HELP emotion_uptime_seconds Server uptime in seconds',
        '# TYPE emotion_uptime_seconds gauge',
        `emotion_uptime_seconds ${uptime}`,
        ''
    ];

    // Per-emotion metrics
    if (Object.keys(metrics.requestsByEmotion).length > 0) {
        lines.push('# HELP emotion_by_type Number of requests by detected emotion');
        lines.push('# TYPE emotion_by_type counter');
        for (const [emotion, count] of Object.entries(metrics.requestsByEmotion)) {
            lines.push(`emotion_by_type{emotion="${emotion}"} ${count}`);
        }
        lines.push('');
    }

    // Per-mode metrics
    if (Object.keys(metrics.requestsByMode).length > 0) {
        lines.push('# HELP emotion_by_mode Number of requests by processing mode');
        lines.push('# TYPE emotion_by_mode counter');
        for (const [mode, count] of Object.entries(metrics.requestsByMode)) {
            lines.push(`emotion_by_mode{mode="${mode}"} ${count}`);
        }
        lines.push('');
    }

    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end(lines.join('\n'));
}

/**
 * Main HTTP server
 */
const server = http.createServer(async (req, res) => {
    // CORS headers for development
    if (NODE_ENV === 'development') {
        res.setHeader('Access-Control-Allow-Origin', '*');
        res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
        res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

        if (req.method === 'OPTIONS') {
            res.writeHead(200);
            res.end();
            return;
        }
    }

    // Route requests
    if (req.method === 'POST' && req.url === '/analyze') {
        await handleAnalyze(req, res);
    } else if (req.method === 'GET' && req.url === '/health') {
        handleHealth(req, res);
    } else if (req.method === 'GET' && req.url === '/metrics') {
        handleMetrics(req, res);
    } else {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Not found' }));
    }
});

/**
 * Metrics server (Prometheus)
 */
const metricsServer = http.createServer((req, res) => {
    if (req.method === 'GET' && req.url === '/metrics') {
        handleMetrics(req, res);
    } else {
        res.writeHead(404);
        res.end();
    }
});

// Start servers
server.listen(PORT, () => {
    console.log(`🧠 Emotion Intelligence API listening on port ${PORT}`);
    console.log(`   POST http://localhost:${PORT}/analyze - Analyze emotion`);
    console.log(`   GET  http://localhost:${PORT}/health - Health check`);
});

metricsServer.listen(METRICS_PORT, () => {
    console.log(`📊 Prometheus metrics on port ${METRICS_PORT}`);
    console.log(`   GET  http://localhost:${METRICS_PORT}/metrics`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('SIGTERM received, shutting down gracefully...');
    server.close(() => {
        metricsServer.close(() => {
            console.log('Servers closed');
            process.exit(0);
        });
    });
});

process.on('SIGINT', () => {
    console.log('SIGINT received, shutting down gracefully...');
    server.close(() => {
        metricsServer.close(() => {
            console.log('Servers closed');
            process.exit(0);
        });
    });
});
