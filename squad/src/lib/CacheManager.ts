/**
 * CacheManager - Embedding cache with SQLite persistence
 *
 * Features:
 * - SQLite persistent storage
 * - LRU eviction policy
 * - Incremental updates (only re-embed changed code)
 * - File watcher integration for auto-invalidation
 * - Statistics tracking
 */

import * as vscode from 'vscode';
import Database from 'better-sqlite3';
import * as crypto from 'crypto';
import * as path from 'path';
import { CacheEntry, CacheStats } from '../types';

export class CacheManager {
    private db: Database.Database;
    private maxSize: number;
    private stats: CacheStats;
    private fileWatcher: vscode.FileSystemWatcher | null = null;

    constructor(cachePath: string, maxSize: number = 10000) {
        this.maxSize = maxSize;
        this.db = new Database(cachePath);
        this.stats = {
            totalEntries: 0,
            totalHits: 0,
            totalMisses: 0,
            hitRate: 0,
            size: 0,
            maxSize: maxSize
        };

        this.initializeDatabase();
        this.loadStats();
        this.setupFileWatcher();
    }

    /**
     * Initialize SQLite database schema
     */
    private initializeDatabase() {
        this.db.exec(`
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                hits INTEGER DEFAULT 0,
                size INTEGER NOT NULL,
                file_path TEXT,
                content_hash TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_timestamp ON cache(timestamp);
            CREATE INDEX IF NOT EXISTS idx_file_path ON cache(file_path);
            CREATE INDEX IF NOT EXISTS idx_hits ON cache(hits);

            CREATE TABLE IF NOT EXISTS stats (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                total_hits INTEGER DEFAULT 0,
                total_misses INTEGER DEFAULT 0,
                last_updated INTEGER
            );

            INSERT OR IGNORE INTO stats (id, total_hits, total_misses, last_updated)
            VALUES (1, 0, 0, ${Date.now()});
        `);
    }

    /**
     * Load statistics from database
     */
    private loadStats() {
        const statsRow = this.db.prepare('SELECT * FROM stats WHERE id = 1').get() as any;
        const countRow = this.db.prepare('SELECT COUNT(*) as count, SUM(size) as total_size FROM cache').get() as any;

        this.stats = {
            totalEntries: countRow.count || 0,
            totalHits: statsRow?.total_hits || 0,
            totalMisses: statsRow?.total_misses || 0,
            hitRate: this.calculateHitRate(statsRow?.total_hits || 0, statsRow?.total_misses || 0),
            size: countRow.total_size || 0,
            maxSize: this.maxSize
        };
    }

    /**
     * Setup file watcher for cache invalidation
     */
    private setupFileWatcher() {
        // Watch for file changes in workspace
        this.fileWatcher = vscode.workspace.createFileSystemWatcher('**/*');

        this.fileWatcher.onDidChange(uri => {
            this.invalidateFile(uri.fsPath);
        });

        this.fileWatcher.onDidDelete(uri => {
            this.invalidateFile(uri.fsPath);
        });
    }

    /**
     * Get item from cache
     */
    get(key: string): any | null {
        const stmt = this.db.prepare('SELECT value, hits FROM cache WHERE key = ?');
        const row = stmt.get(key) as any;

        if (row) {
            // Update hit count
            this.db.prepare('UPDATE cache SET hits = hits + 1 WHERE key = ?').run(key);
            this.updateStats({ hits: 1 });

            try {
                return JSON.parse(row.value);
            } catch (error) {
                console.error('Failed to parse cached value:', error);
                return null;
            }
        }

        this.updateStats({ misses: 1 });
        return null;
    }

    /**
     * Set item in cache
     */
    set(key: string, value: any, filePath?: string) {
        const valueStr = JSON.stringify(value);
        const size = Buffer.byteLength(valueStr, 'utf8');
        const timestamp = Date.now();
        const contentHash = filePath ? this.hashFile(filePath) : null;

        // Check if we need to evict entries
        if (this.stats.totalEntries >= this.maxSize) {
            this.evictLRU();
        }

        // Insert or replace
        const stmt = this.db.prepare(`
            INSERT OR REPLACE INTO cache (key, value, timestamp, hits, size, file_path, content_hash)
            VALUES (?, ?, ?, 0, ?, ?, ?)
        `);

        stmt.run(key, valueStr, timestamp, size, filePath || null, contentHash || null);

        // Update stats
        this.stats.totalEntries = (this.db.prepare('SELECT COUNT(*) as count FROM cache').get() as any).count;
        this.stats.size += size;
    }

    /**
     * Check if cache needs update for file
     * Returns true if file has changed since last cache
     */
    needsUpdate(filePath: string): boolean {
        const currentHash = this.hashFile(filePath);
        const stmt = this.db.prepare('SELECT content_hash FROM cache WHERE file_path = ? LIMIT 1');
        const row = stmt.get(filePath) as any;

        if (!row) {
            return true; // Not in cache
        }

        return row.content_hash !== currentHash; // Hash changed
    }

    /**
     * Invalidate all cache entries for a file
     */
    invalidateFile(filePath: string) {
        const stmt = this.db.prepare('DELETE FROM cache WHERE file_path = ?');
        const result = stmt.run(filePath);

        if (result.changes > 0) {
            console.log(`Invalidated ${result.changes} cache entries for ${filePath}`);
            this.loadStats(); // Reload stats
        }
    }

    /**
     * Hash file content for change detection
     */
    private hashFile(filePath: string): string {
        try {
            const fs = require('fs');
            const content = fs.readFileSync(filePath, 'utf8');
            return crypto.createHash('sha256').update(content).digest('hex');
        } catch (error) {
            console.error('Failed to hash file:', error);
            return '';
        }
    }

    /**
     * Evict least recently used entries
     */
    private evictLRU() {
        // Remove 10% of entries with lowest hits and oldest timestamps
        const evictCount = Math.floor(this.maxSize * 0.1);

        this.db.prepare(`
            DELETE FROM cache
            WHERE key IN (
                SELECT key FROM cache
                ORDER BY hits ASC, timestamp ASC
                LIMIT ?
            )
        `).run(evictCount);

        console.log(`Evicted ${evictCount} LRU cache entries`);
        this.loadStats();
    }

    /**
     * Update statistics
     */
    private updateStats(update: { hits?: number; misses?: number }) {
        if (update.hits) {
            this.stats.totalHits += update.hits;
        }
        if (update.misses) {
            this.stats.totalMisses += update.misses;
        }

        this.stats.hitRate = this.calculateHitRate(this.stats.totalHits, this.stats.totalMisses);

        // Persist stats every 10 operations
        if ((this.stats.totalHits + this.stats.totalMisses) % 10 === 0) {
            this.db.prepare(`
                UPDATE stats
                SET total_hits = ?, total_misses = ?, last_updated = ?
                WHERE id = 1
            `).run(this.stats.totalHits, this.stats.totalMisses, Date.now());
        }
    }

    /**
     * Calculate hit rate
     */
    private calculateHitRate(hits: number, misses: number): number {
        const total = hits + misses;
        return total === 0 ? 0 : Math.round((hits / total) * 100 * 100) / 100;
    }

    /**
     * Get cache statistics
     */
    getStats(): CacheStats {
        return { ...this.stats };
    }

    /**
     * Clear all cache entries
     */
    clear() {
        this.db.prepare('DELETE FROM cache').run();
        this.stats.totalEntries = 0;
        this.stats.size = 0;
        console.log('Cache cleared');
    }

    /**
     * Clear cache entries older than specified days
     */
    clearOlderThan(days: number) {
        const cutoff = Date.now() - (days * 24 * 60 * 60 * 1000);
        const stmt = this.db.prepare('DELETE FROM cache WHERE timestamp < ?');
        const result = stmt.run(cutoff);

        console.log(`Cleared ${result.changes} cache entries older than ${days} days`);
        this.loadStats();
    }

    /**
     * Get cache entries for debugging
     */
    getEntries(limit: number = 100): CacheEntry[] {
        const stmt = this.db.prepare(`
            SELECT key, value, timestamp, hits, size
            FROM cache
            ORDER BY timestamp DESC
            LIMIT ?
        `);

        const rows = stmt.all(limit) as any[];

        return rows.map(row => ({
            key: row.key,
            value: JSON.parse(row.value),
            timestamp: row.timestamp,
            hits: row.hits,
            size: row.size
        }));
    }

    /**
     * Cleanup and close database
     */
    dispose() {
        if (this.fileWatcher) {
            this.fileWatcher.dispose();
        }

        // Final stats update
        this.db.prepare(`
            UPDATE stats
            SET total_hits = ?, total_misses = ?, last_updated = ?
            WHERE id = 1
        `).run(this.stats.totalHits, this.stats.totalMisses, Date.now());

        this.db.close();
    }
}
