// engine/retrieval/neo4jAdapter.ts
// Minimal Neo4j adapter for junction-first retrieval.
// Requires neo4j-driver. Cypher enforces strict AND across required thread *types*.

import neo4j, { Driver, Session } from 'neo4j-driver';
import { ThreadType, Thread, Knot } from './retrieval_intersect';

export interface Neo4jAdapterOpts {
  uri: string;
  user: string;
  password: string;
  database?: string;
}

export class Neo4jIndexAdapter {
  private driver: Driver;
  private database?: string;

  constructor(opts: Neo4jAdapterOpts) {
    this.driver = neo4j.driver(opts.uri, neo4j.auth.basic(opts.user, opts.password));
    this.database = opts.database;
  }

  private session(): Session {
    return this.driver.session({ database: this.database });
  }

  /**
   * Resolve threads by text for a given type.
   * This is a tiny FTS-style like query; replace with your real FTS/ANN bridge if available.
   */
  async resolveThreadsByText(type: ThreadType, texts: string[]): Promise<Thread[]> {
    if (!texts?.length) return [];
    const s = this.session();
    try {
      const res = await s.run(
        `
        MATCH (t:Thread {type: $type})
        WHERE ANY(txt IN $texts WHERE toLower(t.key) CONTAINS toLower(txt))
        RETURN t.id AS id, t.type AS type, t.key AS key
        LIMIT 200
        `,
        { type, texts }
      );
      return res.records.map(r => ({ id: r.get('id'), type: r.get('type'), key: r.get('key') }));
    } finally {
      await s.close();
    }
  }

  /**
   * Strict AND across thread *types*: a knot qualifies only if it crosses
   * at least one thread for *each* required type in $requiredTypes.
   * We pass threadIds to narrow space, then enforce coverage by DISTINCT t.type count.
   */
  async knotsCrossingThreads(threadIds: string[]): Promise<Knot[]> {
    if (!threadIds?.length) return [];
    const s = this.session();
    try {
      const res = await s.run(
        `
        // Collect the required types present in the provided threadIds
        MATCH (t:Thread)
        WHERE t.id IN $ids
        WITH collect(DISTINCT t.type) AS requiredTypes, $ids AS ids

        // Find knots that cross any of the provided threads
        MATCH (k:Knot)-[r:IN_TIME|AT_PLACE|WITH_ACTOR|ABOUT_THEME|WEARS_GLYPH]->(t:Thread)
        WHERE t.id IN ids
        WITH k, requiredTypes, collect(DISTINCT t.type) AS hitTypes

        // Strict AND: knot must include all requiredTypes
        WHERE ALL(tt IN requiredTypes WHERE tt IN hitTypes)

        RETURN k.id AS id, k.t AS t, k.salience AS salience,
               // crude evidence score proxy: number of shards linked (0..1 scale later)
               size([(k)-[:HAS_SHARD]->(:Shard) | 1]) AS evCount
        LIMIT 1000
        `,
        { ids: threadIds }
      );

      return res.records.map(r => ({
        id: r.get('id'),
        t: (r.get('t') as any)?.toNumber ? (r.get('t') as any).toNumber() : (typeof r.get('t') === 'number' ? r.get('t') : Date.now()),
        salience: typeof r.get('salience') === 'number' ? r.get('salience') : 0.5,
        evidence: Math.min(1, Number(r.get('evCount') || 0) / 4) // normalize a bit
      }));
    } finally {
      await s.close();
    }
  }

  /**
   * OR retrieval (used for relax / exploration when you drop constraints).
   * Returns knots that cross *any* of the provided threadIds.
   */
  async knotsCrossingAny(threadIds: string[]): Promise<Knot[]> {
    if (!threadIds?.length) return [];
    const s = this.session();
    try {
      const res = await s.run(
        `
        MATCH (k:Knot)-[:IN_TIME|AT_PLACE|WITH_ACTOR|ABOUT_THEME|WEARS_GLYPH]->(t:Thread)
        WHERE t.id IN $ids
        WITH k, count(DISTINCT t) AS hits,
             size([(k)-[:HAS_SHARD]->(:Shard) | 1]) AS evCount
        RETURN k.id AS id, k.t AS t, k.salience AS salience, evCount
        ORDER BY hits DESC, k.salience DESC, k.t DESC
        LIMIT 2000
        `,
        { ids: threadIds }
      );
      return res.records.map(r => ({
        id: r.get('id'),
        t: (r.get('t') as any)?.toNumber ? (r.get('t') as any).toNumber() : (typeof r.get('t') === 'number' ? r.get('t') : Date.now()),
        salience: typeof r.get('salience') === 'number' ? r.get('salience') : 0.5,
        evidence: Math.min(1, Number(r.get('evCount') || 0) / 4)
      }));
    } finally {
      await s.close();
    }
  }

  /**
   * Simple recency boost helper (same notion used in retrieval_intersect).
   * If you want per-user decay tuning, inject it from the orchestrator.
   */
  recentBoost(now: number, t: number): number {
    const halfLifeMs = 1000 * 60 * 60 * 24 * 90; // ~90 days
    const dt = Math.max(0, now - t);
    return Math.exp(-dt / halfLifeMs);
  }

  /**
   * Content similarity placeholder for MMR.
   * Replace with a cosine distance over S2 embeddings if you have them.
   */
  contentSim(a: Knot, b: Knot): number {
    // TODO: inject vector store; for now, prefer temporal diversity
    const d = Math.abs((a.t || 0) - (b.t || 0));
    const day = 1000 * 60 * 60 * 24;
    // map 0..∞ days to 1..0 similarity (closer in time = more similar)
    const sim = Math.max(0, 1 - d / (30 * day));
    return sim;
  }

  async close() {
    await this.driver.close();
  }
}