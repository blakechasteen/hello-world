import { knotsCrossingThreads } from '../adapters/neo4j.js';
import { relaxAll, StrictQuery } from './relax_strategies.js';
export async function retrievalIntersect(q: StrictQuery, limit=200) {
  for (const step of relaxAll(q)) {
    const js = await knotsCrossingThreads(step.requiredKnots, step.requiredThreads);
    if (js.length) return js.slice(0, limit);
  }
  return [];
}
