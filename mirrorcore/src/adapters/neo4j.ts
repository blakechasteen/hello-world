import neo4j from 'neo4j-driver'; import { env } from '../env.js';
export const driver = neo4j.driver(env.neo4j.uri, neo4j.auth.basic(env.neo4j.user, env.neo4j.password));
export async function knotsCrossingThreads(requiredKnotIds: string[], requiredThreadIds: string[]) {
  const s = driver.session();
  try {
    const res = await s.run(
`MATCH (j:Junction)-[:AT_KNOT]->(k:Knot) WHERE k.id IN $knotIds
 WITH j, collect(DISTINCT k.id) AS ks WHERE size(ks) = $kCount
 MATCH (t:Thread)-[:HAS_JUNCTION]->(j) WHERE t.id IN $threadIds
 WITH j, collect(DISTINCT t.id) AS ts WHERE size(ts) = $tCount
 RETURN DISTINCT j LIMIT 500`,
      { knotIds: requiredKnotIds, threadIds: requiredThreadIds, kCount: requiredKnotIds.length, tCount: requiredThreadIds.length }
    );
    return res.records.map(r => r.get('j').properties);
  } finally { await s.close(); }
}
