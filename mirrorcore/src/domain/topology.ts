/** Persistent motif score: if an item keeps reappearing across buckets, boost it. */
export type MotifObs = { id:string; bucket:number }; // bucket from time_bucket()
export function persistentBoost(observations:MotifObs[], halfLifeBuckets=6){
  const byId = new Map<string, number[]>();
  for (const o of observations){
    const xs = byId.get(o.id) ?? []; xs.push(o.bucket); byId.set(o.id, xs);
  }
  const now = Math.max(...observations.map(o=>o.bucket), 0);
  const boost = new Map<string, number>();
  for (const [id,bks] of byId){
    let s = 0;
    for (const b of bks){
      const age = (now-b);
      s += Math.exp(-age/halfLifeBuckets);
    }
    boost.set(id, s/(1+boost.size));
  }
  return boost; // map id -> small positive boost
}
