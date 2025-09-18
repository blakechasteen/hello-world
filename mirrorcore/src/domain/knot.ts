export type KnotOp =
  | { kind:'loop'; times:number }
  | { kind:'bind'; tag:string }
  | { kind:'gate'; require:string[] }; // symbolic gates before selection

/** Applies a simple "knot ritual" over candidate ids: loop amplifies usageBoost. */
export function applyKnotOps<T extends {id:string; usageBoost?:number}>(cands:T[], ops:KnotOp[]){
  let out = cands.map(c => ({...c}));
  for (const op of ops){
    if (op.kind==='loop'){
      out = out.map(c => ({...c, usageBoost: (c.usageBoost ?? 0) + 0.02*op.times }));
    } else if (op.kind==='bind'){
      // no-op placeholder: could tag for downstream filters
    } else if (op.kind==='gate'){
      // no-op placeholder: could filter by required symbolic ids
    }
  }
  return out;
}
