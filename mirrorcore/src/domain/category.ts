/** Tiny categorical veneer: Adapters are functors between Worlds (Notion, Graph, Local).
 * Objects: Entities (Task, TimeEntry, Junction). Morphisms: Sync/Query transforms.
 */
export type World = 'Notion'|'Graph'|'Local';
export type Morphism<I,O> = (i:I)=>Promise<O>|O;
export function compose<A,B,C>(g:Morphism<B,C>, f:Morphism<A,B>):Morphism<A,C>{
  return async (a:A)=> g(await f(a));
}
