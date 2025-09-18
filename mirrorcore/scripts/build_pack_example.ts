import { buildPack } from '../src/pack/pack_builder.js';
import { Candidate, Weights } from '../src/domain/scoring.js';

const query = [1,0.2,0.1];
const weights: Weights = { w_sym:0.6, w_txt:0.3, w_g:0.2, w_rec:0.1, w_use:0.1, w_div:0.2, lambda:0.5, tau:8*3600*1000 };
const candidates: Candidate[] = [
  { id:'a', tokens:120, symScore:0.9, textVec:[0.9,0.1,0.2], content:'Spell thread a' },
  { id:'b', tokens:80, symScore:0.7, textVec:[0.8,0.2,0.1], content:'Spell thread b' },
  { id:'c', tokens:60, symScore:0.4, textVec:[0.2,0.9,0.1], content:'Spell thread c' }
];

const pack = buildPack(query, candidates, 200, weights);
console.log(JSON.stringify(pack, null, 2));
