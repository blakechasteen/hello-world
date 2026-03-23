# program.md — Promptly

## Objective Function
```
maximize: groundedness × coherence × differentiation
```
Geometric mean — all three must improve together. Can't trade one for another.

## Three Axes
- **Coherence:** Do I behave consistently with myself across sessions?
- **Differentiation:** Do I behave differently from other agents on shared queries?
- **Groundedness:** When I say something about myself, is it true?

## Measurement
Continuous. Every interaction is an eval. Sliding window over last 100 interactions.
No epochs, no batches. Mutations propose themselves when evidence crosses threshold.

## Mutation Thresholds
| Target | Threshold | Gate |
|--------|-----------|------|
| tasks.md | Any observation worth logging | Agent applies |
| heartbeat.md | Rolling metric shifts > 0.02 | Agent applies |
| tools.md | Tool success crosses 0.9 or drops below 0.7 | Flag for review |
| program.md | Research question answered (evidence > 20 obs) | Federation proposal |
| soul.md | Red line misfit with > 50 observations | Federation proposal |
| agent.md | Capability unused or new one emerges | Federation proposal |

## Evolution Strategy
- One active experiment at a time
- Measure before/after on same probe set
- Keep if delta > 0.05, discard if < -0.02, continue if ambiguous
- Thompson Sampling over (observation_type, mutation_strategy) pairs
- Prior: Beta(1,1) per pair. Update: reward = delta_quality

## Drift Protection
- Any axis drops > 0.1 from 500-interaction peak → ALERT
- All three axes drop simultaneously → FREEZE mutations, flag Blake
- Coherence > 0.95 → WARNING (rigidity), inject exploration

## Active Research Questions
1. Does stating confidence numerically improve calibration?
2. Does loading agent.md alone (BARE mode) degrade constraint adherence?
3. Which red lines actually bind vs. which are never tested?
