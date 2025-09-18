# MirrorCore Loom v2

MirrorCore Loom v2 captures the layer-tagged architecture for the holographic loom.
It ships as a TypeScript project with adapters for Neo4j, Notion, and SQLite along
with retrieval and pack-building utilities that align with the layer diagram.

## Getting Started

1. Install dependencies using pnpm (Node.js 20+).
   ```sh
   pnpm install
   ```
2. Copy `.env.example` to `.env` and fill in the secrets for Neo4j, Notion, and SQLite.
3. Run the dev script to execute the sample pack build.
   ```sh
   pnpm dev
   ```

## Scripts

- `pnpm dev` – boots the development harness in `scripts/dev.ts`.
- `pnpm sync:notion` – synchronises Notion tasks and time entries into SQLite.
- `pnpm build:pack` – runs an example pack build flow using mock data.

## Database Setup

Run the seed SQL into your local SQLite database before syncing Notion or retrieving
from Neo4j. Apply the Cypher migration after seeding your Neo4j graph.

# Holographic Loom — Layer Map
- [10 Knot] src/domain/knot.ts — narrative/loop operators over graph selections.
- [9 Category] Adapters + types provide objects/morphisms; index wires functors (adapters).
- [8 Information] scoring: entropy/novelty, MMR redundancy penalty.
- [7 Statistics] selectors + weight fitting hooks; simple regressions later.
- [6 Topology] persistent motifs across scales; stable “shapes” of content.
- [5 Geometry] metric choices over embedding manifolds; distance → similarity.
- [4 Combinatorics] strict→relax ladder + greedy per-token pack.
- [3 Graph] junction-first model + strict AND intersections in Neo4j.
- [2 Calculus] exponential decays/time flow; smooth weighting.
- [1 Linear Algebra] embeddings + cosine.
- [0 Bedrock] env, IO, token estimates, schema mirrors.
