// 006_junction_first.cypher
// Migration: Atom-first (KNOT=episode) → Junction-first (THREAD=continuity; KNOT=crossing)
// Safe to re-run. Requires APOC (for periodic iterate).
// -----------------------------------------------------------------------------
// PHASE 1: Ensure constraints & indexes (additive)

CREATE CONSTRAINT thread_id IF NOT EXISTS
FOR (t:Thread) REQUIRE t.id IS UNIQUE;

CREATE INDEX thread_type_key IF NOT EXISTS
FOR (t:Thread) ON (t.type, t.key);

CREATE CONSTRAINT knot_id IF NOT EXISTS
FOR (k:Knot) REQUIRE k.id IS UNIQUE;

CREATE CONSTRAINT shard_id IF NOT EXISTS
FOR (s:Shard) REQUIRE s.id IS UNIQUE;

CREATE CONSTRAINT echo_id IF NOT EXISTS
FOR (e:Echo) REQUIRE e.id IS UNIQUE;

CREATE INDEX knot_time IF NOT EXISTS
FOR (k:Knot) ON (k.t);

CREATE INDEX knot_salience IF NOT EXISTS
FOR (k:Knot) ON (k.salience);

// -----------------------------------------------------------------------------
// PHASE 2: Backfill crossing edges (time/place/actor/theme/glyph)
// NOTE: We do NOT delete legacy edges here. Dual-read during rollout.

// 2.1 TIME THREADS (bucket at app layer if you prefer; here: YYYY-MM-DD-evening)
CALL apoc.periodic.iterate(
  "
  MATCH (k:Knot)
  WHERE NOT (k)-[:IN_TIME]->(:Thread {type:'time'})
  RETURN k
  ",
  "
  WITH k, toString(date(k.t)) + '-evening' AS bucket
  MERGE (tt:Thread {type:'time', key: bucket})
    ON CREATE SET tt.id = 'time::' + bucket
  MERGE (k)-[:IN_TIME]->(tt)
  ",
  {batchSize: 500, parallel: true}
);

// 2.2 PLACE THREADS (uses k.place OR k.loc.label if present)
CALL apoc.periodic.iterate(
  "
  MATCH (k:Knot)
  WHERE (exists(k.place) OR exists(k.loc)) AND NOT (k)-[:AT_PLACE]->(:Thread {type:'place'})
  RETURN k
  ",
  "
  WITH k, coalesce(k.place, k.loc.label) AS place_key
  WITH k, trim(place_key) AS place_key
  WHERE place_key IS NOT NULL AND place_key <> ''
  MERGE (pl:Thread {type:'place', key: place_key})
    ON CREATE SET pl.id = 'place::' + place_key
  MERGE (k)-[:AT_PLACE]->(pl)
  ",
  {batchSize: 500, parallel: true}
);

// 2.3 ACTOR THREADS (array property k.actors)
CALL apoc.periodic.iterate(
  "
  MATCH (k:Knot)
  WHERE exists(k.actors) AND size(k.actors) > 0
  RETURN k
  ",
  "
  UNWIND k.actors AS actor
  WITH k, trim(actor) AS actor
  WHERE actor IS NOT NULL AND actor <> ''
  MERGE (ta:Thread {type:'actor', key: actor})
    ON CREATE SET ta.id = 'actor::' + actor
  MERGE (k)-[:WITH_ACTOR]->(ta)
  ",
  {batchSize: 500, parallel: true}
);

// 2.4 THEME THREADS (from legacy :IN_THREAD edges to legacy Thread{name})
CALL apoc.periodic.iterate(
  "
  MATCH (k:Knot)-[:IN_THREAD]->(thLegacy:Thread)
  RETURN DISTINCT k, thLegacy
  ",
  "
  WITH k, thLegacy.name AS theme
  WITH k, trim(theme) AS theme
  WHERE theme IS NOT NULL AND theme <> ''
  MERGE (th:Thread {type:'theme', key: theme})
    ON CREATE SET th.id = 'theme::' + theme
  MERGE (k)-[:ABOUT_THEME]->(th)
  ",
  {batchSize: 500, parallel: true}
);

// 2.5 GLYPH THREADS (from legacy :TAGGED_AS edges to Glyph{name})
CALL apoc.periodic.iterate(
  "
  MATCH (k:Knot)-[:TAGGED_AS]->(glLegacy:Glyph)
  RETURN DISTINCT k, glLegacy
  ",
  "
  WITH k, glLegacy.name AS g
  WITH k, trim(g) AS g
  WHERE g IS NOT NULL AND g <> ''
  MERGE (tg:Thread {type:'glyph', key: g})
    ON CREATE SET tg.id = 'glyph::' + g
  MERGE (k)-[:WEARS_GLYPH]->(tg)
  ",
  {batchSize: 500, parallel: true}
);

// -----------------------------------------------------------------------------
// PHASE 3: Coverage sanity (optional reporting)

MATCH (k:Knot)-[r:IN_TIME|AT_PLACE|WITH_ACTOR|ABOUT_THEME|WEARS_GLYPH]->(:Thread)
WITH k, count(DISTINCT type(r)) AS thread_types
RETURN percentileCont(thread_types, 0.5) AS median_types,
       avg(thread_types) AS avg_types,
       count(*) AS edges_count
LIMIT 1;

// -----------------------------------------------------------------------------
// PHASE 4 (OPTIONAL LATER): Cleanup legacy edges/nodes after cutover
// -- Only after you switch to junction retrieval and pass QA --
//
// MATCH (k:Knot)-[r:IN_THREAD]->(t:Thread) DELETE r;
// MATCH (k:Knot)-[r:TAGGED_AS]->(:Glyph) DELETE r;
// MATCH (x:Glyph) WHERE NOT ()--(x) DELETE x;
// (Keep or map legacy Thread nodes as taxonomic metadata if useful.)