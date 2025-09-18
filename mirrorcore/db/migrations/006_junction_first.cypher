CALL {
  WITH 1000 AS batch
  MATCH (t:Thread)-[r:CROSSES]->(k:Knot)
  WITH t,k LIMIT batch
  MERGE (j:Junction {threadId: t.id, knotId: k.id})
  ON CREATE SET j.createdAt = datetime()
  MERGE (t)-[:HAS_JUNCTION]->(j)
  MERGE (j)-[:AT_KNOT]->(k)
} IN TRANSACTIONS OF 1000 ROWS;
