"""
Neo4j Memory Store - Thread-Based Graph Storage
===============================================
Implements thread model: KNOT nodes crossing THREAD nodes.

Thread Types:
- TIME: Temporal threads (when)
- PLACE: Spatial threads (where)
- ACTOR: Person threads (who)
- THEME: Topic threads (what)
- GLYPH: Symbol threads (pattern)
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from ..protocol import Memory, MemoryQuery, RetrievalResult, Strategy

# Optional neo4j import
try:
    from neo4j import GraphDatabase, Driver
    _HAVE_NEO4J = True
except ImportError:
    GraphDatabase = None
    Driver = None
    _HAVE_NEO4J = False


class Neo4jMemoryStore:
    """
    Neo4j-backed memory store using thread model.
    
    Model:
        (KNOT:Memory) -[IN_TIME]-> (THREAD:Time)
        (KNOT:Memory) -[AT_PLACE]-> (THREAD:Place)
        (KNOT:Memory) -[WITH_ACTOR]-> (THREAD:Actor)
        (KNOT:Memory) -[ABOUT_THEME]-> (THREAD:Theme)
        (KNOT:Memory) -[WEARS_GLYPH]-> (THREAD:Glyph)
    
    Retrieval:
        Find KNOTs that cross multiple THREADs (thread intersection)
    
    Requires: pip install neo4j
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        if not _HAVE_NEO4J:
            raise RuntimeError(
                "neo4j driver not installed. Install with: pip install neo4j"
            )
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.logger = logging.getLogger(__name__)
        
        # Create constraints and indexes
        self._setup_schema()
        
        self.logger.info(f"Neo4j store initialized: {uri}/{database}")
    
    def _setup_schema(self):
        """Create indexes and constraints."""
        with self.driver.session(database=self.database) as session:
            # Constraints
            try:
                session.run("CREATE CONSTRAINT knot_id IF NOT EXISTS FOR (k:KNOT) REQUIRE k.id IS UNIQUE")
                session.run("CREATE CONSTRAINT thread_name IF NOT EXISTS FOR (t:THREAD) REQUIRE (t.type, t.name) IS UNIQUE")
            except Exception as e:
                self.logger.warning(f"Schema setup warning: {e}")
    
    async def store(self, memory: Memory) -> str:
        """
        Store memory as KNOT crossing THREADs.
        
        Extracts threads from context:
        - context['time'] → TIME thread
        - context['place'] → PLACE thread
        - context['people'] → ACTOR threads
        - context['topics'] → THEME threads
        """
        with self.driver.session(database=self.database) as session:
            # Generate ID if needed
            mem_id = memory.id or f"knot_{memory.timestamp.isoformat()}"
            
            # Create KNOT node
            session.run("""
                CREATE (k:KNOT {
                    id: $id,
                    text: $text,
                    timestamp: datetime($timestamp),
                    user_id: $user_id
                })
            """, {
                'id': mem_id,
                'text': memory.text,
                'timestamp': memory.timestamp.isoformat(),
                'user_id': memory.metadata.get('user_id', 'default')
            })
            
            # Create thread connections
            self._connect_threads(session, mem_id, memory.context)
            
            self.logger.info(f"Stored KNOT {mem_id} with threads")
            return mem_id
    
    def _connect_threads(self, session, knot_id: str, context: Dict):
        """Connect KNOT to THREADs based on context."""
        # TIME thread
        if 'time' in context:
            session.run("""
                MERGE (t:THREAD {type: 'TIME', name: $name})
                WITH t
                MATCH (k:KNOT {id: $knot_id})
                MERGE (k)-[:IN_TIME]->(t)
            """, {'knot_id': knot_id, 'name': str(context['time'])})
        
        # PLACE thread
        if 'place' in context:
            session.run("""
                MERGE (t:THREAD {type: 'PLACE', name: $name})
                WITH t
                MATCH (k:KNOT {id: $knot_id})
                MERGE (k)-[:AT_PLACE]->(t)
            """, {'knot_id': knot_id, 'name': str(context['place'])})
        
        # ACTOR threads (can be list)
        people = context.get('people', [])
        if isinstance(people, str):
            people = [people]
        for person in people:
            session.run("""
                MERGE (t:THREAD {type: 'ACTOR', name: $name})
                WITH t
                MATCH (k:KNOT {id: $knot_id})
                MERGE (k)-[:WITH_ACTOR]->(t)
            """, {'knot_id': knot_id, 'name': str(person)})
        
        # THEME threads (can be list)
        topics = context.get('topics', [])
        if isinstance(topics, str):
            topics = [topics]
        for topic in topics:
            session.run("""
                MERGE (t:THREAD {type: 'THEME', name: $name})
                WITH t
                MATCH (k:KNOT {id: $knot_id})
                MERGE (k)-[:ABOUT_THEME]->(t)
            """, {'knot_id': knot_id, 'name': str(topic)})
    
    async def retrieve(
        self,
        query: MemoryQuery,
        strategy: Strategy = Strategy.FUSED
    ) -> RetrievalResult:
        """
        Retrieve memories via thread intersection.
        
        Strategies:
        - TEMPORAL: Find recent memories (TIME threads)
        - SEMANTIC: Text search (full-text index)
        - GRAPH: Thread traversal (connected KNOTs)
        - PATTERN: Pattern matching (Cypher)
        - FUSED: Combined scoring
        """
        with self.driver.session(database=self.database) as session:
            if strategy == Strategy.TEMPORAL:
                results = self._retrieve_temporal(session, query)
            elif strategy == Strategy.GRAPH:
                results = self._retrieve_graph(session, query)
            else:
                # Default: text search
                results = self._retrieve_text(session, query)
            
            return results
    
    def _retrieve_temporal(self, session, query: MemoryQuery) -> RetrievalResult:
        """Get recent memories."""
        result = session.run("""
            MATCH (k:KNOT)
            WHERE k.user_id = $user_id
            RETURN k.id AS id, k.text AS text, k.timestamp AS timestamp
            ORDER BY k.timestamp DESC
            LIMIT $limit
        """, {'user_id': query.user_id, 'limit': query.limit})
        
        return self._results_to_retrieval(result, 'temporal')
    
    def _retrieve_text(self, session, query: MemoryQuery) -> RetrievalResult:
        """Text-based search (contains)."""
        result = session.run("""
            MATCH (k:KNOT)
            WHERE k.user_id = $user_id
              AND toLower(k.text) CONTAINS toLower($query)
            RETURN k.id AS id, k.text AS text, k.timestamp AS timestamp
            LIMIT $limit
        """, {
            'user_id': query.user_id,
            'query': query.text,
            'limit': query.limit
        })
        
        return self._results_to_retrieval(result, 'text_search')
    
    def _retrieve_graph(self, session, query: MemoryQuery) -> RetrievalResult:
        """Graph traversal from query threads."""
        # Extract potential threads from query
        # For now, simple text match
        result = session.run("""
            MATCH (k:KNOT)
            WHERE k.user_id = $user_id
            RETURN k.id AS id, k.text AS text, k.timestamp AS timestamp
            LIMIT $limit
        """, {'user_id': query.user_id, 'limit': query.limit})
        
        return self._results_to_retrieval(result, 'graph')
    
    def _results_to_retrieval(self, result, strategy_name: str) -> RetrievalResult:
        """Convert Neo4j results to RetrievalResult."""
        memories = []
        scores = []
        
        for record in result:
            mem = Memory(
                id=record['id'],
                text=record['text'],
                timestamp=record['timestamp'].to_native() if hasattr(record['timestamp'], 'to_native') else datetime.now(),
                context={},
                metadata={'source': 'neo4j'}
            )
            memories.append(mem)
            scores.append(1.0)  # Uniform scores for now
        
        return RetrievalResult(
            memories=memories,
            scores=scores,
            strategy_used=strategy_name,
            metadata={'backend': 'neo4j', 'result_count': len(memories)}
        )
    
    async def delete(self, memory_id: str) -> bool:
        """Delete KNOT and its thread connections."""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (k:KNOT {id: $id})
                DETACH DELETE k
                RETURN count(k) AS deleted
            """, {'id': memory_id})
            
            record = result.single()
            deleted = record['deleted'] if record else 0
            return deleted > 0
    
    async def health_check(self) -> Dict:
        """Check Neo4j connection."""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("MATCH (k:KNOT) RETURN count(k) AS count")
                record = result.single()
                knot_count = record['count'] if record else 0
                
                result = session.run("MATCH (t:THREAD) RETURN count(t) AS count")
                record = result.single()
                thread_count = record['count'] if record else 0
                
                return {
                    'status': 'healthy',
                    'backend': 'neo4j',
                    'knot_count': knot_count,
                    'thread_count': thread_count,
                    'features': ['graph_traversal', 'thread_model', 'pattern_matching']
                }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'backend': 'neo4j',
                'error': str(e)
            }
    
    def close(self):
        """Close Neo4j driver."""
        if self.driver:
            self.driver.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
