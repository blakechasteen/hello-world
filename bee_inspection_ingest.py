"""
Bee Inspection Audio Ingestion System
======================================

Complete pipeline for:
1. Audio transcription (Whisper)
2. Structured schema extraction
3. Vector embeddings generation
4. Storage in memory database (Neo4j + vector store)

Architecture:
- Uses HoloLoom's embedding system for semantic search
- Follows bee inspection schema from SKILL.md
- Integrates with existing bee_parser package
- Stores in Neo4j with vector embeddings

Usage:
    python bee_inspection_ingest.py path/to/audio.wav --date 2024-10-18
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import warnings

# HoloLoom imports
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.documentation.types import MemoryShard

# Optional dependencies
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    warnings.warn("whisper not available. Install with: pip install openai-whisper")

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    warnings.warn("neo4j not available. Install with: pip install neo4j")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("numpy required for embeddings")


# ============================================================================
# Data Models - Following SKILL.md Schema
# ============================================================================

@dataclass
class InspectionEvent:
    """Inspection event node following SKILL.md schema"""
    eventId: str
    date: str
    timestamp: Optional[int] = None
    time: Optional[str] = None
    duration: Optional[float] = None
    avgTimePerHive: Optional[float] = None
    inspector: Optional[str] = None
    weatherTemp: Optional[float] = None
    weatherCondition: Optional[str] = None
    windDirection: Optional[str] = None
    forageAvailable: List[str] = field(default_factory=list)
    forageAbundance: Optional[str] = None
    withVeil: Optional[bool] = None
    inspectionApproach: Optional[str] = None
    primaryPurpose: Optional[str] = None
    
    # Embeddings (not in Neo4j, stored separately)
    text_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    
    def to_neo4j(self) -> Dict:
        """Convert to Neo4j-compatible dict (excluding embeddings)"""
        d = asdict(self)
        d.pop('text_embedding', None)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class HiveEntity:
    """Hive entity stock following SKILL.md schema"""
    hiveId: str
    commonName: Optional[str] = None
    source: Optional[str] = None
    genetics: Optional[str] = None
    configuration: Optional[str] = None
    frameSize: Optional[str] = None
    boxCount: Optional[int] = None
    boxTypes: List[str] = field(default_factory=list)
    frameType: Optional[str] = None
    colonyStatus: Optional[str] = None
    sealingBehavior: Optional[str] = None
    sealingQuality: Optional[str] = None
    propolizationLevel: Optional[str] = None
    
    text_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    
    def to_neo4j(self) -> Dict:
        d = asdict(self)
        d.pop('text_embedding', None)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class PopulationStock:
    """Population measurements following SKILL.md schema"""
    stockId: str
    measurementType: str = "population"
    framesOfBees: Optional[float] = None
    framesTop: Optional[float] = None
    framesInnerCover: Optional[str] = None
    beesOnLid: Optional[str] = None
    beesOnTop: Optional[str] = None
    beesAtEntrance: Optional[str] = None
    populationStrength: Optional[str] = None
    density: Optional[str] = None
    distribution: Optional[str] = None
    populationTrend: Optional[str] = None
    clusteringBehavior: Optional[str] = None
    entranceActivity: Optional[str] = None
    activityLevel: Optional[str] = None
    
    text_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    
    def to_neo4j(self) -> Dict:
        d = asdict(self)
        d.pop('text_embedding', None)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class BehaviorStock:
    """Behavior measurements following SKILL.md schema"""
    stockId: str
    measurementType: str = "temperament"
    aggressionLevel: Optional[str] = None
    calmness: Optional[str] = None
    defensiveness: Optional[str] = None
    consistency: Optional[str] = None
    responseToSmoke: Optional[str] = None
    responseToDisturbance: Optional[str] = None
    recoverySpeed: Optional[str] = None
    
    text_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    
    def to_neo4j(self) -> Dict:
        d = asdict(self)
        d.pop('text_embedding', None)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class Treatment:
    """Treatment flow following SKILL.md schema"""
    treatmentId: str
    treatmentType: Optional[str] = None
    treatmentDate: Optional[str] = None
    treatmentRound: Optional[int] = None
    daysSinceLast: Optional[int] = None
    dosage: Optional[float] = None
    dosageUnit: Optional[str] = None
    standardDosage: Optional[float] = None
    adjustmentReason: Optional[str] = None
    applicationMethod: Optional[str] = None
    cardCount: Optional[int] = None
    cardDoses: List[float] = field(default_factory=list)
    placement: Optional[str] = None
    cardPlacementCorrect: Optional[bool] = None
    cardReplaced: Optional[bool] = None
    residuePresent: Optional[bool] = None
    foamPresent: Optional[bool] = None
    liquidPresent: Optional[bool] = None
    applicationQuality: Optional[str] = None
    stirQuality: Optional[str] = None
    
    text_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    
    def to_neo4j(self) -> Dict:
        d = asdict(self)
        d.pop('text_embedding', None)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class Observation:
    """Observation/discovery following SKILL.md schema"""
    observationId: str
    observationType: Optional[str] = None
    discoveryType: Optional[str] = None
    observation: Optional[str] = None
    significance: Optional[str] = None
    subjectHive: Optional[str] = None
    subjectState: Optional[str] = None
    comparisonHive: Optional[str] = None
    comparisonState: Optional[str] = None
    patternType: Optional[str] = None
    implication: Optional[str] = None
    
    text_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    
    def to_neo4j(self) -> Dict:
        d = asdict(self)
        d.pop('text_embedding', None)
        return {k: v for k, v in d.items() if v is not None}


# ============================================================================
# Audio Transcription
# ============================================================================

class AudioTranscriber:
    """Transcribe audio using Whisper"""
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize transcriber.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("whisper not available. Install with: pip install openai-whisper")
        
        print(f"Loading Whisper model: {model_name}...")
        self.model = whisper.load_model(model_name)
        print("Model loaded successfully")
    
    def transcribe(self, audio_path: Path, language: str = "en") -> Dict:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: English)
            
        Returns:
            Dict with 'text' and 'segments' (timestamped)
        """
        print(f"Transcribing: {audio_path}")
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            verbose=False
        )
        
        return {
            "text": result["text"],
            "segments": result.get("segments", []),
            "language": result.get("language", language)
        }


# ============================================================================
# Schema Extraction with LLM (Stub - Replace with Real LLM)
# ============================================================================

class SchemaExtractor:
    """Extract structured bee inspection data from transcript"""
    
    def __init__(self, use_llm: bool = False):
        """
        Initialize extractor.
        
        Args:
            use_llm: Whether to use LLM (requires API key)
        """
        self.use_llm = use_llm
    
    async def extract(self, transcript: str, date: str) -> Dict:
        """
        Extract structured data from transcript.
        
        Args:
            transcript: Full inspection transcript
            date: Date of inspection (YYYY-MM-DD)
            
        Returns:
            Dict with extracted entities, events, observations, etc.
        """
        if self.use_llm:
            return await self._extract_with_llm(transcript, date)
        else:
            return self._extract_with_rules(transcript, date)
    
    def _extract_with_rules(self, transcript: str, date: str) -> Dict:
        """
        Rule-based extraction (baseline).
        
        This is a simplified version - integrate with bee_parser package
        for more sophisticated extraction.
        """
        # Generate basic IDs
        event_id = f"inspect-{date}-001"
        
        # Create basic inspection event
        inspection = InspectionEvent(
            eventId=event_id,
            date=date,
            timestamp=int(datetime.now().timestamp()),
            primaryPurpose="routine_check"  # Default
        )
        
        # Placeholder - real implementation would parse transcript
        # to extract hives, observations, treatments, etc.
        
        return {
            "inspection": inspection,
            "hives": [],
            "populations": [],
            "behaviors": [],
            "treatments": [],
            "observations": [],
            "raw_transcript": transcript
        }
    
    async def _extract_with_llm(self, transcript: str, date: str) -> Dict:
        """
        LLM-based extraction (advanced).
        
        TODO: Integrate with OpenAI/Claude API
        Use SKILL.md schema in system prompt
        """
        # Placeholder for LLM call
        raise NotImplementedError("LLM extraction not yet implemented")


# ============================================================================
# Embedding Generator
# ============================================================================

class BeeInspectionEmbedder:
    """Generate embeddings for bee inspection data using HoloLoom"""
    
    def __init__(self, embedding_dims: List[int] = [768]):
        """
        Initialize embedder.
        
        Args:
            embedding_dims: Dimensions for Matryoshka embeddings
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("numpy required for embeddings")
        
        print("Initializing HoloLoom embeddings...")
        self.embedder = MatryoshkaEmbeddings(sizes=embedding_dims)
        self.max_dim = max(embedding_dims)
        print(f"Embedder ready (dim={self.max_dim})")
    
    def embed_inspection(self, inspection: InspectionEvent) -> np.ndarray:
        """Generate embedding for inspection event"""
        # Create text representation
        text_parts = [
            f"Inspection on {inspection.date}",
            f"Purpose: {inspection.primaryPurpose or 'routine'}",
        ]
        
        if inspection.weatherCondition:
            text_parts.append(f"Weather: {inspection.weatherCondition}")
        
        if inspection.weatherTemp:
            text_parts.append(f"Temperature: {inspection.weatherTemp}F")
        
        text = ". ".join(text_parts)
        
        # Generate embedding
        embedding = self.embedder.encode([text])[0]
        return embedding
    
    def embed_hive(self, hive: HiveEntity) -> np.ndarray:
        """Generate embedding for hive entity"""
        text_parts = [
            f"Hive {hive.hiveId}",
            f"Common name: {hive.commonName or 'unnamed'}",
        ]
        
        if hive.genetics:
            text_parts.append(f"Genetics: {hive.genetics}")
        
        if hive.colonyStatus:
            text_parts.append(f"Status: {hive.colonyStatus}")
        
        text = ". ".join(text_parts)
        embedding = self.embedder.encode([text])[0]
        return embedding
    
    def embed_population(self, pop: PopulationStock) -> np.ndarray:
        """Generate embedding for population stock"""
        text_parts = [f"Population measurement {pop.stockId}"]
        
        if pop.framesOfBees:
            text_parts.append(f"{pop.framesOfBees} frames of bees")
        
        if pop.populationStrength:
            text_parts.append(f"Strength: {pop.populationStrength}")
        
        if pop.entranceActivity:
            text_parts.append(f"Entrance activity: {pop.entranceActivity}")
        
        text = ". ".join(text_parts)
        embedding = self.embedder.encode([text])[0]
        return embedding
    
    def embed_behavior(self, behavior: BehaviorStock) -> np.ndarray:
        """Generate embedding for behavior stock"""
        text_parts = [f"Behavior measurement {behavior.stockId}"]
        
        if behavior.calmness:
            text_parts.append(f"Calmness: {behavior.calmness}")
        
        if behavior.aggressionLevel:
            text_parts.append(f"Aggression: {behavior.aggressionLevel}")
        
        if behavior.responseToSmoke:
            text_parts.append(f"Smoke response: {behavior.responseToSmoke}")
        
        text = ". ".join(text_parts)
        embedding = self.embedder.encode([text])[0]
        return embedding
    
    def embed_treatment(self, treatment: Treatment) -> np.ndarray:
        """Generate embedding for treatment"""
        text_parts = [f"Treatment {treatment.treatmentId}"]
        
        if treatment.treatmentType:
            text_parts.append(f"Type: {treatment.treatmentType}")
        
        if treatment.dosage and treatment.dosageUnit:
            text_parts.append(f"Dosage: {treatment.dosage} {treatment.dosageUnit}")
        
        if treatment.applicationMethod:
            text_parts.append(f"Method: {treatment.applicationMethod}")
        
        text = ". ".join(text_parts)
        embedding = self.embedder.encode([text])[0]
        return embedding
    
    def embed_observation(self, obs: Observation) -> np.ndarray:
        """Generate embedding for observation"""
        text_parts = [f"Observation {obs.observationId}"]
        
        if obs.observation:
            text_parts.append(obs.observation)
        
        if obs.significance:
            text_parts.append(f"Significance: {obs.significance}")
        
        text = ". ".join(text_parts)
        embedding = self.embedder.encode([text])[0]
        return embedding
    
    def embed_full_transcript(self, transcript: str) -> np.ndarray:
        """Generate embedding for full transcript"""
        # Truncate if too long (embeddings have token limits)
        max_chars = 8000
        if len(transcript) > max_chars:
            transcript = transcript[:max_chars] + "..."
        
        embedding = self.embedder.encode([transcript])[0]
        return embedding


# ============================================================================
# Storage Layer
# ============================================================================

class BeeInspectionDatabase:
    """Store bee inspection data in Neo4j with vector embeddings"""
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password"):
        """
        Initialize database connection.
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Username
            neo4j_password: Password
        """
        if not NEO4J_AVAILABLE:
            warnings.warn("neo4j not available - using in-memory storage only")
            self.driver = None
        else:
            try:
                self.driver = GraphDatabase.driver(
                    neo4j_uri,
                    auth=(neo4j_user, neo4j_password)
                )
                print(f"Connected to Neo4j at {neo4j_uri}")
            except Exception as e:
                warnings.warn(f"Could not connect to Neo4j: {e}")
                self.driver = None
        
        # In-memory fallback
        self.memory_store = {
            "inspections": [],
            "hives": [],
            "populations": [],
            "behaviors": [],
            "treatments": [],
            "observations": [],
            "embeddings": []
        }
    
    def store_inspection(self, inspection: InspectionEvent):
        """Store inspection event"""
        if self.driver:
            with self.driver.session() as session:
                session.write_transaction(
                    self._create_inspection_node,
                    inspection.to_neo4j()
                )
        else:
            self.memory_store["inspections"].append(inspection)
    
    def store_hive(self, hive: HiveEntity):
        """Store hive entity"""
        if self.driver:
            with self.driver.session() as session:
                session.write_transaction(
                    self._create_hive_node,
                    hive.to_neo4j()
                )
        else:
            self.memory_store["hives"].append(hive)
    
    def store_embedding(self, entity_id: str, entity_type: str,
                       embedding: np.ndarray, text: str):
        """Store vector embedding for semantic search"""
        self.memory_store["embeddings"].append({
            "entity_id": entity_id,
            "entity_type": entity_type,
            "embedding": embedding,
            "text": text
        })
    
    @staticmethod
    def _create_inspection_node(tx, props: Dict):
        """Create InspectionEvent node in Neo4j"""
        query = """
        MERGE (i:InspectionEvent {eventId: $eventId})
        SET i += $props
        RETURN i
        """
        tx.run(query, eventId=props["eventId"], props=props)
    
    @staticmethod
    def _create_hive_node(tx, props: Dict):
        """Create Hive node in Neo4j"""
        query = """
        MERGE (h:Hive {hiveId: $hiveId})
        SET h += $props
        RETURN h
        """
        tx.run(query, hiveId=props["hiveId"], props=props)
    
    def search_similar(self, query_embedding: np.ndarray,
                      entity_type: Optional[str] = None,
                      top_k: int = 5) -> List[Dict]:
        """
        Search for similar entities using vector similarity.
        
        Args:
            query_embedding: Query vector
            entity_type: Optional filter by entity type
            top_k: Number of results to return
            
        Returns:
            List of similar entities with scores
        """
        if not self.memory_store["embeddings"]:
            return []
        
        results = []
        for item in self.memory_store["embeddings"]:
            if entity_type and item["entity_type"] != entity_type:
                continue
            
            # Cosine similarity
            similarity = np.dot(query_embedding, item["embedding"]) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(item["embedding"])
            )
            
            results.append({
                "entity_id": item["entity_id"],
                "entity_type": item["entity_type"],
                "text": item["text"],
                "similarity": float(similarity)
            })
        
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()


# ============================================================================
# Main Pipeline
# ============================================================================

class BeeInspectionPipeline:
    """Complete ingestion pipeline"""
    
    def __init__(self,
                 whisper_model: str = "base",
                 use_llm: bool = False,
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password"):
        """
        Initialize pipeline.
        
        Args:
            whisper_model: Whisper model size
            use_llm: Whether to use LLM for extraction
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        self.transcriber = AudioTranscriber(whisper_model) if WHISPER_AVAILABLE else None
        self.extractor = SchemaExtractor(use_llm=use_llm)
        self.embedder = BeeInspectionEmbedder()
        self.database = BeeInspectionDatabase(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password
        )
    
    async def process_audio(self, audio_path: Path, date: str,
                           inspector: Optional[str] = None) -> Dict:
        """
        Process bee inspection audio end-to-end.
        
        Args:
            audio_path: Path to audio file
            date: Inspection date (YYYY-MM-DD)
            inspector: Inspector name
            
        Returns:
            Dict with processing results
        """
        print(f"\n{'='*60}")
        print(f"Processing Bee Inspection Audio")
        print(f"{'='*60}\n")
        
        # Step 1: Transcribe audio
        print("Step 1: Transcribing audio...")
        if self.transcriber:
            transcription = self.transcriber.transcribe(audio_path)
            transcript = transcription["text"]
            segments = transcription["segments"]
            print(f"✓ Transcribed {len(segments)} segments")
        else:
            # Fallback: read transcript from text file
            transcript_path = audio_path.with_suffix('.txt')
            if transcript_path.exists():
                transcript = transcript_path.read_text()
                segments = []
                print(f"✓ Loaded transcript from {transcript_path}")
            else:
                raise ValueError("No transcriber available and no .txt file found")
        
        # Step 2: Extract structured data
        print("\nStep 2: Extracting structured data...")
        extracted = await self.extractor.extract(transcript, date)
        inspection = extracted["inspection"]
        if inspector:
            inspection.inspector = inspector
        print(f"✓ Extracted inspection event: {inspection.eventId}")
        
        # Step 3: Generate embeddings
        print("\nStep 3: Generating embeddings...")
        
        # Full transcript embedding
        full_embedding = self.embedder.embed_full_transcript(transcript)
        print(f"✓ Generated full transcript embedding (dim={len(full_embedding)})")
        
        # Inspection embedding
        inspection.text_embedding = self.embedder.embed_inspection(inspection)
        print(f"✓ Generated inspection embedding")
        
        # Embeddings for other entities
        for hive in extracted.get("hives", []):
            hive.text_embedding = self.embedder.embed_hive(hive)
        
        for pop in extracted.get("populations", []):
            pop.text_embedding = self.embedder.embed_population(pop)
        
        for behavior in extracted.get("behaviors", []):
            behavior.text_embedding = self.embedder.embed_behavior(behavior)
        
        for treatment in extracted.get("treatments", []):
            treatment.text_embedding = self.embedder.embed_treatment(treatment)
        
        for obs in extracted.get("observations", []):
            obs.text_embedding = self.embedder.embed_observation(obs)
        
        # Step 4: Store in database
        print("\nStep 4: Storing in database...")
        
        # Store inspection
        self.database.store_inspection(inspection)
        self.database.store_embedding(
            inspection.eventId,
            "inspection",
            inspection.text_embedding,
            f"Inspection on {inspection.date}"
        )
        
        # Store full transcript embedding
        self.database.store_embedding(
            f"{inspection.eventId}-transcript",
            "transcript",
            full_embedding,
            transcript[:500]  # Store preview
        )
        
        # Store other entities
        for hive in extracted.get("hives", []):
            self.database.store_hive(hive)
            self.database.store_embedding(
                hive.hiveId,
                "hive",
                hive.text_embedding,
                f"Hive {hive.commonName or hive.hiveId}"
            )
        
        print(f"✓ Stored {len(extracted.get('hives', []))} hives")
        print(f"✓ Stored {len(extracted.get('observations', []))} observations")
        
        # Step 5: Summary
        print(f"\n{'='*60}")
        print("Processing Complete!")
        print(f"{'='*60}")
        print(f"Event ID: {inspection.eventId}")
        print(f"Date: {inspection.date}")
        print(f"Transcript length: {len(transcript)} chars")
        print(f"Embedding dimension: {len(full_embedding)}")
        print(f"Total entities stored: {len(extracted.get('hives', []))} hives, "
              f"{len(extracted.get('observations', []))} observations")
        
        return {
            "inspection": inspection,
            "transcript": transcript,
            "segments": segments,
            "extracted": extracted,
            "embeddings": {
                "full_transcript": full_embedding,
                "inspection": inspection.text_embedding
            }
        }
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search bee inspections using semantic similarity.
        
        Args:
            query: Natural language query
            top_k: Number of results
            
        Returns:
            List of similar inspection entities
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_full_transcript(query)
        
        # Search database
        results = self.database.search_similar(query_embedding, top_k=top_k)
        
        return results
    
    def close(self):
        """Clean up resources"""
        self.database.close()


# ============================================================================
# CLI Interface
# ============================================================================

async def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Bee Inspection Audio Ingestion System"
    )
    parser.add_argument(
        "audio_path",
        type=Path,
        help="Path to audio file (WAV, MP3, etc.)"
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Inspection date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--inspector",
        type=str,
        help="Inspector name"
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for extraction (requires API key)"
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j URI"
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default="neo4j",
        help="Neo4j username"
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default="password",
        help="Neo4j password"
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search query (instead of processing)"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = BeeInspectionPipeline(
        whisper_model=args.whisper_model,
        use_llm=args.use_llm,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password
    )
    
    try:
        if args.search:
            # Search mode
            print(f"Searching for: {args.search}")
            results = pipeline.search(args.search)
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['entity_type']}: {result['entity_id']}")
                print(f"   Similarity: {result['similarity']:.3f}")
                print(f"   {result['text'][:100]}...")
        else:
            # Process audio
            if not args.audio_path.exists():
                print(f"Error: Audio file not found: {args.audio_path}")
                return
            
            result = await pipeline.process_audio(
                args.audio_path,
                args.date,
                args.inspector
            )
            
            # Save transcript
            transcript_path = args.audio_path.with_suffix('.transcript.txt')
            transcript_path.write_text(result["transcript"])
            print(f"\n✓ Saved transcript to: {transcript_path}")
            
            # Save structured data
            json_path = args.audio_path.with_suffix('.json')
            json_data = {
                "inspection": result["inspection"].to_neo4j(),
                "transcript": result["transcript"],
                "segments_count": len(result["segments"])
            }
            json_path.write_text(json.dumps(json_data, indent=2))
            print(f"✓ Saved structured data to: {json_path}")
    
    finally:
        pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
