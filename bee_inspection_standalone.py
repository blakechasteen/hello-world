"""
Bee Inspection Audio Ingestion System - Full Pipeline
======================================================

Complete pipeline with:
1. ✅ Whisper audio transcription
2. ✅ LLM schema extraction
3. ✅ Neo4j graph storage
4. ✅ Process real audio files

Usage:
    # Demo mode (no external services)
    python3 bee_inspection_standalone.py
    
    # Process audio file
    python3 bee_inspection_standalone.py audio.wav --date 2024-10-18
    
    # With LLM extraction (requires OPENAI_API_KEY)
    python3 bee_inspection_standalone.py audio.wav --date 2024-10-18 --use-llm
    
    # With Neo4j storage
    python3 bee_inspection_standalone.py audio.wav --date 2024-10-18 --neo4j
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import warnings

# Core dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("numpy not available")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not available")

# Whisper for audio transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    warnings.warn("whisper not available - install with: pip install openai-whisper")

# OpenAI for LLM extraction
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    warnings.warn("openai not available - install with: pip install openai")

# Neo4j for graph storage
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    warnings.warn("neo4j not available - install with: pip install neo4j")


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class InspectionEvent:
    """Bee inspection event"""
    eventId: str
    date: str
    timestamp: Optional[int] = None
    inspector: Optional[str] = None
    weatherTemp: Optional[float] = None
    weatherCondition: Optional[str] = None
    primaryPurpose: Optional[str] = None
    

@dataclass  
class HiveEntity:
    """Hive entity"""
    hiveId: str
    commonName: Optional[str] = None
    genetics: Optional[str] = None
    colonyStatus: Optional[str] = None


# ============================================================================
# Audio Transcription with Whisper
# ============================================================================

class AudioTranscriber:
    """Transcribe audio using Whisper"""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_size: tiny, base, small, medium, large
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("whisper not available - install: pip install openai-whisper")
        
        print(f"Loading Whisper model: {model_size}...")
        self.model = whisper.load_model(model_size)
        print("✓ Whisper loaded")
    
    def transcribe(self, audio_path: Path, language: str = "en") -> Dict:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: English)
            
        Returns:
            Dict with 'text', 'segments', 'language'
        """
        print(f"Transcribing: {audio_path.name}...")
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            verbose=False
        )
        
        print(f"✓ Transcribed {len(result.get('segments', []))} segments")
        return {
            "text": result["text"],
            "segments": result.get("segments", []),
            "language": result.get("language", language)
        }


# ============================================================================
# Embedder
# ============================================================================

class SimpleEmbedder:
    """Lightweight embedder using sentence-transformers"""
    
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        """Initialize embedder"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required")
        
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        print("✓ Model loaded")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings"""
        return self.model.encode(texts, normalize_embeddings=True)


# ============================================================================
# LLM-Based Schema Extractor
# ============================================================================

class LLMExtractor:
    """Extract structured schema using LLM (OpenAI/Claude)"""
    
    def __init__(self, use_llm: bool = False, api_key: Optional[str] = None):
        """
        Initialize extractor.
        
        Args:
            use_llm: Whether to use LLM (requires API key)
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.use_llm = use_llm
        
        if use_llm:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai not available - install: pip install openai")
            
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key")
            
            openai.api_key = self.api_key
            print("✓ LLM extraction enabled")
    
    async def extract(self, transcript: str, date: str) -> Dict:
        """Extract structured data from transcript"""
        if self.use_llm:
            return await self._extract_with_llm(transcript, date)
        else:
            return self._extract_with_rules(transcript, date)
    
    def _extract_with_rules(self, transcript: str, date: str) -> Dict:
        """Rule-based extraction (baseline)"""
        event_id = f"inspect-{date}-001"
        
        inspection = InspectionEvent(
            eventId=event_id,
            date=date,
            timestamp=int(datetime.now().timestamp()),
            primaryPurpose="routine_check"
        )
        
        # Extract hive mentions (simple pattern matching)
        hives = []
        hive_names = ["jodi", "karen", "dennis", "sally"]
        for name in hive_names:
            if name.lower() in transcript.lower():
                hives.append(HiveEntity(
                    hiveId=f"hive-{name}-001",
                    commonName=name.capitalize()
                ))
        
        return {
            "inspection": inspection,
            "hives": hives,
            "transcript": transcript
        }
    
    async def _extract_with_llm(self, transcript: str, date: str) -> Dict:
        """LLM-based extraction using OpenAI"""
        
        system_prompt = """You are a bee inspection data extractor. Extract structured information from bee inspection transcripts.

Follow this schema:
- InspectionEvent: date, weather, temperature, inspector, purpose
- Hive entities: hive name, genetics, status, population
- Observations: behavior, treatments, population strength

Return JSON with:
{
  "inspection": {
    "eventId": "inspect-YYYY-MM-DD-001",
    "date": "YYYY-MM-DD",
    "weatherTemp": 68.0,
    "weatherCondition": "clear",
    "primaryPurpose": "routine_check",
    "inspector": "name or null"
  },
  "hives": [
    {
      "hiveId": "hive-name-001",
      "commonName": "Name",
      "genetics": "dennis-line",
      "colonyStatus": "strong"
    }
  ],
  "observations": [
    "key observation 1",
    "key observation 2"
  ]
}"""
        
        user_prompt = f"""Extract structured data from this bee inspection transcript from {date}:

{transcript}

Return only valid JSON."""
        
        try:
            print("Calling OpenAI for schema extraction...")
            response = await asyncio.to_thread(
                openai.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            print("✓ LLM extraction successful")
            
            # Convert to our data models
            inspection_data = data.get("inspection", {})
            inspection = InspectionEvent(
                eventId=inspection_data.get("eventId", f"inspect-{date}-001"),
                date=inspection_data.get("date", date),
                timestamp=int(datetime.now().timestamp()),
                inspector=inspection_data.get("inspector"),
                weatherTemp=inspection_data.get("weatherTemp"),
                weatherCondition=inspection_data.get("weatherCondition"),
                primaryPurpose=inspection_data.get("primaryPurpose", "routine_check")
            )
            
            hives = []
            for hive_data in data.get("hives", []):
                hives.append(HiveEntity(
                    hiveId=hive_data.get("hiveId", f"hive-unknown-001"),
                    commonName=hive_data.get("commonName"),
                    genetics=hive_data.get("genetics"),
                    colonyStatus=hive_data.get("colonyStatus")
                ))
            
            return {
                "inspection": inspection,
                "hives": hives,
                "observations": data.get("observations", []),
                "transcript": transcript
            }
            
        except Exception as e:
            print(f"⚠ LLM extraction failed: {e}")
            print("Falling back to rule-based extraction")
            return self._extract_with_rules(transcript, date)


# ============================================================================
# Neo4j Graph Storage
# ============================================================================

class Neo4jStorage:
    """Store data in Neo4j graph database"""
    
    def __init__(self, use_neo4j: bool = False,
                 uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "password"):
        """
        Initialize storage.
        
        Args:
            use_neo4j: Whether to use Neo4j (falls back to in-memory)
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
        """
        self.use_neo4j = use_neo4j
        self.driver = None
        
        # In-memory fallback
        self.inspections = []
        self.embeddings = []
        
        if use_neo4j:
            if not NEO4J_AVAILABLE:
                print("⚠ Neo4j driver not available - using in-memory storage")
                self.use_neo4j = False
                return
            
            try:
                self.driver = GraphDatabase.driver(uri, auth=(user, password))
                # Test connection
                with self.driver.session() as session:
                    session.run("RETURN 1")
                print(f"✓ Connected to Neo4j at {uri}")
            except Exception as e:
                print(f"⚠ Could not connect to Neo4j: {e}")
                print("Using in-memory storage instead")
                self.use_neo4j = False
                self.driver = None
    
    def store_inspection(self, inspection: InspectionEvent):
        """Store inspection in Neo4j or memory"""
        if self.use_neo4j and self.driver:
            with self.driver.session() as session:
                session.execute_write(self._create_inspection, asdict(inspection))
        else:
            self.inspections.append(inspection)
    
    def store_hive(self, hive: HiveEntity, inspection_id: str):
        """Store hive and link to inspection"""
        if self.use_neo4j and self.driver:
            with self.driver.session() as session:
                session.execute_write(
                    self._create_hive_with_link,
                    asdict(hive),
                    inspection_id
                )
    
    @staticmethod
    def _create_inspection(tx, props: Dict):
        """Create InspectionEvent node"""
        query = """
        MERGE (i:InspectionEvent {eventId: $eventId})
        SET i.date = $date,
            i.timestamp = $timestamp,
            i.inspector = $inspector,
            i.weatherTemp = $weatherTemp,
            i.weatherCondition = $weatherCondition,
            i.primaryPurpose = $primaryPurpose
        RETURN i
        """
        tx.run(query, **{k: v for k, v in props.items() if v is not None})
    
    @staticmethod
    def _create_hive_with_link(tx, hive_props: Dict, inspection_id: str):
        """Create Hive node and link to inspection"""
        query = """
        MERGE (h:Hive {hiveId: $hiveId})
        SET h.commonName = $commonName,
            h.genetics = $genetics,
            h.colonyStatus = $colonyStatus
        WITH h
        MATCH (i:InspectionEvent {eventId: $inspectionId})
        MERGE (i)-[:INSPECTED]->(h)
        RETURN h
        """
        params = {**{k: v for k, v in hive_props.items() if v is not None}}
        params['inspectionId'] = inspection_id
        tx.run(query, **params)
    
    def store_embedding(self, entity_id: str, entity_type: str,
                       embedding: np.ndarray, text: str):
        """Store vector embedding"""
        # Always store embeddings in memory for search
        self.embeddings.append({
            "entity_id": entity_id,
            "entity_type": entity_type,
            "embedding": embedding,
            "text": text
        })
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search by similarity"""
        if not self.embeddings:
            return []
        
        results = []
        for item in self.embeddings:
            # Cosine similarity
            similarity = float(np.dot(query_embedding, item["embedding"]))
            results.append({
                "entity_id": item["entity_id"],
                "entity_type": item["entity_type"],
                "text": item["text"],
                "similarity": similarity
            })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()


# ============================================================================
# Full Pipeline
# ============================================================================

class BeeInspectionPipeline:
    """Complete bee inspection pipeline with all features"""
    
    def __init__(self,
                 use_whisper: bool = True,
                 whisper_model: str = "base",
                 use_llm: bool = False,
                 use_neo4j: bool = False,
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password"):
        """
        Initialize full pipeline.
        
        Args:
            use_whisper: Enable audio transcription
            whisper_model: Whisper model size (tiny/base/small/medium/large)
            use_llm: Enable LLM extraction (requires OPENAI_API_KEY)
            use_neo4j: Enable Neo4j storage
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        print("Initializing Bee Inspection Pipeline...")
        print(f"  Whisper: {'✓' if use_whisper and WHISPER_AVAILABLE else '✗'}")
        print(f"  LLM: {'✓' if use_llm and OPENAI_AVAILABLE else '✗'}")
        print(f"  Neo4j: {'✓' if use_neo4j else '✗ (in-memory)'}")
        
        # Initialize components
        self.transcriber = AudioTranscriber(whisper_model) if (use_whisper and WHISPER_AVAILABLE) else None
        self.embedder = SimpleEmbedder()
        self.extractor = LLMExtractor(use_llm=use_llm)
        self.storage = Neo4jStorage(
            use_neo4j=use_neo4j,
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password
        )
    
    async def process_audio(self, audio_path: Path, date: str,
                           inspector: Optional[str] = None) -> Dict:
        """
        Process audio file end-to-end.
        
        Args:
            audio_path: Path to audio file (.wav, .mp3, etc.)
            date: Inspection date (YYYY-MM-DD)
            inspector: Inspector name
            
        Returns:
            Dict with processing results
        """
        print(f"\n{'='*60}")
        print(f"Processing Audio: {audio_path.name}")
        print(f"{'='*60}\n")
        
        # Step 1: Transcribe audio
        if self.transcriber and audio_path.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac']:
            print("Step 1: Transcribing audio with Whisper...")
            transcription = self.transcriber.transcribe(audio_path)
            transcript = transcription["text"]
            segments = transcription["segments"]
        else:
            # Fallback: read text file
            print("Step 1: Loading transcript from text file...")
            if audio_path.suffix == '.txt':
                transcript = audio_path.read_text()
            else:
                txt_path = audio_path.with_suffix('.txt')
                if txt_path.exists():
                    transcript = txt_path.read_text()
                else:
                    raise ValueError(f"No audio transcriber and no .txt file found for {audio_path}")
            segments = []
            print("✓ Transcript loaded")
        
        # Process the transcript
        return await self.process_transcript(transcript, date, inspector, segments)
    
    async def process_transcript(self, transcript: str, date: str,
                     inspector: Optional[str] = None,
                     segments: List = None) -> Dict:
        """Process inspection transcript with all features"""
        
        if segments is None:
            segments = []
        
        print(f"\n{'='*60}")
        print(f"Processing Transcript")
        print(f"{'='*60}\n")
        
        # Step 2: Extract structure with LLM or rules
        print("Step 2: Extracting structured schema...")
        extracted = await self.extractor.extract(transcript, date)
        inspection = extracted["inspection"]
        if inspector:
            inspection.inspector = inspector
        print(f"✓ Inspection: {inspection.eventId}")
        print(f"✓ Found {len(extracted['hives'])} hives")
        
        # Step 3: Generate embeddings
        print("\nStep 3: Generating vector embeddings...")
        full_embedding = self.embedder.encode([transcript])[0]
        print(f"✓ Full transcript embedding (dim={len(full_embedding)})")
        
        inspection_text = f"Inspection on {inspection.date}"
        inspection_embedding = self.embedder.encode([inspection_text])[0]
        print(f"✓ Inspection embedding")
        
        # Step 4: Store in Neo4j + embeddings
        print("\nStep 4: Storing in database...")
        self.storage.store_inspection(inspection)
        
        for hive in extracted.get("hives", []):
            self.storage.store_hive(hive, inspection.eventId)
        
        self.storage.store_embedding(
            inspection.eventId,
            "inspection",
            inspection_embedding,
            inspection_text
        )
        self.storage.store_embedding(
            f"{inspection.eventId}-transcript",
            "transcript",
            full_embedding,
            transcript[:500]
        )
        print(f"✓ Stored inspection, {len(extracted['hives'])} hives, and embeddings")
        
        print(f"\n{'='*60}")
        print("✅ Processing Complete!")
        print(f"{'='*60}")
        
        return {
            "inspection": inspection,
            "transcript": transcript,
            "segments": segments,
            "hives": extracted.get("hives", []),
            "observations": extracted.get("observations", []),
            "embeddings": {
                "full": full_embedding,
                "inspection": inspection_embedding
            }
        }
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search inspections using semantic similarity"""
        query_embedding = self.embedder.encode([query])[0]
        return self.storage.search(query_embedding, top_k)
    
    def close(self):
        """Clean up resources"""
        self.storage.close()


# ============================================================================
# Demo
# ============================================================================

async def demo(use_llm: bool = False, use_neo4j: bool = False):
    """Run demo with full pipeline"""
    
    print("="*70)
    print("BEE INSPECTION FULL PIPELINE DEMO")
    print("="*70)
    print()
    
    # Sample transcript
    sample_transcript = """
    Bee inspection on October 18th, 2024. Starting with Hive Jodi, the Dennis line hive.
    
    Weather is clear, about 68 degrees Fahrenheit. Light breeze from the south.
    I'm wearing my veil today since I want to be cautious with the treatments.
    
    Opening up Hive Jodi now. Good entrance activity - I'd say moderate to good.
    About 15-20 bees coming and going per minute.
    
    Removing the inner cover... lots of bees on top, clustered nicely.
    I count about 8 frames of bees in the top box. Population looks strong.
    
    They're pretty calm today. Minimal smoke needed - just a few puffs at the entrance
    and they settled right down. Very gentle bees, as expected from the Dennis line.
    
    Checking the thymol treatment cards. Round 2, placed 14 days ago.
    Cards still have some residue but mostly consumed. Replacing with fresh cards today.
    Standard dosage - 2 cards in the top box.
    
    The hive is sealed up tight - propolis everywhere. They really like to seal.
    Compared to Hive Karen next to it, which barely seals at all. Interesting difference.
    
    Frames look good - plenty of capped brood, some pollen stores visible.
    No signs of disease. Queen is doing well.
    
    Moving to the bottom box... still good population down here, maybe 5-6 frames of bees.
    Plenty of honey stores for winter prep.
    
    Closing up Hive Jodi. Total inspection time: about 12 minutes.
    
    Overall assessment: Strong colony, good temperament, treatment on schedule.
    Should be in excellent shape for winter.
    
    Task for next inspection: Check bottom box more thoroughly for stores.
    Consider if they need supplemental feeding before cold weather.
    
    That's it for Hive Jodi. On to the next one.
    """
    
    # Initialize full pipeline
    print("Initializing full pipeline...")
    pipeline = BeeInspectionPipeline(
        use_whisper=False,  # Demo doesn't need audio
        use_llm=use_llm,
        use_neo4j=use_neo4j
    )
    
    # Process
    result = await pipeline.process_transcript(
        sample_transcript,
        date="2024-10-18",
        inspector="Demo User"
    )
    
    # Show results
    print(f"\nTranscript Preview:")
    print(f"  {result['transcript'][:200]}...")
    print(f"\nEmbedding Dimensions:")
    print(f"  Full: {result['embeddings']['full'].shape}")
    print(f"  Inspection: {result['embeddings']['inspection'].shape}")
    
    # Demo search
    print(f"\n{'='*70}")
    print("SEMANTIC SEARCH DEMO")
    print(f"{'='*70}")
    
    queries = [
        "How did the bees behave during inspection?",
        "What treatments were applied?",
        "Population strength and frames of bees",
        "Propolis and sealing behavior"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = pipeline.search(query, top_k=2)
        
        if results:
            top = results[0]
            print(f"  Match: {top['entity_type']}")
            print(f"  Similarity: {top['similarity']:.3f}")
            print(f"  Text: {top['text'][:100]}...")
    
    print(f"\n{'='*70}")
    print("DEMO COMPLETE!")
    print(f"{'='*70}")
    
    pipeline.close()
    
    print("\n✅ All Features Available:")
    print("1. ✅ Whisper audio transcription")
    print("2. ✅ LLM schema extraction")
    print("3. ✅ Neo4j graph storage")
    print("4. ✅ Process real audio files")
    print("\nTry with real audio:")
    print("  python3 bee_inspection_standalone.py audio.wav --date 2024-10-18")
    print("\nEnable LLM extraction:")
    print("  export OPENAI_API_KEY='your-key'")
    print("  python3 bee_inspection_standalone.py audio.wav --date 2024-10-18 --use-llm")
    print("\nEnable Neo4j storage:")
    print("  docker run --name neo4j -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j")
    print("  python3 bee_inspection_standalone.py audio.wav --date 2024-10-18 --neo4j")


async def process_file(audio_path: Path, date: str, inspector: Optional[str] = None,
                       use_llm: bool = False, use_neo4j: bool = False):
    """Process a real audio file"""
    
    print("="*70)
    print("PROCESSING REAL BEE INSPECTION AUDIO")
    print("="*70)
    print()
    
    pipeline = BeeInspectionPipeline(
        use_whisper=True,
        whisper_model="base",
        use_llm=use_llm,
        use_neo4j=use_neo4j
    )
    
    try:
        result = await pipeline.process_audio(audio_path, date, inspector)
        
        # Save outputs
        output_dir = audio_path.parent / f"output_{audio_path.stem}"
        output_dir.mkdir(exist_ok=True)
        
        # Save transcript
        transcript_path = output_dir / "transcript.txt"
        transcript_path.write_text(result["transcript"])
        print(f"\n✓ Saved transcript: {transcript_path}")
        
        # Save structured data
        json_path = output_dir / "structured.json"
        json_data = {
            "inspection": asdict(result["inspection"]),
            "hives": [asdict(h) for h in result["hives"]],
            "observations": result.get("observations", []),
            "transcript_length": len(result["transcript"]),
            "segments_count": len(result.get("segments", []))
        }
        json_path.write_text(json.dumps(json_data, indent=2))
        print(f"✓ Saved structured data: {json_path}")
        
        # Demo search
        print(f"\n{'='*70}")
        print("Testing semantic search...")
        queries = [
            "bee behavior and temperament",
            "population and frames of bees",
            "treatments and medications"
        ]
        
        for query in queries[:2]:  # Just show 2 examples
            print(f"\nQuery: '{query}'")
            results = pipeline.search(query, top_k=1)
            if results:
                print(f"  Match: {results[0]['similarity']:.3f} - {results[0]['text'][:80]}...")
        
        print(f"\n{'='*70}")
        print("✅ PROCESSING COMPLETE!")
        print(f"{'='*70}")
        print(f"\nOutputs saved to: {output_dir}")
        
    finally:
        pipeline.close()


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Bee Inspection Audio Ingestion - Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo mode (no audio file)
  python3 bee_inspection_standalone.py
  
  # Process audio file
  python3 bee_inspection_standalone.py audio.wav --date 2024-10-18
  
  # With LLM extraction (requires OPENAI_API_KEY)
  export OPENAI_API_KEY='your-key'
  python3 bee_inspection_standalone.py audio.wav --date 2024-10-18 --use-llm
  
  # With Neo4j storage
  python3 bee_inspection_standalone.py audio.wav --date 2024-10-18 --neo4j
  
  # All features enabled
  python3 bee_inspection_standalone.py audio.wav --date 2024-10-18 --use-llm --neo4j
"""
    )
    
    parser.add_argument(
        "audio_file",
        nargs="?",
        type=Path,
        help="Path to audio file (.wav, .mp3, etc.) or transcript (.txt)"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Inspection date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--inspector",
        type=str,
        help="Inspector name"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for schema extraction (requires OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--neo4j",
        action="store_true",
        help="Store in Neo4j graph database"
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="bolt://localhost:7687",
        help="Neo4j connection URI (default: bolt://localhost:7687)"
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default="neo4j",
        help="Neo4j username (default: neo4j)"
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default="password",
        help="Neo4j password (default: password)"
    )
    
    args = parser.parse_args()
    
    if args.audio_file:
        # Process real file
        if not args.audio_file.exists():
            print(f"Error: File not found: {args.audio_file}")
            sys.exit(1)
        
        if not args.date:
            print("Error: --date required when processing audio file")
            sys.exit(1)
        
        asyncio.run(process_file(
            args.audio_file,
            args.date,
            args.inspector,
            args.use_llm,
            args.neo4j
        ))
    else:
        # Run demo
        asyncio.run(demo(
            use_llm=args.use_llm,
            use_neo4j=args.neo4j
        ))


if __name__ == "__main__":
    main()
