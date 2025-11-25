"""
Symbol Enrichment Batch Processor

Enriches 500 dream symbols with comprehensive literary references using
the Metaprompting Refinement Framework (MRF).

Key features:
- Async batch processing (10 concurrent LLM calls)
- Quality validation (3 metrics + overall score)
- Progress tracking (tqdm)
- Human review flagging
- Cost estimation

Usage:
    python enrich_symbols_batch.py

Or programmatically:
    enricher = SymbolEnricher(config)
    enriched = await enricher.enrich_all_500()
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    from tqdm.asyncio import tqdm
except ImportError:
    from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EnrichmentConfig:
    """Configuration for symbol enrichment."""

    llm_provider: str = "anthropic"
    llm_model: str = "claude-3-5-sonnet-20241022"
    batch_size: int = 10
    rate_limit_delay: float = 0.5  # seconds between batches

    # Quality thresholds
    min_cultural_diversity: float = 7.0
    min_emotional_resonance: float = 8.0
    min_total_references: int = 15
    max_total_references: int = 25

    # Quality scoring
    cultural_diversity_weight: float = 0.4
    emotional_resonance_weight: float = 0.35
    accessibility_weight: float = 0.25

    # Dry-run mode (no actual LLM calls)
    dry_run: bool = False
    dry_run_quality_offset: float = 0.8  # 80% of max quality


class SymbolEnricher:
    """Batch processor for enriching symbols with literary references."""

    def __init__(self, config: Optional[EnrichmentConfig] = None):
        """
        Initialize enricher.

        Args:
            config: EnrichmentConfig instance
        """
        self.config = config or EnrichmentConfig()
        self.metaprompt_template = self._load_metaprompt_template()
        self.enrichment_stats = {
            "total_enriched": 0,
            "successful": 0,
            "needs_review": 0,
            "failed": 0,
            "quality_scores": []
        }

    def _load_metaprompt_template(self) -> str:
        """Load symbol enrichment metaprompt template from file."""
        template_path = Path(__file__).parent / "symbol_enrichment_metaprompt.txt"

        if template_path.exists():
            with open(template_path, 'r') as f:
                return f.read()
        else:
            logger.warning(f"Metaprompt template not found at {template_path}")
            return self._get_default_metaprompt_template()

    def _get_default_metaprompt_template(self) -> str:
        """Return default metaprompt if file not found."""
        return """You are a comparative literature scholar with expertise in cross-cultural mythology, Jungian archetypal analysis, world literature, modern cinema, and philosophical symbolism.

SYMBOL TO ENRICH:
{SYMBOL_REQUEST}

YOUR TASK:
Expand literary and cultural references for this symbol using the framework below.

### ROLE
Comparative literature scholar with expertise in:
- Cross-cultural mythology (Greek, Norse, Egyptian, Vedic, Indigenous, African)
- World literature (Eastern, Western, Indigenous, Latin American)
- Modern cinema and media studies
- Jungian archetypal analysis
- Religious and philosophical symbolism

### OBJECTIVE
Primary: Generate 15-25 culturally diverse literary references
Secondary: Include archetypal roots, cinematic parallels, philosophical connections
When in doubt, prioritize: Cross-cultural diversity over canonical Western works

### PROCESS
1. Analyze symbol's emotional essence
2. Identify classical mythology parallels (≥3 cultures)
3. Find Western literature connections
4. Find Eastern literature connections
5. Identify Indigenous and African oral traditions
6. Add modern cinema and media
7. Include poetry and visual arts
8. Add philosophical and religious texts
9. Cross-reference with Jungian archetypes
10. Validate cultural diversity

### FORMAT
Output as structured JSON with these categories:
- classical_mythology (≥3 cultures)
- world_literature (≥2 Eastern, ≥2 Western)
- modern_cinema (≥3 films/shows)
- poetry_visual_arts (≥2 poems, ≥2 artworks)
- philosophy_religion (≥3 texts)
- contemporary_culture (≥2 references)

Each reference must include:
- title, culture, connection, quote/scene, emotional_resonance, connection_confidence

### CONSTRAINTS
- Do NOT only use Western sources (must include ≥3 non-Western cultures)
- Avoid obscure academic references without popular appeal
- Do NOT fabricate connections
- Limit to works with clear symbolic parallels

### UNCERTAINTY
If unclear: Mark connection_confidence (0.0-1.0), request validation, provide alternatives

### VALIDATION
Check output for:
✓ Cultural diversity: ≥3 non-Western cultures
✓ All 6 categories have ≥2 references
✓ 15-25 total references
✓ Valid JSON format

OUTPUT:
Provide enriched JSON following the format above."""

    async def enrich_symbol(self, symbol: Dict) -> Optional[Dict]:
        """
        Enrich a single symbol with comprehensive literary references.

        Args:
            symbol: Base symbol data with id, description, emotion_tags, category

        Returns:
            Enriched symbol with literary_references, archetypal_roots, quality scores
            or None if enrichment failed
        """
        symbol_id = symbol.get('symbol_id', 'unknown')

        try:
            # Create metaprompt for this symbol
            symbol_request = f"""
SYMBOL: {symbol['symbol_id']}
DESCRIPTION: {symbol['description']}
EMOTION_TAGS: {', '.join(symbol['emotion_tags'])}
CATEGORY: {symbol['category']}
EXISTING_REFERENCES: {json.dumps(symbol.get('literary_references', []))}
"""

            metaprompt = self.metaprompt_template.replace(
                "{SYMBOL_REQUEST}",
                symbol_request
            )

            # Execute LLM call or use dry-run mock
            if self.config.dry_run:
                enriched_data = self._generate_mock_enrichment(symbol)
            else:
                enriched_data = await self._call_llm(metaprompt)

            if enriched_data is None:
                logger.error(f"Failed to enrich {symbol_id}: LLM call returned None")
                self.enrichment_stats["failed"] += 1
                return None

            # Validate quality
            quality = self._validate_quality(enriched_data)
            enriched_data["quality_scores"] = quality

            if quality["overall"] < self.config.min_cultural_diversity:
                enriched_data["needs_human_review"] = True
                self.enrichment_stats["needs_review"] += 1
                logger.warning(f"Low quality for {symbol_id}: {quality['overall']:.2f}/10")
            else:
                self.enrichment_stats["successful"] += 1

            enriched_data["enrichment_date"] = datetime.now().isoformat()
            enriched_data["llm_model"] = self.config.llm_model
            self.enrichment_stats["quality_scores"].append(quality["overall"])
            self.enrichment_stats["total_enriched"] += 1

            return enriched_data

        except Exception as e:
            logger.error(f"Exception enriching {symbol_id}: {str(e)}")
            self.enrichment_stats["failed"] += 1
            return None

    async def _call_llm(self, metaprompt: str) -> Optional[Dict]:
        """
        Call LLM with metaprompt (placeholder - requires HoloLoom integration).

        In production, integrate with:
            from HoloLoom.llm import create_llm_client
            llm = create_llm_client(config)
            response = await llm.generate(metaprompt)
        """
        # Placeholder for LLM integration
        logger.info("LLM call would happen here - requires HoloLoom integration")
        return None

    def _generate_mock_enrichment(self, symbol: Dict) -> Dict:
        """
        Generate mock enrichment for dry-run mode.

        Creates realistic enriched data without LLM call for testing.
        """
        symbol_id = symbol['symbol_id']

        return {
            "symbol_id": symbol_id,
            "category": symbol['category'],
            "literary_references": {
                "classical_mythology": [
                    {
                        "title": f"Myth: {symbol_id.replace('_', ' ').title()}",
                        "culture": "Greek",
                        "connection": f"Mythological parallel to {symbol_id}",
                        "quote": f"A passage from the {symbol_id} myth",
                        "emotional_resonance": symbol['emotion_tags'][:2],
                        "connection_confidence": 0.85
                    },
                    {
                        "title": f"Legend: {symbol_id.replace('_', ' ').title()}",
                        "culture": "Norse",
                        "connection": f"Norse interpretation of {symbol_id}",
                        "quote": f"From the sagas about {symbol_id}",
                        "emotional_resonance": symbol['emotion_tags'][1:3],
                        "connection_confidence": 0.80
                    },
                    {
                        "title": f"Story: {symbol_id.replace('_', ' ').title()}",
                        "culture": "Hindu",
                        "connection": f"Hindu archetype of {symbol_id}",
                        "quote": f"From sacred texts about {symbol_id}",
                        "emotional_resonance": symbol['emotion_tags'][:2],
                        "connection_confidence": 0.82
                    }
                ],
                "world_literature": [
                    {
                        "title": f"Novel: {symbol_id.replace('_', ' ').title()}",
                        "culture": "Western",
                        "author": "Classic Author",
                        "connection": f"Literary exploration of {symbol_id}",
                        "quote": f"A famous line about {symbol_id}",
                        "emotional_resonance": symbol['emotion_tags'],
                        "connection_confidence": 0.88
                    },
                    {
                        "title": f"Story: Eastern {symbol_id.replace('_', ' ').title()}",
                        "culture": "Eastern",
                        "author": "Eastern Writer",
                        "connection": f"Eastern perspective on {symbol_id}",
                        "quote": f"Wisdom about {symbol_id}",
                        "emotional_resonance": symbol['emotion_tags'][:2],
                        "connection_confidence": 0.83
                    }
                ],
                "modern_cinema": [
                    {
                        "title": f"Film: {symbol_id.replace('_', ' ').title()}",
                        "medium": "Film",
                        "director": "Notable Director",
                        "year": 2020,
                        "connection": f"Cinematic interpretation of {symbol_id}",
                        "key_scene": f"Key scene depicting {symbol_id}",
                        "emotional_resonance": symbol['emotion_tags'],
                        "connection_confidence": 0.82
                    }
                ],
                "poetry_visual_arts": [
                    {
                        "title": f"Poem: {symbol_id.replace('_', ' ').title()}",
                        "medium": "Poem",
                        "author": "Poet",
                        "connection": f"Poetic treatment of {symbol_id}",
                        "key_lines": f"Lines about {symbol_id}",
                        "emotional_resonance": symbol['emotion_tags'],
                        "connection_confidence": 0.85
                    }
                ],
                "philosophy_religion": [
                    {
                        "title": f"Text: {symbol_id.replace('_', ' ').title()}",
                        "source": "Philosophical Work",
                        "connection": f"Philosophical perspective on {symbol_id}",
                        "key_concept": f"Key idea about {symbol_id}",
                        "emotional_resonance": symbol['emotion_tags'][:2],
                        "connection_confidence": 0.81
                    }
                ],
                "contemporary_culture": [
                    {
                        "title": f"Modern: {symbol_id.replace('_', ' ').title()}",
                        "medium": "Media",
                        "connection": f"Contemporary reference to {symbol_id}",
                        "emotional_resonance": symbol['emotion_tags'][:1],
                        "connection_confidence": 0.78
                    }
                ]
            },
            "archetypal_roots": {
                "primary": f"Jungian archetype related to {symbol_id}",
                "secondary": [f"Secondary archetype of {symbol_id}"],
                "mythological_patterns": [f"Pattern: {symbol_id}"]
            },
            "total_references": 14,  # Mock will be below threshold to show review flagging
            "cultural_diversity_score": 7.2
        }

    def _validate_quality(self, enriched_data: Dict) -> Dict:
        """
        Validate enriched symbol quality.

        Returns quality scores:
        - cultural_diversity: 0-10 (≥3 non-Western cultures)
        - emotional_resonance: 0-10 (references evoke symbol essence)
        - accessibility: 0-10 (60% popular, 40% scholarly)
        - overall: weighted average
        """
        # Cultural diversity (count unique cultures)
        all_cultures = set()
        for category_refs in enriched_data.get("literary_references", {}).values():
            if isinstance(category_refs, list):
                for ref in category_refs:
                    if isinstance(ref, dict):
                        all_cultures.add(ref.get("culture", "Unknown"))

        non_western = sum(1 for c in all_cultures
                         if c and not any(w in c for w in ["American", "British", "French", "German", "Russian", "Western"]))
        cultural_diversity = min(10.0, max(0.0, non_western * 2.5))

        # Emotional resonance (average connection_confidence)
        all_confidences = []
        for category_refs in enriched_data.get("literary_references", {}).values():
            if isinstance(category_refs, list):
                for ref in category_refs:
                    if isinstance(ref, dict):
                        conf = ref.get("connection_confidence", 0.5)
                        if isinstance(conf, (int, float)):
                            all_confidences.append(conf)

        emotional_resonance = (sum(all_confidences) / len(all_confidences) * 10) if all_confidences else 5.0

        # Accessibility (count popular vs scholarly)
        popular_refs = {
            "modern_cinema": enriched_data.get("literary_references", {}).get("modern_cinema", []),
            "contemporary_culture": enriched_data.get("literary_references", {}).get("contemporary_culture", [])
        }
        scholarly_refs = {
            "philosophy_religion": enriched_data.get("literary_references", {}).get("philosophy_religion", []),
            "classical_mythology": enriched_data.get("literary_references", {}).get("classical_mythology", [])
        }

        popular_count = sum(len(v) if isinstance(v, list) else 0 for v in popular_refs.values())
        scholarly_count = sum(len(v) if isinstance(v, list) else 0 for v in scholarly_refs.values())
        total_count = popular_count + scholarly_count

        if total_count > 0:
            popular_ratio = popular_count / total_count
            accessibility = 10.0 - abs(popular_ratio - 0.6) * 25
        else:
            accessibility = 5.0

        accessibility = min(10.0, max(0.0, accessibility))

        # Weighted overall score
        overall = (
            cultural_diversity * self.config.cultural_diversity_weight +
            emotional_resonance * self.config.emotional_resonance_weight +
            accessibility * self.config.accessibility_weight
        )

        return {
            "cultural_diversity": round(cultural_diversity, 2),
            "emotional_resonance": round(emotional_resonance, 2),
            "accessibility": round(accessibility, 2),
            "overall": round(overall, 2)
        }

    async def enrich_batch(self, symbols: List[Dict], batch_size: Optional[int] = None) -> List[Dict]:
        """
        Enrich a batch of symbols with rate limiting.

        Args:
            symbols: List of base symbol data
            batch_size: Number of concurrent LLM calls (uses config if None)

        Returns:
            List of enriched symbols
        """
        batch_size = batch_size or self.config.batch_size
        enriched_symbols = []

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]

            # Concurrent LLM calls for batch
            tasks = [self.enrich_symbol(symbol) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task exception: {str(result)}")
                elif result is not None:
                    enriched_symbols.append(result)

            # Rate limiting between batches
            if i + batch_size < len(symbols):
                await asyncio.sleep(self.config.rate_limit_delay)

        return enriched_symbols

    async def enrich_all_500(
        self,
        input_file: str = "symbol_database_base.json",
        output_file: str = "symbol_database_enriched.json"
    ) -> Tuple[List[Dict], Dict]:
        """
        Enrich all 500 symbols and save to JSON.

        Args:
            input_file: Path to base symbol database
            output_file: Path to save enriched database

        Returns:
            Tuple of (enriched_symbols, statistics)
        """
        input_path = Path(input_file)
        output_path = Path(output_file)

        # Load base symbols
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return [], {}

        with open(input_path, 'r') as f:
            data = json.load(f)
            base_symbols = data.get('symbols', [])

        logger.info(f"Loaded {len(base_symbols)} symbols from {input_file}")

        # Enrich in batches
        enriched_symbols = await self.enrich_batch(base_symbols, batch_size=self.config.batch_size)

        # Save enriched database
        output_data = {
            "metadata": {
                "version": "2.0",
                "enriched_date": datetime.now().isoformat(),
                "total_symbols": len(enriched_symbols),
                "enrichment_config": {
                    "llm_provider": self.config.llm_provider,
                    "llm_model": self.config.llm_model,
                    "batch_size": self.config.batch_size,
                    "dry_run": self.config.dry_run
                }
            },
            "symbols": enriched_symbols
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Saved {len(enriched_symbols)} enriched symbols to {output_file}")

        # Print statistics
        self._print_statistics()

        return enriched_symbols, self.enrichment_stats

    def _print_statistics(self):
        """Print enrichment statistics."""
        stats = self.enrichment_stats

        print("\n" + "="*60)
        print("ENRICHMENT STATISTICS")
        print("="*60)
        print(f"Total enriched: {stats['total_enriched']}")
        print(f"Successful: {stats['successful']}")
        print(f"Needs human review: {stats['needs_review']}")
        print(f"Failed: {stats['failed']}")

        if stats['quality_scores']:
            avg_quality = sum(stats['quality_scores']) / len(stats['quality_scores'])
            print(f"\nAverage quality score: {avg_quality:.2f}/10")
            print(f"Min quality: {min(stats['quality_scores']):.2f}/10")
            print(f"Max quality: {max(stats['quality_scores']):.2f}/10")

        print("\n" + "="*60)
        print("COST ESTIMATION")
        print("="*60)
        cost_per_symbol = 0.015  # Claude Sonnet ~$0.015 per symbol enrichment
        total_cost = stats['total_enriched'] * cost_per_symbol
        print(f"Symbols enriched: {stats['total_enriched']}")
        print(f"Cost per symbol: ${cost_per_symbol:.4f}")
        print(f"Total cost: ${total_cost:.2f}")
        print(f"Estimated time: ~{stats['total_enriched'] // 10 * 2} minutes (batched)")
        print("\n" + "="*60)


async def main():
    """Main entry point for batch enrichment."""
    import argparse

    parser = argparse.ArgumentParser(description="Enrich dream symbols with literary references")
    parser.add_argument(
        "--input",
        default="NeuroHood/dreams/symbol_database_base.json",
        help="Path to base symbol database"
    )
    parser.add_argument(
        "--output",
        default="NeuroHood/dreams/symbol_database_enriched.json",
        help="Path to save enriched database"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without making actual LLM calls"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of concurrent enrichments"
    )

    args = parser.parse_args()

    config = EnrichmentConfig(
        dry_run=args.dry_run,
        batch_size=args.batch_size
    )

    enricher = SymbolEnricher(config)

    if args.dry_run:
        print("\n⚠️  DRY-RUN MODE: No actual LLM calls will be made")
        print("This will generate mock enrichment data for testing.\n")

    enriched, stats = await enricher.enrich_all_500(
        input_file=args.input,
        output_file=args.output
    )

    return enriched, stats


if __name__ == "__main__":
    import sys
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
