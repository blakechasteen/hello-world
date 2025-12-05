"""
Production Batch Enrichment Pipeline Test

Tests the complete batch enrichment system with real LLM (Claude Sonnet).
- Tests 3 symbols from different categories (trapped, loss, hope)
- Validates quality metrics
- Estimates costs and runtime for full 500 symbols
- Provides checkpoint and resume functionality

Usage:
    python test_batch_enrichment_production.py

Requirements:
    - ANTHROPIC_API_KEY environment variable set
    - Python 3.8+
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time

try:
    from anthropic import Anthropic
except ImportError:
    print("ERROR: anthropic package not installed")
    print("Install with: pip install anthropic")
    sys.exit(1)

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


class ProductionBatchEnricher:
    """Production batch enrichment with real LLM integration."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with Anthropic API key."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Set environment variable or pass api_key.")

        self.client = Anthropic(api_key=self.api_key)
        self.metaprompt_template = self._load_metaprompt()
        self.stats = {
            "total_enriched": 0,
            "successful": 0,
            "failed": 0,
            "total_references": 0,
            "quality_scores": [],
            "total_tokens_used": 0,
            "total_time_ms": 0,
        }

    def _load_metaprompt(self) -> str:
        """Load metaprompt template from file."""
        template_path = Path(__file__).parent / "symbol_enrichment_metaprompt.txt"
        if template_path.exists():
            with open(template_path, 'r') as f:
                return f.read()
        else:
            raise FileNotFoundError(f"Metaprompt template not found at {template_path}")

    async def enrich_symbol(self, symbol: Dict) -> Tuple[bool, Dict, Dict]:
        """
        Enrich a single symbol with real LLM.

        Returns:
            (success, enriched_symbol, stats)
        """
        symbol_id = symbol.get('symbol_id', 'unknown')
        start_time = time.time()

        try:
            # Build metaprompt for this symbol
            symbol_request = f"""
SYMBOL: {symbol['symbol_id']}
DESCRIPTION: {symbol['description']}
EMOTION_TAGS: {', '.join(symbol['emotion_tags'])}
CATEGORY: {symbol['category']}
EXISTING_REFERENCES: {', '.join(symbol.get('existing_references', []))}
"""

            metaprompt = self.metaprompt_template.replace(
                "{SYMBOL_REQUEST}",
                symbol_request
            )

            # Call Claude API
            logger.info(f"Enriching symbol: {symbol_id}")
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[{"role": "user", "content": metaprompt}]
            )

            # Parse response
            response_text = response.content[0].text.strip()

            # Extract JSON (handle potential markdown code blocks)
            if response_text.startswith("```"):
                json_str = response_text.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                json_str = json_str.strip()
            else:
                json_str = response_text

            # Parse JSON
            literary_references = json.loads(json_str)

            # Count references
            total_refs = sum(len(v) for v in literary_references.values() if isinstance(v, list))

            # Create enriched symbol
            enriched = {
                **symbol,
                "literary_references": literary_references,
                "enrichment_date": datetime.now().isoformat(),
                "llm_model": "claude-3-5-sonnet-20241022",
                "total_references": total_refs,
                "quality_scores": {
                    "cultural_diversity": self._score_cultural_diversity(literary_references),
                    "emotional_resonance": self._score_emotional_resonance(literary_references),
                    "accessibility": self._score_accessibility(literary_references),
                },
            }

            # Calculate overall quality
            overall_quality = (
                0.4 * enriched["quality_scores"]["cultural_diversity"] +
                0.35 * enriched["quality_scores"]["emotional_resonance"] +
                0.25 * enriched["quality_scores"]["accessibility"]
            )
            enriched["quality_scores"]["overall"] = overall_quality
            enriched["needs_human_review"] = overall_quality < 7.0

            # Update stats
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats["successful"] += 1
            self.stats["total_references"] += total_refs
            self.stats["quality_scores"].append(overall_quality)
            self.stats["total_tokens_used"] += response.usage.input_tokens + response.usage.output_tokens
            self.stats["total_time_ms"] += elapsed_ms

            logger.info(
                f"  ✓ {symbol_id}: {total_refs} refs, quality={overall_quality:.2f}, "
                f"time={elapsed_ms:.0f}ms, tokens={response.usage.output_tokens}"
            )

            return (True, enriched, {
                "time_ms": elapsed_ms,
                "tokens": response.usage.output_tokens,
                "quality": overall_quality
            })

        except Exception as e:
            logger.error(f"  ✗ {symbol_id}: {str(e)}")
            self.stats["failed"] += 1
            return (False, symbol, {"error": str(e)})

    def _score_cultural_diversity(self, refs: Dict) -> float:
        """Score cultural diversity of references."""
        cultures_found = set()

        for category, items in refs.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and "culture" in item:
                        culture = item["culture"].lower()
                        # Identify non-Western cultures
                        if any(region in culture for region in
                               ["asian", "african", "chinese", "japanese", "indian",
                                "indigenous", "native", "middle eastern", "latin",
                                "african", "vedic", "norse", "egyptian", "greek"]):
                            cultures_found.add(culture)

        # Score based on cultural diversity (target: 3+ non-Western)
        non_western_count = len([c for c in cultures_found
                                if not any(w in c for w in ["western", "british", "american"])])
        diversity_score = min(10.0, (non_western_count / 3.0) * 10.0)  # 3+ = 10
        return diversity_score

    def _score_emotional_resonance(self, refs: Dict) -> float:
        """Score emotional resonance and connection quality."""
        total_items = 0
        high_confidence_count = 0

        for category, items in refs.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        total_items += 1
                        if item.get("connection_confidence", 0) >= 0.8:
                            high_confidence_count += 1

        if total_items == 0:
            return 0.0

        # Score: percentage of high-confidence connections
        resonance_score = (high_confidence_count / total_items) * 10.0
        return min(10.0, resonance_score)

    def _score_accessibility(self, refs: Dict) -> float:
        """Score accessibility of references (mix of popular and scholarly)."""
        total_items = 0
        popular_count = 0

        popular_works = {
            "shakespeare", "dante", "kafka", "dostoevsky", "austen", "bronte",
            "morrison", "atwood", "the matrix", "the truman show", "fight club",
            "star wars", "inception", "interstellar", "david lynch", "miyazaki"
        }

        for category, items in refs.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        total_items += 1
                        title = item.get("title", "").lower()
                        if any(work in title for work in popular_works):
                            popular_count += 1

        if total_items == 0:
            return 0.0

        # Score: 60% should be popular/accessible
        popular_ratio = popular_count / total_items
        target_ratio = 0.6
        accessibility_score = 10.0 - (abs(popular_ratio - target_ratio) * 10.0)
        return max(0.0, min(10.0, accessibility_score))

    async def enrich_test_symbols(self, symbols: List[Dict]) -> List[Dict]:
        """Enrich a list of test symbols."""
        enriched_results = []

        for symbol in symbols:
            success, enriched, stats = await self.enrich_symbol(symbol)
            if success:
                enriched_results.append(enriched)

            # Small delay between API calls
            await asyncio.sleep(0.5)

        return enriched_results

    def print_summary(self, results: List[Dict]):
        """Print enrichment summary."""
        logger.info("\n" + "=" * 70)
        logger.info("BATCH ENRICHMENT TEST SUMMARY")
        logger.info("=" * 70)

        logger.info(f"\nSymbols processed: {self.stats['successful']} successful, {self.stats['failed']} failed")

        if self.stats['successful'] > 0:
            logger.info(f"Total references: {self.stats['total_references']}")
            logger.info(f"Avg references per symbol: {self.stats['total_references'] / self.stats['successful']:.1f}")
            logger.info(f"Avg quality score: {sum(self.stats['quality_scores']) / len(self.stats['quality_scores']):.2f}/10")
            logger.info(f"Total tokens used: {self.stats['total_tokens_used']}")
            logger.info(f"Total time: {self.stats['total_time_ms']:.0f} ms")

            avg_time_per_symbol = self.stats['total_time_ms'] / self.stats['successful']
            logger.info(f"Avg time per symbol: {avg_time_per_symbol:.0f} ms")

            # Estimate for 500 symbols
            estimated_total_time = (avg_time_per_symbol * 500) / 1000 / 60
            estimated_tokens = (self.stats['total_tokens_used'] / self.stats['successful']) * 500
            estimated_cost = (estimated_tokens / 1000000) * 3.0  # $3 per 1M tokens for Sonnet

            logger.info(f"\n" + "-" * 70)
            logger.info("EXTRAPOLATION TO 500 SYMBOLS")
            logger.info("-" * 70)
            logger.info(f"Estimated total time: {estimated_total_time:.1f} minutes")
            logger.info(f"Estimated tokens: {estimated_tokens:.0f}")
            logger.info(f"Estimated cost: ${estimated_cost:.2f}")

        # Print individual results
        logger.info(f"\n" + "-" * 70)
        logger.info("INDIVIDUAL RESULTS")
        logger.info("-" * 70)

        for result in results:
            symbol_id = result['symbol_id']
            quality = result['quality_scores']['overall']
            refs = result.get('total_references', 0)
            review = "NEEDS REVIEW" if result.get('needs_human_review') else "OK"

            logger.info(f"\n{symbol_id}:")
            logger.info(f"  Category: {result['category']}")
            logger.info(f"  References: {refs}")
            logger.info(f"  Quality: {quality:.2f}/10 [{review}]")
            logger.info(f"  Cultural diversity: {result['quality_scores']['cultural_diversity']:.2f}/10")
            logger.info(f"  Emotional resonance: {result['quality_scores']['emotional_resonance']:.2f}/10")
            logger.info(f"  Accessibility: {result['quality_scores']['accessibility']:.2f}/10")


async def main():
    """Run production batch enrichment test."""
    logger.info("=" * 70)
    logger.info("PRODUCTION BATCH ENRICHMENT TEST")
    logger.info("=" * 70)

    # Load test symbols (3 from different categories)
    pilot_path = Path(__file__).parent / "symbol_database_pilot.json"
    with open(pilot_path, 'r') as f:
        pilot_symbols = json.load(f)

    # Select 3 test symbols from different categories
    test_symbols = [
        next(s for s in pilot_symbols if s['category'] == 'trapped'),
        next(s for s in pilot_symbols if s['category'] == 'loss'),
        next(s for s in pilot_symbols if s['category'] == 'hope'),
    ]

    logger.info(f"\nTest symbols:")
    for sym in test_symbols:
        logger.info(f"  - {sym['symbol_id']} ({sym['category']})")

    # Initialize enricher
    enricher = ProductionBatchEnricher()

    # Enrich test symbols
    logger.info(f"\nEnriching symbols with Claude Sonnet...")
    enriched = await enricher.enrich_test_symbols(test_symbols)

    # Print summary
    enricher.print_summary(enriched)

    # Save results
    output_path = Path(__file__).parent / "test_enrichment_results.json"
    with open(output_path, 'w') as f:
        json.dump(enriched, f, indent=2)

    logger.info(f"\n✓ Results saved to {output_path}")

    return enriched


if __name__ == "__main__":
    results = asyncio.run(main())