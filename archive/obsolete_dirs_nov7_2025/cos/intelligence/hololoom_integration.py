"""
HoloLoom COS - Memory Integration

Converts COS events into HoloLoom MemoryShards and enables
agentic reasoning over business data.

Philosophy:
- Events are ephemeral facts (discrete data points)
- Memory is semantic understanding (connected knowledge)
- Agentic reasoning enables strategic insights
"""

import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HoloLoom.documentation.types import MemoryShard, Query
from HoloLoom.agentic.core import AgenticOrchestrator, ReasoningMode
from HoloLoom.config import Config

from cos.core.types import Event, EventType, DailySummary, WeeklySummary
from cos.core.storage import EventStore


class COSMemoryBridge:
    """
    Bridge between COS events and HoloLoom memory.

    Converts business events into semantic memories that HoloLoom
    can reason over.
    """

    def __init__(self, store: EventStore):
        self.store = store

    def event_to_shard(self, event: Event) -> MemoryShard:
        """
        Convert COS event to HoloLoom MemoryShard.

        Each event becomes a memory that captures:
        - What happened (content)
        - When it happened (timestamp)
        - Business context (metadata)
        - Relationships (entities, categories)
        """
        # Build rich content description
        content = self._build_content(event)

        # Extract entities and motifs
        entities = self._extract_entities(event)
        motifs = self._extract_motifs(event)

        # Build metadata
        metadata = {
            'event_id': event.id,
            'event_type': event.type.value,
            'timestamp': event.timestamp.isoformat(),
            'source': event.source.value,
            'verified': event.verified,
            'confidence': float(event.confidence) if event.confidence else None,
        }

        if event.amount:
            metadata['amount'] = float(event.amount)
        if event.quantity:
            metadata['quantity'] = float(event.quantity)
        if event.product_line:
            metadata['product_line'] = event.product_line.value
        if event.category:
            metadata['category'] = event.category.value

        return MemoryShard(
            content=content,
            source=f"COS_{event.type.value}_{event.id}",
            metadata=metadata,
            entities=entities,
            tags=[event.type.value] + (event.tags if event.tags else []),
        )

    def _build_content(self, event: Event) -> str:
        """Build human-readable content from event"""
        if event.type == EventType.TASK:
            hours = event.quantity or 0
            cost = abs(event.amount) if event.amount else 0
            output = event.parsed_data.get('output', '')

            content = f"Worked on {event.item} for {hours} hours (${cost:.2f} labor cost)"
            if output:
                content += f", produced {output} units"
            return content

        elif event.type == EventType.SALE:
            revenue = event.amount or 0
            qty = abs(event.quantity) if event.quantity else None

            content = f"Sold {event.item} for ${revenue:.2f}"
            if qty:
                content += f" ({qty} units)"
            return content

        elif event.type == EventType.PURCHASE:
            cost = abs(event.amount) if event.amount else 0
            qty = event.quantity or None
            vendor = event.parsed_data.get('vendor', '')

            content = f"Purchased {event.item} for ${cost:.2f}"
            if qty and event.unit:
                content += f" ({qty} {event.unit})"
            if vendor:
                content += f" from {vendor}"
            return content

        elif event.type == EventType.INVENTORY:
            qty = event.quantity or 0
            content = f"Inventory: {qty} {event.item}"
            return content

        else:
            # NOTE, REVIEW, PLAN, GOAL
            return event.raw_input

    def _extract_entities(self, event: Event) -> List[str]:
        """Extract business entities from event"""
        entities = []

        if event.item:
            entities.append(event.item)

        if event.product_line:
            entities.append(event.product_line.value)

        if event.parsed_data.get('vendor'):
            entities.append(event.parsed_data['vendor'])

        return entities

    def _extract_motifs(self, event: Event) -> List[str]:
        """Extract business motifs (patterns) from event"""
        motifs = []

        # Event type is a motif
        motifs.append(event.type.value)

        # Category is a motif
        if event.category:
            motifs.append(event.category.value)

        # Add business-specific motifs
        if event.type == EventType.TASK:
            if event.quantity and event.quantity > 8:
                motifs.append("long_work_session")

        elif event.type == EventType.PURCHASE:
            if event.amount and abs(event.amount) > 100:
                motifs.append("large_purchase")

        elif event.type == EventType.SALE:
            if event.amount and event.amount > 200:
                motifs.append("large_sale")

        return motifs

    async def sync_to_hololoom(
        self,
        orchestrator: AgenticOrchestrator,
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> int:
        """
        Sync COS events to HoloLoom memory.

        Args:
            orchestrator: HoloLoom agentic orchestrator
            since: Only sync events after this time (None = all)
            limit: Max events to sync

        Returns:
            Number of events synced
        """
        # Query events
        events = await self.store.query(
            start_date=since,
            verified_only=True,  # Only sync verified data
            limit=limit
        )

        # Convert to shards
        shards = [self.event_to_shard(e) for e in events]

        # Store in HoloLoom memory
        # TODO: Orchestrator needs a method to bulk-add shards
        # For now, we'd need to add to the knowledge graph directly

        return len(shards)


class COSAgenticInterface:
    """
    Agentic reasoning interface for COS business intelligence.

    Provides high-level business queries using HoloLoom's
    agentic reasoning capabilities.
    """

    def __init__(
        self,
        store: EventStore,
        config: Optional[Config] = None
    ):
        self.store = store
        self.config = config or Config.fused()
        self.bridge = COSMemoryBridge(store)

        # Initialize HoloLoom orchestrator
        # TODO: Need to create with business memory shards
        self.orchestrator = None  # Will be initialized async

    async def initialize(self):
        """Initialize HoloLoom orchestrator with business context"""
        # Convert recent events to shards
        events = await self.store.query(limit=100, verified_only=True)
        shards = [self.bridge.event_to_shard(e) for e in events]

        # Create orchestrator
        # TODO: AgenticOrchestrator needs shards parameter
        # For now, placeholder

        # Load business plan and SOPs as additional context
        await self._load_business_documents()

    async def _load_business_documents(self):
        """Load business plan, SOPs, etc. into memory"""
        # TODO: Use SpinningWheel to ingest:
        # - 90-day timeline
        # - Business plan
        # - Recipes/SOPs
        # - Budget
        pass

    async def ask_strategic_question(
        self,
        question: str,
        mode: ReasoningMode = ReasoningMode.RESEARCH
    ) -> str:
        """
        Ask a strategic business question.

        Uses HoloLoom agentic reasoning to analyze business data
        and provide insights.

        Examples:
        - "What should I focus on this week?"
        - "Is bread profitable enough?"
        - "Am I on track for my Week 4 goals?"
        - "Where are my bottlenecks?"
        """
        if not self.orchestrator:
            await self.initialize()

        query = Query(text=question)
        result = await self.orchestrator.reason(query, mode=mode)

        return result.response

    async def get_insights(self, timeframe: str = "week") -> Dict[str, Any]:
        """
        Get automated business insights.

        Analyzes recent data and provides:
        - Performance vs targets
        - Trend analysis
        - Bottleneck detection
        - Recommendations
        """
        if timeframe == "today":
            summary = await self.store.get_daily_summary(datetime.now())
        elif timeframe == "week":
            week = datetime.now().strftime('%Y-W%W')
            summary = await self.store.get_weekly_summary(week)
        else:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        # Analyze performance
        insights = {
            'summary': summary,
            'alerts': await self._generate_alerts(summary),
            'recommendations': await self._generate_recommendations(summary),
            'trends': await self._analyze_trends(timeframe),
        }

        return insights

    async def _generate_alerts(self, summary) -> List[Dict[str, Any]]:
        """Generate alerts based on business rules"""
        alerts = []

        # Burnout risk
        if hasattr(summary, 'hours_worked') and summary.hours_worked > 50:
            alerts.append({
                'type': 'BURNOUT_RISK',
                'severity': 'CRITICAL',
                'message': f'Working {summary.hours_worked} hours this week - unsustainable!',
                'action': 'Reduce product lines or hire help'
            })
        elif hasattr(summary, 'hours_worked') and summary.hours_worked > 40:
            alerts.append({
                'type': 'BURNOUT_WARNING',
                'severity': 'MEDIUM',
                'message': f'Working {summary.hours_worked} hours this week',
                'action': 'Monitor workload, consider scaling back'
            })

        # Low profit margin
        if hasattr(summary, 'profit_margin'):
            margin = summary.profit_margin()
            if margin < 20:
                alerts.append({
                    'type': 'LOW_MARGIN',
                    'severity': 'HIGH',
                    'message': f'Profit margin only {margin:.1f}% (target: 75%+)',
                    'action': 'Raise prices or reduce costs'
                })

        # Negative profit
        if hasattr(summary, 'profit') and summary.profit < 0:
            alerts.append({
                'type': 'NEGATIVE_PROFIT',
                'severity': 'CRITICAL',
                'message': f'Losing ${abs(summary.profit):.2f}!',
                'action': 'Review costs and pricing immediately'
            })

        # Revenue below target
        if hasattr(summary, 'revenue_target') and summary.revenue_target:
            variance_pct = summary.variance_pct()
            if variance_pct and variance_pct < -20:
                alerts.append({
                    'type': 'REVENUE_SHORTFALL',
                    'severity': 'HIGH',
                    'message': f'Revenue {abs(variance_pct):.1f}% below target',
                    'action': 'Increase marketing or production'
                })

        return alerts

    async def _generate_recommendations(self, summary) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Product performance analysis
        from cos.core.types import ProductLine

        product_performances = []
        for product in ProductLine:
            perf = await self.store.get_product_performance(product)
            if perf.revenue > 0:
                product_performances.append(perf)

        if product_performances:
            # Sort by profit per hour
            product_performances.sort(
                key=lambda p: p.profit_per_hour(),
                reverse=True
            )

            best = product_performances[0]
            worst = product_performances[-1] if len(product_performances) > 1 else None

            recommendations.append(
                f"Focus on {best.product_line.value} (${best.profit_per_hour():.2f}/hr)"
            )

            if worst and worst.profit_per_hour() < 10:
                recommendations.append(
                    f"Consider stopping {worst.product_line.value} (only ${worst.profit_per_hour():.2f}/hr)"
                )

        return recommendations

    async def _analyze_trends(self, timeframe: str) -> Dict[str, Any]:
        """Analyze trends over time"""
        # TODO: Implement trend analysis
        # - Revenue trend (up/down/flat)
        # - Hours trend (increasing = burnout risk)
        # - Margin trend (improving/declining)
        return {}


async def demo():
    """Demo the HoloLoom integration"""
    print("\n" + "="*60)
    print("HoloLoom COS - Memory Integration Demo")
    print("="*60)

    # Create store
    store = EventStore("test_cos.db")

    # Test 1: Convert events to shards
    print("\n\nTEST 1: Event → MemoryShard Conversion")
    print("-"*60)

    events = await store.query(limit=5, verified_only=True)
    bridge = COSMemoryBridge(store)

    for event in events:
        shard = bridge.event_to_shard(event)
        print(f"\nEvent: {event.raw_input[:50]}")
        print(f"  Content: {shard.content}")
        print(f"  Entities: {shard.entities}")
        print(f"  Tags: {shard.tags}")

    # Test 2: Business insights
    print("\n\nTEST 2: Business Insights")
    print("-"*60)

    interface = COSAgenticInterface(store)
    insights = await interface.get_insights("week")

    print(f"\nAlerts: {len(insights['alerts'])}")
    for alert in insights['alerts']:
        print(f"  [{alert['severity']}] {alert['message']}")
        print(f"    → {alert['action']}")

    print(f"\nRecommendations:")
    for rec in insights['recommendations']:
        print(f"  • {rec}")

    print("\n\n" + "="*60)
    print("✅ Integration Demo Complete")
    print("="*60)


if __name__ == '__main__':
    asyncio.run(demo())
