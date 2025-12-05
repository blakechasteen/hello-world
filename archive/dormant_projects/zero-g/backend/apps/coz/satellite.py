"""
COZ (Cozbi's Organix Zentrum) Satellite App for Zero-G

Production management system for beekeeping and organic products business.

Features:
- Time tracking with efficiency analysis
- Cost tracking (materials, labor, overhead)
- Customer order management and revenue pipeline
- Production logging (output, waste, sales)
- Standard Operating Procedures (SOPs)
- Cross-parser intelligence (profit, efficiency, waste reduction, etc.)
- Daily brief generation with actionable insights

Integration with Zero-G:
- Docks to shared Loom (WarpSpace, YarnGraph, Rift)
- Registers 10+ tools for production management
- Uses semantic search for orders, inventory, SOPs
- Complete provenance via SpacetimeFabric
- Learns from outcomes via ReflectionBuffer

Created: 2025-11-22
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "zero-g" / "backend"))

# Import Zero-G satellite protocol
from apps.satellite_protocol import (
    SatelliteApp,
    AppManifest,
    GLevelRequirement,
    DataSource
)
from loom_core.protocols import LoomCore, RiftAction

# Import COZ components
from elle.coz.sync_manager import SyncManager


class COZSatellite(SatelliteApp):
    """
    COZ production management satellite

    Docks to Zero-G and exposes all COZ intelligence as Rift tools.
    """

    def __init__(self, coz_dir: str = "coz"):
        super().__init__()
        self.coz_dir = coz_dir
        self.sync: Optional[SyncManager] = None

    def get_manifest(self) -> AppManifest:
        """Return COZ app manifest"""
        return AppManifest(
            app_id="coz",
            app_name="Cozbi's Organix Zentrum",
            version="1.0.0",
            description=(
                "Production management system for beekeeping and organic products. "
                "Tracks time, costs, orders, production, and provides intelligence "
                "for profit optimization, waste reduction, and order fulfillment."
            ),

            # COZ needs G2 for SQL schema discovery (production database)
            required_g_level=GLevelRequirement.G2_LIFTOFF,

            # Loom requirements
            requires_warp_space=True,   # Semantic search for orders, inventory, SOPs
            requires_yarn_graph=True,   # Relationships: customer → order → product → batch
            requires_rift=True,         # Tool execution
            requires_spacetime_fabric=True,  # Complete audit trail
            requires_reflection_buffer=True,  # Learn from production outcomes
            requires_convergence_engine=False,  # Not needed yet (may add for optimization)
            requires_resonance_shed=False,     # Not needed (text-only for now)
            requires_thread_spinner=False,     # Not needed yet

            # Data sources
            data_sources=[
                DataSource(
                    source_id="coz_csv_files",
                    source_type="csv_files",
                    access_mode="read_write",
                    sensitivity="confidential",
                    connection_params={
                        "directory": self.coz_dir,
                        "files": [
                            "time_tracking.csv",
                            "cost_tracking.csv",
                            "customer_orders.csv",
                            "production_log.csv"
                        ]
                    }
                ),
                DataSource(
                    source_id="coz_markdown_files",
                    source_type="markdown_files",
                    access_mode="read_write",
                    sensitivity="internal",
                    connection_params={
                        "directory": self.coz_dir,
                        "files": [
                            "sops.md",
                            "kanban.csv",
                            "financials.md",
                            "schedule.md",
                            "research_notes.md",
                            "inventory.md"
                        ]
                    }
                )
            ],

            # Capabilities (tools COZ provides)
            capabilities=[
                # Core intelligence
                "coz.get_daily_brief",
                "coz.get_profit_analysis",
                "coz.get_efficiency_insights",
                "coz.get_cost_insights",

                # Advanced intelligence
                "coz.get_waste_reduction",
                "coz.get_order_fulfillment",
                "coz.get_customer_insights",
                "coz.get_production_efficiency",

                # Data queries
                "coz.get_pending_orders",
                "coz.get_critical_orders",
                "coz.get_inventory_status",
                "coz.get_production_summary",

                # Actions
                "coz.sync_all_files",
                "coz.export_data"
            ]
        )

    async def on_dock(self, loom: LoomCore) -> None:
        """
        Dock COZ to Zero-G Loom

        - Initialize SyncManager
        - Parse all COZ files
        - Register tools via Rift
        - Index content in WarpSpace
        """
        # Store Loom reference
        self.loom = loom

        # Initialize COZ SyncManager
        print(f"🔧 Initializing COZ SyncManager from {self.coz_dir}...")
        self.sync = SyncManager(coz_dir=self.coz_dir)

        # Parse all files
        print("📄 Parsing COZ files...")
        result = self.sync.parse_all()
        print(f"   ✅ Parsed {len(result.files_synced)} files: {', '.join(result.files_synced)}")

        if result.errors:
            print(f"   ⚠️  Errors: {result.errors}")

        # Register all COZ tools via Rift
        print("🔧 Registering COZ tools...")
        await self._register_tools()

        # Index COZ content in WarpSpace for semantic search
        print("🔍 Indexing COZ content in WarpSpace...")
        await self._index_content()

        # Build knowledge graph in YarnGraph
        print("🕸️  Building COZ knowledge graph...")
        await self._build_knowledge_graph()

        print(f"✅ COZ satellite docked successfully!")

    async def on_undock(self) -> None:
        """
        Undock COZ from Zero-G Loom

        - Save any pending state
        - Clean up resources
        """
        print("👋 Undocking COZ satellite...")

        # Save state if needed
        if self.sync:
            # Could export data here if needed
            pass

        # Clear references
        self.sync = None
        self.loom = None

        print("✅ COZ undocked successfully")

    # ==================== Tool Registration ====================

    async def _register_tools(self) -> None:
        """Register all COZ intelligence tools via Rift"""

        # Tool schemas (simplified for MVP)
        tools = [
            # Core intelligence
            ("coz.get_daily_brief", self._tool_get_daily_brief, {
                "description": "Get comprehensive daily brief with profit, orders, waste alerts",
                "parameters": {}
            }),

            ("coz.get_profit_analysis", self._tool_get_profit_analysis, {
                "description": "Analyze profit (time × cost vs revenue)",
                "parameters": {
                    "hourly_rate": {"type": "number", "default": 25.0}
                }
            }),

            ("coz.get_efficiency_insights", self._tool_get_efficiency_insights, {
                "description": "Time efficiency insights (overruns, category efficiency)",
                "parameters": {}
            }),

            ("coz.get_cost_insights", self._tool_get_cost_insights, {
                "description": "Cost optimization insights",
                "parameters": {}
            }),

            # Advanced intelligence
            ("coz.get_waste_reduction", self._tool_get_waste_reduction, {
                "description": "Waste reduction recommendations",
                "parameters": {}
            }),

            ("coz.get_order_fulfillment", self._tool_get_order_fulfillment, {
                "description": "Order fulfillment optimization (capacity, schedule)",
                "parameters": {}
            }),

            ("coz.get_customer_insights", self._tool_get_customer_insights, {
                "description": "Customer behavior insights (top customers, at-risk)",
                "parameters": {}
            }),

            ("coz.get_production_efficiency", self._tool_get_production_efficiency, {
                "description": "Production efficiency vs SOPs",
                "parameters": {}
            }),

            # Data queries
            ("coz.sync_all_files", self._tool_sync_all, {
                "description": "Re-parse all COZ files",
                "parameters": {}
            })
        ]

        # Register each tool
        for tool_name, executor, schema in tools:
            await self.loom.rift.register_tool(
                tool_name=tool_name,
                executor=executor,
                schema=schema
            )

        print(f"   ✅ Registered {len(tools)} tools")

    # ==================== Tool Implementations ====================

    async def _tool_get_daily_brief(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get daily brief"""
        brief = self.sync.get_daily_brief()
        return {
            "success": True,
            "brief": brief
        }

    async def _tool_get_profit_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get profit analysis"""
        hourly_rate = params.get("hourly_rate", 25.0)
        analysis = self.sync.get_profit_analysis(hourly_rate=hourly_rate)
        return {
            "success": True,
            "analysis": analysis
        }

    async def _tool_get_efficiency_insights(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get efficiency insights"""
        insights = self.sync.get_efficiency_insights()
        return {
            "success": True,
            "insights": insights
        }

    async def _tool_get_cost_insights(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get cost insights"""
        insights = self.sync.get_cost_insights()
        return {
            "success": True,
            "insights": insights
        }

    async def _tool_get_waste_reduction(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get waste reduction recommendations"""
        recommendations = self.sync.get_waste_reduction_recommendations()
        return {
            "success": True,
            "recommendations": recommendations
        }

    async def _tool_get_order_fulfillment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get order fulfillment optimization"""
        optimization = self.sync.get_order_fulfillment_optimization()
        return {
            "success": True,
            "optimization": optimization
        }

    async def _tool_get_customer_insights(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get customer insights"""
        insights = self.sync.get_customer_insights()
        return {
            "success": True,
            "insights": insights
        }

    async def _tool_get_production_efficiency(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get production efficiency analysis"""
        analysis = self.sync.get_production_efficiency_analysis()
        return {
            "success": True,
            "analysis": analysis
        }

    async def _tool_sync_all(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Re-parse all COZ files"""
        result = self.sync.parse_all()
        return {
            "success": result.status.value == "success",
            "files_synced": result.files_synced,
            "errors": result.errors
        }

    # ==================== Loom Integration ====================

    async def _index_content(self) -> None:
        """
        Index COZ content in WarpSpace for semantic search

        Enables queries like:
        - "Find orders due this week"
        - "Show me honey lip balm production batches"
        - "What are the SOPs for candle making?"
        """
        from loom_core.protocols import Thread

        # Index customer orders
        for order in self.sync.customer_orders.orders:
            thread = Thread(
                id=f"order_{order.order_id}",
                content=f"Order {order.order_id}: {order.quantity} units of {order.product} "
                       f"for {order.customer_name}, due {order.due_date.strftime('%Y-%m-%d')}, "
                       f"status: {order.status}, total: ${order.total_value:.2f}",
                metadata={
                    "type": "customer_order",
                    "order_id": order.order_id,
                    "customer": order.customer_name,
                    "product": order.product,
                    "status": order.status,
                    "priority": order.fulfillment_priority
                }
            )
            await self.loom.warp_space.index_thread(thread)

        # Index SOPs
        for sop in self.sync.sops.sops:
            thread = Thread(
                id=f"sop_{sop.sop_id}",
                content=f"SOP {sop.sop_id}: {sop.title}. "
                       f"Estimated time: {sop.estimated_time_hours:.1f}h. "
                       f"Steps: {', '.join(sop.steps[:3])}...",
                metadata={
                    "type": "sop",
                    "sop_id": sop.sop_id,
                    "title": sop.title,
                    "category": sop.category,
                    "estimated_hours": sop.estimated_time_hours
                }
            )
            await self.loom.warp_space.index_thread(thread)

        # Index production entries
        for entry in self.sync.production_log.entries:
            thread = Thread(
                id=f"production_{entry.date.isoformat()}_{entry.product_name}",
                content=f"Production on {entry.date.strftime('%Y-%m-%d')}: "
                       f"{entry.units_produced} units of {entry.product_name}, "
                       f"{entry.units_sold} sold, {entry.waste_units} wasted",
                metadata={
                    "type": "production",
                    "product": entry.product_name,
                    "date": entry.date.isoformat(),
                    "produced": entry.units_produced,
                    "sold": entry.units_sold,
                    "waste": entry.waste_units
                }
            )
            await self.loom.warp_space.index_thread(thread)

        print(f"   ✅ Indexed {len(self.sync.customer_orders.orders)} orders, "
              f"{len(self.sync.sops.sops)} SOPs, "
              f"{len(self.sync.production_log.entries)} production entries")

    async def _build_knowledge_graph(self) -> None:
        """
        Build knowledge graph in YarnGraph

        Graph structure:
        - customer → (PLACED) → order
        - order → (FOR_PRODUCT) → product
        - product → (HAS_SOP) → sop
        - order → (FULFILLED_BY) → production_batch
        - production_batch → (CONSUMED) → materials
        """
        from loom_core.protocols import YarnNode, YarnEdge

        # Add customer nodes
        customers = {}
        for order in self.sync.customer_orders.orders:
            if order.customer_name not in customers:
                node = YarnNode(
                    id=f"customer_{order.customer_name.replace(' ', '_')}",
                    entity_type="customer",
                    properties={"name": order.customer_name}
                )
                await self.loom.yarn_graph.add_node(node)
                customers[order.customer_name] = node.id

        # Add order nodes + edges
        for order in self.sync.customer_orders.orders:
            # Order node
            order_node = YarnNode(
                id=f"order_{order.order_id}",
                entity_type="order",
                properties={
                    "order_id": order.order_id,
                    "product": order.product,
                    "quantity": order.quantity,
                    "status": order.status,
                    "total_value": order.total_value
                }
            )
            await self.loom.yarn_graph.add_node(order_node)

            # Customer → Order edge
            customer_id = customers[order.customer_name]
            edge = YarnEdge(
                source_id=customer_id,
                target_id=order_node.id,
                relationship_type="PLACED",
                weight=1.0
            )
            await self.loom.yarn_graph.add_edge(edge)

        # Add product nodes
        products = set(order.product for order in self.sync.customer_orders.orders)
        for product_name in products:
            product_node = YarnNode(
                id=f"product_{product_name.replace(' ', '_')}",
                entity_type="product",
                properties={"name": product_name}
            )
            await self.loom.yarn_graph.add_node(product_node)

            # Order → Product edges
            for order in self.sync.customer_orders.orders:
                if order.product == product_name:
                    edge = YarnEdge(
                        source_id=f"order_{order.order_id}",
                        target_id=product_node.id,
                        relationship_type="FOR_PRODUCT",
                        weight=1.0
                    )
                    await self.loom.yarn_graph.add_edge(edge)

        print(f"   ✅ Built graph: {len(customers)} customers, "
              f"{len(self.sync.customer_orders.orders)} orders, "
              f"{len(products)} products")


# ==================== Factory Function ====================

def create_coz_satellite(coz_dir: str = "coz") -> COZSatellite:
    """
    Factory function to create COZ satellite

    Usage:
        coz = create_coz_satellite()
        await orbit_manager.dock_app(coz)
    """
    return COZSatellite(coz_dir=coz_dir)
