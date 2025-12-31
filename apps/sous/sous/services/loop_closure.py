#!/usr/bin/env python3
"""
Loop Closure Tracker for Phase 3

Tracks circular economy flows in the farm network:
- Food scraps → composting → farms
- Waste reduction and diversion
- Carbon sequestration
- Nutrient cycling
- Circularity metrics
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Set, Tuple
from decimal import Decimal
from collections import defaultdict
from enum import Enum

from sous.models.farm import Farm
from sous.models.organization import Organization


class MaterialType(Enum):
    """Types of materials in circular flow"""
    FOOD_SCRAPS = "food_scraps"
    COMPOST = "compost"
    ORGANIC_WASTE = "organic_waste"
    ANIMAL_FEED = "animal_feed"
    PACKAGING = "packaging"


class FlowStage(Enum):
    """Stages in circular material flow"""
    GENERATION = "generation"  # Waste created
    COLLECTION = "collection"  # Picked up
    PROCESSING = "processing"  # Composted/processed
    DISTRIBUTION = "distribution"  # Returned to farms
    APPLICATION = "application"  # Applied to soil


@dataclass
class MaterialFlow:
    """Record of material moving through the system"""
    flow_id: str

    # Source and destination
    source_org_id: str
    source_org_name: str
    destination_org_id: str
    destination_org_name: str

    # Material details
    material_type: MaterialType
    quantity_lbs: Decimal
    stage: FlowStage

    # Timing
    generation_date: date
    collection_date: Optional[date] = None
    processing_date: Optional[date] = None
    application_date: Optional[date] = None

    # Impact
    carbon_avoided_lbs: Decimal = Decimal("0.0")
    nutrients_recovered_lbs: Dict[str, Decimal] = field(default_factory=dict)

    # Metadata
    notes: str = ""
    verified: bool = False


@dataclass
class CircularityMetrics:
    """Metrics for circularity performance"""
    org_id: str
    org_name: str
    period_start: date
    period_end: date

    # Material flows
    food_scraps_generated_lbs: Decimal = Decimal("0.0")
    food_scraps_diverted_lbs: Decimal = Decimal("0.0")
    compost_produced_lbs: Decimal = Decimal("0.0")
    compost_applied_lbs: Decimal = Decimal("0.0")

    # Diversion rate
    waste_diversion_rate: float = 0.0  # Percentage diverted from landfill

    # Carbon impact
    carbon_avoided_lbs: Decimal = Decimal("0.0")
    carbon_sequestered_lbs: Decimal = Decimal("0.0")
    net_carbon_impact_lbs: Decimal = Decimal("0.0")

    # Nutrient recovery
    nitrogen_recovered_lbs: Decimal = Decimal("0.0")
    phosphorus_recovered_lbs: Decimal = Decimal("0.0")
    potassium_recovered_lbs: Decimal = Decimal("0.0")

    # Circularity score (0-100)
    circularity_score: float = 0.0


@dataclass
class LoopConnection:
    """Connection between organizations in circular loop"""
    connection_id: str

    # Organizations
    donor_org_id: str  # Provides food scraps
    recipient_org_id: str  # Receives food scraps (farm)

    # Capacity
    weekly_capacity_lbs: Decimal = Decimal("100.0")
    current_utilization_lbs: Decimal = Decimal("0.0")

    # Status
    active: bool = True
    established_date: date = field(default_factory=date.today)

    # Performance
    total_materials_transferred_lbs: Decimal = Decimal("0.0")
    total_carbon_avoided_lbs: Decimal = Decimal("0.0")

    def get_utilization_rate(self) -> float:
        """Calculate capacity utilization"""
        if self.weekly_capacity_lbs == 0:
            return 0.0
        return float(self.current_utilization_lbs / self.weekly_capacity_lbs * 100)

    def has_capacity(self, quantity_lbs: Decimal) -> bool:
        """Check if connection can handle additional flow"""
        return (self.current_utilization_lbs + quantity_lbs) <= self.weekly_capacity_lbs


class LoopClosureTracker:
    """
    Circular economy tracking system.

    Monitors material flows, calculates impacts, and measures circularity.
    """

    def __init__(self):
        # Core data
        self.organizations: Dict[str, Organization] = {}
        self.material_flows: Dict[str, MaterialFlow] = {}
        self.loop_connections: Dict[str, LoopConnection] = {}

        # Indexes
        self.flows_by_source: Dict[str, List[str]] = defaultdict(list)
        self.flows_by_destination: Dict[str, List[str]] = defaultdict(list)
        self.connections_by_donor: Dict[str, List[str]] = defaultdict(list)
        self.connections_by_recipient: Dict[str, List[str]] = defaultdict(list)

    # ========================================================================
    # Organization Management
    # ========================================================================

    def register_organization(self, org: Organization):
        """Register an organization in the loop"""
        self.organizations[org.org_id] = org

    def get_organization(self, org_id: str) -> Optional[Organization]:
        """Get organization by ID"""
        return self.organizations.get(org_id)

    # ========================================================================
    # Loop Connections
    # ========================================================================

    def create_loop_connection(
        self,
        donor_org_id: str,
        recipient_org_id: str,
        weekly_capacity_lbs: Decimal = Decimal("100.0")
    ) -> LoopConnection:
        """Create a circular loop connection between organizations"""
        donor = self.get_organization(donor_org_id)
        recipient = self.get_organization(recipient_org_id)

        if not donor or not recipient:
            raise ValueError("Both organizations must be registered")

        # Verify recipient can accept materials
        if isinstance(recipient, Farm):
            if not recipient.accepts_food_scraps:
                raise ValueError(f"{recipient.name} does not accept food scraps")

        connection_id = f"loop_{donor_org_id}_{recipient_org_id}"

        connection = LoopConnection(
            connection_id=connection_id,
            donor_org_id=donor_org_id,
            recipient_org_id=recipient_org_id,
            weekly_capacity_lbs=weekly_capacity_lbs
        )

        self.loop_connections[connection_id] = connection
        self.connections_by_donor[donor_org_id].append(connection_id)
        self.connections_by_recipient[recipient_org_id].append(connection_id)

        return connection

    def get_loop_connection(self, connection_id: str) -> Optional[LoopConnection]:
        """Get loop connection by ID"""
        return self.loop_connections.get(connection_id)

    def get_organization_connections(
        self,
        org_id: str,
        as_donor: bool = True
    ) -> List[LoopConnection]:
        """Get all loop connections for an organization"""
        if as_donor:
            connection_ids = self.connections_by_donor.get(org_id, [])
        else:
            connection_ids = self.connections_by_recipient.get(org_id, [])

        return [
            self.loop_connections[cid] for cid in connection_ids
            if cid in self.loop_connections
        ]

    def find_available_recipient(
        self,
        donor_org_id: str,
        quantity_lbs: Decimal
    ) -> Optional[LoopConnection]:
        """Find a recipient farm with capacity for material"""
        connections = self.get_organization_connections(donor_org_id, as_donor=True)

        # Filter active connections with capacity
        available = [
            c for c in connections
            if c.active and c.has_capacity(quantity_lbs)
        ]

        if not available:
            return None

        # Return connection with most available capacity
        return max(available, key=lambda c: c.weekly_capacity_lbs - c.current_utilization_lbs)

    # ========================================================================
    # Material Flow Tracking
    # ========================================================================

    def record_material_generation(
        self,
        source_org_id: str,
        material_type: MaterialType,
        quantity_lbs: Decimal,
        generation_date: date
    ) -> MaterialFlow:
        """Record material generated (e.g., food scraps created)"""
        source_org = self.get_organization(source_org_id)
        if not source_org:
            raise ValueError(f"Organization {source_org_id} not found")

        flow_id = f"flow_{source_org_id}_{generation_date.isoformat()}_{material_type.value}"

        flow = MaterialFlow(
            flow_id=flow_id,
            source_org_id=source_org_id,
            source_org_name=source_org.name,
            destination_org_id="",  # Will be set during collection
            destination_org_name="",
            material_type=material_type,
            quantity_lbs=quantity_lbs,
            stage=FlowStage.GENERATION,
            generation_date=generation_date
        )

        self.material_flows[flow_id] = flow
        self.flows_by_source[source_org_id].append(flow_id)

        return flow

    def record_material_collection(
        self,
        flow_id: str,
        destination_org_id: str,
        collection_date: date,
        connection_id: Optional[str] = None
    ):
        """Record material collected from source"""
        flow = self.material_flows.get(flow_id)
        if not flow:
            raise ValueError(f"Flow {flow_id} not found")

        dest_org = self.get_organization(destination_org_id)
        if not dest_org:
            raise ValueError(f"Destination org {destination_org_id} not found")

        flow.destination_org_id = destination_org_id
        flow.destination_org_name = dest_org.name
        flow.collection_date = collection_date
        flow.stage = FlowStage.COLLECTION

        self.flows_by_destination[destination_org_id].append(flow_id)

        # Update connection utilization
        if connection_id:
            connection = self.get_loop_connection(connection_id)
            if connection:
                connection.current_utilization_lbs += flow.quantity_lbs
                connection.total_materials_transferred_lbs += flow.quantity_lbs

    def record_material_processing(
        self,
        flow_id: str,
        processing_date: date,
        compost_yield_lbs: Optional[Decimal] = None
    ):
        """Record material processed (e.g., composted)"""
        flow = self.material_flows.get(flow_id)
        if not flow:
            raise ValueError(f"Flow {flow_id} not found")

        flow.processing_date = processing_date
        flow.stage = FlowStage.PROCESSING

        # Calculate carbon impact
        # Composting avoids landfill methane: ~1.5 lbs CO2e per lb of organics
        flow.carbon_avoided_lbs = flow.quantity_lbs * Decimal("1.5")

        # Estimate nutrient recovery (typical values)
        flow.nutrients_recovered_lbs = {
            "nitrogen": flow.quantity_lbs * Decimal("0.015"),  # 1.5% N
            "phosphorus": flow.quantity_lbs * Decimal("0.005"),  # 0.5% P
            "potassium": flow.quantity_lbs * Decimal("0.010")  # 1.0% K
        }

        # Update connection carbon tracking
        if flow.destination_org_id:
            connections = self.get_organization_connections(
                flow.source_org_id, as_donor=True
            )
            for conn in connections:
                if conn.recipient_org_id == flow.destination_org_id:
                    conn.total_carbon_avoided_lbs += flow.carbon_avoided_lbs
                    break

    def record_material_application(
        self,
        flow_id: str,
        application_date: date,
        acres_applied: Optional[Decimal] = None
    ):
        """Record material applied to farm (e.g., compost spread on fields)"""
        flow = self.material_flows.get(flow_id)
        if not flow:
            raise ValueError(f"Flow {flow_id} not found")

        flow.application_date = application_date
        flow.stage = FlowStage.APPLICATION
        flow.verified = True

    def get_material_flows(
        self,
        org_id: Optional[str] = None,
        as_source: bool = True,
        material_type: Optional[MaterialType] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[MaterialFlow]:
        """Get material flows with filters"""
        if org_id:
            if as_source:
                flow_ids = self.flows_by_source.get(org_id, [])
            else:
                flow_ids = self.flows_by_destination.get(org_id, [])
            flows = [self.material_flows[fid] for fid in flow_ids if fid in self.material_flows]
        else:
            flows = list(self.material_flows.values())

        # Filter by material type
        if material_type:
            flows = [f for f in flows if f.material_type == material_type]

        # Filter by date range
        if start_date:
            flows = [f for f in flows if f.generation_date >= start_date]
        if end_date:
            flows = [f for f in flows if f.generation_date <= end_date]

        return flows

    # ========================================================================
    # Circularity Metrics
    # ========================================================================

    def calculate_circularity_metrics(
        self,
        org_id: str,
        start_date: date,
        end_date: date
    ) -> CircularityMetrics:
        """Calculate circularity metrics for an organization"""
        org = self.get_organization(org_id)
        if not org:
            raise ValueError(f"Organization {org_id} not found")

        metrics = CircularityMetrics(
            org_id=org_id,
            org_name=org.name,
            period_start=start_date,
            period_end=end_date
        )

        # Get flows as source (generating food scraps)
        source_flows = self.get_material_flows(
            org_id, as_source=True,
            material_type=MaterialType.FOOD_SCRAPS,
            start_date=start_date,
            end_date=end_date
        )

        # Get flows as destination (receiving compost)
        dest_flows = self.get_material_flows(
            org_id, as_source=False,
            material_type=MaterialType.COMPOST,
            start_date=start_date,
            end_date=end_date
        )

        # Calculate food scraps generated
        metrics.food_scraps_generated_lbs = sum(f.quantity_lbs for f in source_flows)

        # Calculate food scraps diverted (collected)
        diverted_flows = [f for f in source_flows if f.stage != FlowStage.GENERATION]
        metrics.food_scraps_diverted_lbs = sum(f.quantity_lbs for f in diverted_flows)

        # Waste diversion rate
        if metrics.food_scraps_generated_lbs > 0:
            metrics.waste_diversion_rate = float(
                metrics.food_scraps_diverted_lbs / metrics.food_scraps_generated_lbs * 100
            )

        # Carbon avoided
        metrics.carbon_avoided_lbs = sum(f.carbon_avoided_lbs for f in diverted_flows)

        # If this is a farm, calculate compost received and applied
        if isinstance(org, Farm):
            metrics.compost_produced_lbs = sum(f.quantity_lbs for f in dest_flows)
            applied_flows = [f for f in dest_flows if f.stage == FlowStage.APPLICATION]
            metrics.compost_applied_lbs = sum(f.quantity_lbs for f in applied_flows)

            # Estimate carbon sequestered (compost sequesters ~0.5 lbs CO2e per lb)
            metrics.carbon_sequestered_lbs = metrics.compost_applied_lbs * Decimal("0.5")

            # Calculate nutrient recovery
            for flow in diverted_flows:
                if flow.nutrients_recovered_lbs:
                    metrics.nitrogen_recovered_lbs += flow.nutrients_recovered_lbs.get("nitrogen", Decimal("0.0"))
                    metrics.phosphorus_recovered_lbs += flow.nutrients_recovered_lbs.get("phosphorus", Decimal("0.0"))
                    metrics.potassium_recovered_lbs += flow.nutrients_recovered_lbs.get("potassium", Decimal("0.0"))

        # Net carbon impact (avoided + sequestered)
        metrics.net_carbon_impact_lbs = metrics.carbon_avoided_lbs + metrics.carbon_sequestered_lbs

        # Calculate circularity score (0-100)
        score = 0.0

        # Waste diversion (40 points)
        score += min(40.0, metrics.waste_diversion_rate * 0.4)

        # Carbon impact (30 points)
        if metrics.food_scraps_generated_lbs > 0:
            carbon_per_lb = float(metrics.net_carbon_impact_lbs / metrics.food_scraps_generated_lbs)
            score += min(30.0, carbon_per_lb * 15.0)

        # Compost application (30 points - only for farms)
        if isinstance(org, Farm) and metrics.compost_produced_lbs > 0:
            application_rate = float(metrics.compost_applied_lbs / metrics.compost_produced_lbs * 100)
            score += min(30.0, application_rate * 0.3)
        elif not isinstance(org, Farm):
            # Non-farms get points for diversion
            score += 30.0 if metrics.waste_diversion_rate >= 75.0 else 0.0

        metrics.circularity_score = min(100.0, score)

        return metrics

    def get_network_circularity_score(self) -> float:
        """Calculate overall network circularity score"""
        if not self.organizations:
            return 0.0

        # Calculate metrics for all organizations
        end_date = date.today()
        start_date = end_date - timedelta(days=90)  # Last 90 days

        org_scores = []
        for org_id in self.organizations:
            try:
                metrics = self.calculate_circularity_metrics(org_id, start_date, end_date)
                org_scores.append(metrics.circularity_score)
            except Exception:
                pass

        if not org_scores:
            return 0.0

        return sum(org_scores) / len(org_scores)

    # ========================================================================
    # Impact Reporting
    # ========================================================================

    def generate_impact_report(
        self,
        start_date: date,
        end_date: date
    ) -> Dict:
        """Generate impact report for entire network"""
        report = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "materials": {
                "food_scraps_diverted_lbs": Decimal("0.0"),
                "compost_produced_lbs": Decimal("0.0"),
                "landfill_avoidance_lbs": Decimal("0.0")
            },
            "carbon": {
                "carbon_avoided_lbs": Decimal("0.0"),
                "carbon_sequestered_lbs": Decimal("0.0"),
                "net_impact_lbs": Decimal("0.0"),
                "co2e_equivalent_tons": Decimal("0.0")
            },
            "nutrients": {
                "nitrogen_recovered_lbs": Decimal("0.0"),
                "phosphorus_recovered_lbs": Decimal("0.0"),
                "potassium_recovered_lbs": Decimal("0.0")
            },
            "circularity": {
                "network_score": 0.0,
                "active_connections": 0,
                "participating_organizations": 0
            }
        }

        # Aggregate across all organizations
        for org_id in self.organizations:
            try:
                metrics = self.calculate_circularity_metrics(org_id, start_date, end_date)

                report["materials"]["food_scraps_diverted_lbs"] += metrics.food_scraps_diverted_lbs
                report["materials"]["compost_produced_lbs"] += metrics.compost_produced_lbs
                report["materials"]["landfill_avoidance_lbs"] += metrics.food_scraps_diverted_lbs

                report["carbon"]["carbon_avoided_lbs"] += metrics.carbon_avoided_lbs
                report["carbon"]["carbon_sequestered_lbs"] += metrics.carbon_sequestered_lbs

                report["nutrients"]["nitrogen_recovered_lbs"] += metrics.nitrogen_recovered_lbs
                report["nutrients"]["phosphorus_recovered_lbs"] += metrics.phosphorus_recovered_lbs
                report["nutrients"]["potassium_recovered_lbs"] += metrics.potassium_recovered_lbs
            except Exception:
                pass

        # Calculate net carbon impact
        report["carbon"]["net_impact_lbs"] = (
            report["carbon"]["carbon_avoided_lbs"] +
            report["carbon"]["carbon_sequestered_lbs"]
        )

        # Convert to tons CO2e
        report["carbon"]["co2e_equivalent_tons"] = report["carbon"]["net_impact_lbs"] / Decimal("2000.0")

        # Network circularity
        report["circularity"]["network_score"] = self.get_network_circularity_score()
        report["circularity"]["active_connections"] = len([
            c for c in self.loop_connections.values() if c.active
        ])
        report["circularity"]["participating_organizations"] = len(self.organizations)

        return report
