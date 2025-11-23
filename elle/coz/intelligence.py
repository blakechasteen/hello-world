"""
Intelligence Engine for COZ System

Provides cross-parser insights, recommendations, and predictions by analyzing
data from multiple parsers simultaneously.

Features:
- Profit analysis (time + cost + financials integration)
- Efficiency insights and recommendations
- Trend detection and forecasting
- Actionable recommendations
- Executive daily brief generation (using HoloLoom prompt refinement)

Part of COZ Expansion Phase 1 (Intelligence Layer)
Created: 2025-11-21
Updated: 2025-11-22 (added daily brief generation)
"""

from typing import Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime
import logging

if TYPE_CHECKING:
    from elle.coz.sync_manager import SyncManager

# Import HoloLoom prompt refinement (graceful degradation if not available)
try:
    from HoloLoom.prompting.metaprompt import create_metaprompt_auto
    from HoloLoom.config import Config
    REFINEMENT_AVAILABLE = True
except ImportError:
    REFINEMENT_AVAILABLE = False


@dataclass
class ProfitAnalysis:
    """Comprehensive profit analysis result"""
    total_revenue: float
    total_costs: float
    total_labor: float
    total_expenses: float
    net_profit: float
    profit_margin: float
    hourly_profit: float
    breakeven_hours: float
    recommendations: List[str]


class IntelligenceEngine:
    """Cross-parser intelligence and insights"""

    def __init__(self, sync_manager: 'SyncManager', logger: Optional[logging.Logger] = None):
        """
        Initialize intelligence engine

        Args:
            sync_manager: SyncManager instance with all parsers loaded
            logger: Optional logger instance
        """
        self.sync = sync_manager
        self.logger = logger or logging.getLogger(__name__)

    def analyze_profit(self, hourly_rate: float = 25.0) -> Dict:
        """
        Comprehensive profit analysis integrating time, cost, and financial data

        Cross-parser: Time + Cost + Financials

        Args:
            hourly_rate: Hourly labor rate for calculations

        Returns:
            Dict containing:
            - total_revenue: From financials or orders
            - total_costs: From cost tracking
            - total_labor: From time tracking × hourly rate
            - total_expenses: From expenses parser (if available)
            - net_profit: Revenue - costs - labor - expenses
            - profit_margin: (net_profit / revenue) × 100
            - hourly_profit: net_profit / total_hours_worked
            - breakeven_hours: hours needed to cover costs
            - recommendations: List of actionable insights
        """
        # Get data from parsers
        time_summary = self.sync.time_tracking.get_time_summary() if hasattr(self.sync, 'time_tracking') else {}
        cost_summary = self.sync.cost_tracking.get_cost_summary() if hasattr(self.sync, 'cost_tracking') else {}
        # Revenue comes from customer orders (fulfilled orders)
        revenue_pipeline = self.sync.customer_orders.get_revenue_pipeline() if hasattr(self.sync, 'customer_orders') else {}

        # Calculate totals
        total_hours = time_summary.get('total_actual_hours', 0.0)
        total_revenue = revenue_pipeline.get('Fulfilled', 0.0) + revenue_pipeline.get('In Progress', 0.0)
        total_costs = cost_summary.get('total_cost', 0.0)
        total_labor = total_hours * hourly_rate
        total_expenses = 0.0  # TODO: Add when expenses parser exists

        # Calculate profit metrics
        net_profit = total_revenue - total_costs
        profit_margin = (net_profit / total_revenue) if total_revenue > 0 else 0.0
        hourly_profit = (net_profit / total_hours) if total_hours > 0 else 0.0
        breakeven_hours = (total_costs / hourly_rate) if hourly_rate > 0 else 0.0

        # Generate recommendations
        recommendations = self._generate_profit_recommendations(
            profit_margin=profit_margin,
            hourly_profit=hourly_profit,
            time_summary=time_summary,
            cost_summary=cost_summary
        )

        return {
            'total_revenue': round(total_revenue, 2),
            'total_costs': round(total_costs, 2),
            'total_labor': round(total_labor, 2),
            'total_expenses': round(total_expenses, 2),
            'net_profit': round(net_profit, 2),
            'profit_margin': round(profit_margin * 100, 1),  # As percentage
            'hourly_profit': round(hourly_profit, 2),
            'breakeven_hours': round(breakeven_hours, 1),
            'recommendations': recommendations,
            'total_hours_worked': round(total_hours, 1),
        }

    def _generate_profit_recommendations(
        self,
        profit_margin: float,
        hourly_profit: float,
        time_summary: Dict,
        cost_summary: Dict
    ) -> List[str]:
        """Generate actionable profit recommendations"""
        recommendations = []

        # Profit margin recommendations
        if profit_margin < 0:
            recommendations.append("🚨 CRITICAL: Operating at a loss. Review pricing and costs immediately.")
        elif profit_margin < 0.15:
            recommendations.append("⚠️ Low profit margin (<15%). Consider raising prices or reducing costs.")
        elif profit_margin > 0.5:
            recommendations.append("✅ Excellent profit margin (>50%). Consider reinvesting in growth.")

        # Hourly profit recommendations
        target_hourly = 15.0  # Target $15/hour profit
        if hourly_profit < target_hourly:
            recommendations.append(
                f"💰 Hourly profit (${hourly_profit:.2f}) below target (${target_hourly:.2f}). "
                f"Focus on high-margin tasks."
            )

        # Efficiency recommendations
        avg_efficiency = time_summary.get('avg_efficiency_score', 1.0)
        if avg_efficiency < 0.8:
            recommendations.append(
                f"⏱️ Average efficiency ({avg_efficiency:.1%}) below 80%. "
                f"Tasks consistently taking longer than estimated."
            )

        # Cost recommendations
        avg_cost = cost_summary.get('avg_cost_per_task', 0.0)
        if avg_cost > 100:
            recommendations.append(
                f"💸 High average cost per task (${avg_cost:.2f}). "
                f"Review material costs and supplier pricing."
            )

        # General recommendations
        if not recommendations:
            recommendations.append("✅ Profit metrics look healthy. Continue current operations.")

        return recommendations

    def get_efficiency_insights(self) -> Dict:
        """
        Analyze time efficiency across tasks and categories

        Returns insights on:
        - Tasks with worst time overruns
        - Categories with efficiency issues
        - Patterns in time estimation accuracy
        """
        if not hasattr(self.sync, 'time_tracking'):
            return {"error": "Time tracking parser not available"}

        time_parser = self.sync.time_tracking

        # Get overruns
        overruns = time_parser.get_overruns(threshold=0.2)

        # Get category summary
        category_summary = time_parser.get_category_summary()

        # Find categories with worst efficiency
        inefficient_categories = [
            (cat, data['avg_efficiency_score'])
            for cat, data in category_summary.items()
        ]
        inefficient_categories.sort(key=lambda x: x[1])  # Lowest efficiency first

        # Generate insights
        insights = []

        if overruns:
            worst_overrun = overruns[0]
            insights.append(
                f"⚠️ '{worst_overrun['task_name']}' had worst overrun: "
                f"{worst_overrun['variance_percent']}% over estimate"
            )

        if inefficient_categories:
            worst_category, worst_score = inefficient_categories[0]
            insights.append(
                f"📂 '{worst_category}' category has lowest efficiency: "
                f"{worst_score:.1%}"
            )

        return {
            'top_overruns': overruns[:5],
            'category_efficiency': category_summary,
            'insights': insights,
        }

    def get_cost_insights(self) -> Dict:
        """
        Analyze cost patterns and identify optimization opportunities

        Returns insights on:
        - Most expensive tasks
        - Cost breakdown (material/labor/overhead)
        - Potential cost savings
        """
        if not hasattr(self.sync, 'cost_tracking'):
            return {"error": "Cost tracking parser not available"}

        cost_parser = self.sync.cost_tracking

        # Get top expensive tasks
        expensive_tasks = cost_parser.get_top_expensive_tasks(limit=5)

        # Get cost breakdown
        breakdown = cost_parser.get_cost_breakdown()

        # Get overhead rate
        overhead = cost_parser.get_overhead_rate()

        # Generate insights
        insights = []

        # Material cost insights
        if breakdown['material_percent'] > 50:
            insights.append(
                f"💰 Material costs are {breakdown['material_percent']}% of total. "
                f"Negotiate bulk discounts with suppliers."
            )

        # Overhead insights
        if overhead['overhead_rate'] > 30:
            insights.append(
                f"📊 Overhead rate is {overhead['overhead_rate']}%. "
                f"Review fixed costs and allocations."
            )

        # Expensive task insights
        if expensive_tasks:
            most_expensive = expensive_tasks[0]
            insights.append(
                f"💸 '{most_expensive['task_name']}' most expensive: ${most_expensive['total_cost']}. "
                f"Review for optimization opportunities."
            )

        return {
            'expensive_tasks': expensive_tasks,
            'cost_breakdown': breakdown,
            'overhead_rate': overhead,
            'insights': insights,
        }

    def get_task_prioritization_insights(self) -> Dict:
        """
        Suggest task prioritization based on profit, efficiency, and urgency

        Returns recommended focus areas for maximum impact
        """
        insights = []

        # Profit analysis
        profit = self.analyze_profit()
        if profit['hourly_profit'] < 10:
            insights.append(
                "Focus on high-margin tasks. Current hourly profit is low."
            )

        # Efficiency analysis
        efficiency = self.get_efficiency_insights()
        if 'top_overruns' in efficiency and efficiency['top_overruns']:
            worst_task = efficiency['top_overruns'][0]['task_name']
            insights.append(
                f"Create SOP for '{worst_task}' - consistent time overruns."
            )

        # Cost analysis
        cost = self.get_cost_insights()
        if 'expensive_tasks' in cost and cost['expensive_tasks']:
            expensive_task = cost['expensive_tasks'][0]['task_name']
            insights.append(
                f"Review costs for '{expensive_task}' - highest total cost."
            )

        return {
            'insights': insights,
            'recommended_actions': self._generate_action_plan(profit, efficiency, cost)
        }

    def _generate_action_plan(self, profit: Dict, efficiency: Dict, cost: Dict) -> List[Dict]:
        """Generate prioritized action plan"""
        actions = []

        # Critical profit issues (highest priority)
        if profit.get('profit_margin', 0) < 0:
            actions.append({
                'priority': 'CRITICAL',
                'category': 'Profit',
                'action': 'Review pricing - operating at loss',
                'expected_impact': 'High',
            })

        # Efficiency improvements (medium priority)
        if 'top_overruns' in efficiency and efficiency['top_overruns']:
            overrun = efficiency['top_overruns'][0]
            actions.append({
                'priority': 'HIGH',
                'category': 'Efficiency',
                'action': f"Create SOP for '{overrun['task_name']}'",
                'expected_impact': 'Medium',
            })

        # Cost optimizations (medium priority)
        if 'expensive_tasks' in cost and cost['expensive_tasks']:
            task = cost['expensive_tasks'][0]
            actions.append({
                'priority': 'MEDIUM',
                'category': 'Cost',
                'action': f"Optimize costs for '{task['task_name']}'",
                'expected_impact': 'Medium',
            })

        return actions

    # ========================
    # Phase 2: Advanced Intelligence Features
    # ========================

    def analyze_revenue_vs_cost(self) -> Dict:
        """
        Revenue vs. Cost Analysis

        Cross-parser: Orders + Costs + Financials

        Compares revenue pipeline with actual costs to identify profitability gaps.
        Analyzes revenue by customer, product, and time period.

        Returns:
            Dict containing:
            - revenue_pipeline: Total pending revenue from orders
            - fulfilled_revenue: Revenue from completed orders
            - total_costs: Actual costs from cost tracking
            - projected_profit: Expected profit from pending orders
            - profitability_by_product: Margin per product
            - recommendations: Revenue optimization suggestions
        """
        insights = []

        # Get order revenue pipeline
        if hasattr(self.sync, 'customer_orders'):
            pipeline = self.sync.customer_orders.get_revenue_pipeline()
            revenue_pending = pipeline.get('total_pending', 0.0)
            revenue_fulfilled = pipeline.get('Fulfilled', 0.0)
        else:
            pipeline = {}
            revenue_pending = 0.0
            revenue_fulfilled = 0.0
            insights.append("⚠️ Customer orders parser not available")

        # Get actual costs
        if hasattr(self.sync, 'cost_tracking'):
            cost_summary = self.sync.cost_tracking.get_cost_summary()
            total_costs = cost_summary.get('total_cost', 0.0)
        else:
            total_costs = 0.0
            insights.append("⚠️ Cost tracking parser not available")

        # Calculate projected profit
        projected_profit = revenue_pending - (total_costs * (revenue_pending / revenue_fulfilled)) if revenue_fulfilled > 0 else 0.0
        actual_profit = revenue_fulfilled - total_costs

        # Profitability by product (if data available)
        profitability_by_product = {}
        if hasattr(self.sync, 'customer_orders') and hasattr(self.sync, 'production_log'):
            product_demand = self.sync.customer_orders.get_product_demand()
            production_performance = self.sync.production_log.get_product_performance()

            for product in product_demand:
                product_revenue = product_demand[product].get('revenue', 0.0)
                # Estimate costs (simplified - would need product-specific cost tracking)
                product_profit = product_revenue  # Placeholder

                profitability_by_product[product] = {
                    'revenue': round(product_revenue, 2),
                    'profit': round(product_profit, 2),  # Simplified
                }

        # Generate recommendations
        if revenue_pending < total_costs:
            insights.append("🚨 CRITICAL: Pending revenue insufficient to cover costs")

        if actual_profit < 0:
            insights.append("💰 Operating at a loss. Increase pricing or reduce costs.")

        if revenue_pending > revenue_fulfilled * 2:
            insights.append("📈 Strong pipeline! Ensure production capacity can meet demand.")

        return {
            'revenue_pipeline': round(revenue_pending, 2),
            'fulfilled_revenue': round(revenue_fulfilled, 2),
            'total_costs': round(total_costs, 2),
            'projected_profit': round(projected_profit, 2),
            'actual_profit': round(actual_profit, 2),
            'profitability_by_product': profitability_by_product,
            'insights': insights,
        }

    def analyze_production_efficiency(self) -> Dict:
        """
        Production Efficiency Analysis

        Cross-parser: Production + Time + SOPs

        Compares actual production time/output with SOP estimates.
        Identifies bottlenecks and optimization opportunities.

        Returns:
            Dict containing:
            - production_vs_sops: Comparison of actual vs. estimated
            - time_variances: Time overruns per product
            - output_efficiency: Actual vs. expected output
            - recommendations: Process improvement suggestions
        """
        insights = []

        # Get production log data
        if hasattr(self.sync, 'production_log'):
            production_summary = self.sync.production_log.get_production_summary()
            product_performance = self.sync.production_log.get_product_performance()
        else:
            insights.append("⚠️ Production log parser not available")
            return {'insights': insights}

        # Get SOP estimates
        if hasattr(self.sync, 'sops'):
            sop_library = self.sync.sops.get_sop_library()
        else:
            insights.append("⚠️ SOP parser not available")
            sop_library = {}

        # Get time tracking data
        if hasattr(self.sync, 'time_tracking'):
            time_summary = self.sync.time_tracking.get_time_summary()
            avg_efficiency = time_summary.get('avg_efficiency_score', 1.0)
        else:
            avg_efficiency = 1.0

        # Compare production runs with SOP estimates (simplified)
        production_vs_sops = {}
        for product, perf in product_performance.items():
            # Would match with SOP data in full implementation
            production_vs_sops[product] = {
                'avg_produced_per_run': perf.get('avg_produced_per_run', 0),
                'sellthrough_rate': perf.get('avg_sellthrough_rate', 0),
                'waste_rate': perf.get('avg_waste_percent', 0),
            }

        # Time variance analysis
        time_variances = {
            'overall_efficiency': round(avg_efficiency * 100, 1),
            'below_target': avg_efficiency < 0.8,
        }

        # Generate recommendations
        avg_waste = production_summary.get('avg_waste_percent', 0.0)
        if avg_waste > 15:
            insights.append(f"🗑️ High waste rate ({avg_waste:.1f}%). Review production quantities.")

        avg_sellthrough = production_summary.get('avg_sellthrough_rate', 0.0)
        if avg_sellthrough < 80:
            insights.append(f"📉 Low sellthrough ({avg_sellthrough:.1f}%). Reduce batch sizes.")

        if avg_efficiency < 0.8:
            insights.append(f"⏱️ Production efficiency below target. Review SOPs for bottlenecks.")

        return {
            'production_vs_sops': production_vs_sops,
            'time_variances': time_variances,
            'output_efficiency': {
                'waste_rate': round(avg_waste, 1),
                'sellthrough_rate': round(avg_sellthrough, 1),
            },
            'insights': insights,
        }

    def get_waste_reduction_recommendations(self) -> Dict:
        """
        Waste Reduction Recommendations

        Cross-parser: Production + Inventory + SOPs

        Analyzes waste patterns and recommends production adjustments.
        Identifies overproduction, quality issues, and demand mismatches.

        Returns:
            Dict containing:
            - waste_analysis: Breakdown by reason and product
            - overproduction_alerts: Products consistently overproduced
            - quality_issues: Quality-related waste incidents
            - recommended_actions: Specific waste reduction steps
        """
        insights = []

        # Get waste analysis from production log
        if not hasattr(self.sync, 'production_log'):
            return {'error': 'Production log parser not available'}

        waste_analysis = self.sync.production_log.get_waste_analysis()
        overproduction_alerts = self.sync.production_log.get_overproduction_alerts()
        quality_issues = self.sync.production_log.get_quality_issues()
        production_forecast = self.sync.production_log.get_production_forecast()

        # Waste by reason
        waste_reasons = waste_analysis.get('waste_reasons', {})
        high_waste_products = waste_analysis.get('high_waste_products', [])

        # Generate specific recommendations
        recommended_actions = []

        # Overproduction recommendations
        if 'Overproduction' in waste_reasons:
            overproduction_waste = waste_reasons['Overproduction']
            insights.append(f"🚨 Overproduction waste: {overproduction_waste} units")
            recommended_actions.append({
                'priority': 'HIGH',
                'action': 'Reduce batch sizes based on demand forecast',
                'expected_impact': f'Save ~{overproduction_waste} units per period',
            })

        # Quality issue recommendations
        if quality_issues:
            total_quality_waste = sum(q['quantity_wasted'] for q in quality_issues)
            insights.append(f"⚠️ Quality issues: {total_quality_waste} units wasted")
            recommended_actions.append({
                'priority': 'MEDIUM',
                'action': 'Review quality control processes',
                'expected_impact': f'Reduce {total_quality_waste} units waste',
            })

        # High waste products
        for product_data in high_waste_products:
            product = product_data['product']
            waste_rate = product_data['waste_rate']

            insights.append(f"📊 {product}: {waste_rate:.1f}% waste rate")

            # Recommend following production forecast
            if product in production_forecast:
                recommended_qty = production_forecast[product]
                recommended_actions.append({
                    'priority': 'MEDIUM',
                    'action': f"Produce {int(recommended_qty)} {product} (forecast-based)",
                    'expected_impact': f'Reduce {product} waste',
                })

        return {
            'waste_analysis': waste_analysis,
            'overproduction_alerts': overproduction_alerts[:5],
            'quality_issues': quality_issues,
            'recommended_actions': recommended_actions,
            'insights': insights,
        }

    def optimize_order_fulfillment(self) -> Dict:
        """
        Order Fulfillment Optimization

        Cross-parser: Orders + Production + SOPs + Kanban

        Matches pending orders with production capacity and task priorities.
        Recommends fulfillment schedule based on due dates and resources.

        Returns:
            Dict containing:
            - fulfillment_schedule: Prioritized order list
            - production_plan: What to produce and when
            - capacity_analysis: Time/resource constraints
            - recommendations: Fulfillment optimization steps
        """
        insights = []

        # Get pending orders
        if not hasattr(self.sync, 'customer_orders'):
            return {'error': 'Customer orders parser not available'}

        pending_orders = self.sync.customer_orders.get_pending_orders()
        fulfillment_schedule = self.sync.customer_orders.get_fulfillment_schedule()

        # Get current inventory
        if hasattr(self.sync, 'production_log'):
            current_inventory = self.sync.production_log.get_inventory_status()
        else:
            current_inventory = {}

        # Get SOP time estimates
        if hasattr(self.sync, 'sops'):
            sop_library = self.sync.sops.get_sop_library()
            avg_sop_time = sop_library.get('avg_time', 2.0)
        else:
            avg_sop_time = 2.0

        # Build production plan
        production_plan = []
        total_time_required = 0.0

        for priority_group, orders in fulfillment_schedule.items():
            for order in orders:
                # Calculate production needed (simplified - assumes 1:1 order:production)
                products_needed = order.products
                quantities_needed = order.quantities

                for product, quantity in zip(products_needed, quantities_needed):
                    # Check if inventory covers it
                    available = current_inventory.get(product, 0)
                    to_produce = max(0, quantity - available)

                    if to_produce > 0:
                        # Estimate time (simplified)
                        estimated_time = to_produce * avg_sop_time
                        total_time_required += estimated_time

                        production_plan.append({
                            'order_id': order.order_id,
                            'product': product,
                            'quantity': to_produce,
                            'due_date': order.due_date.date().isoformat(),
                            'estimated_time_hours': round(estimated_time, 1),
                            'priority': priority_group,
                        })

        # Capacity analysis
        capacity_analysis = {
            'total_hours_required': round(total_time_required, 1),
            'orders_pending': len(pending_orders),
            'critical_orders': len(fulfillment_schedule.get('Critical (Overdue/Due Today)', [])),
        }

        # Generate recommendations
        critical_count = capacity_analysis['critical_orders']
        if critical_count > 0:
            insights.append(f"🚨 {critical_count} critical orders! Prioritize immediately.")

        if total_time_required > 40:  # More than 1 week of work
            insights.append(f"⚠️ {total_time_required:.1f}h of production needed. Consider additional capacity.")

        overdue = self.sync.customer_orders.get_overdue_orders()
        if overdue:
            insights.append(f"📅 {len(overdue)} overdue orders. Expedite fulfillment.")

        return {
            'fulfillment_schedule': {k: [o.to_dict() for o in v] for k, v in fulfillment_schedule.items()},
            'production_plan': production_plan,
            'capacity_analysis': capacity_analysis,
            'insights': insights,
        }

    def get_customer_insights(self) -> Dict:
        """
        Customer Insights

        Cross-parser: Orders + Financials + Historical data

        Analyzes customer behavior, profitability, and loyalty patterns.
        Identifies high-value customers and churn risks.

        Returns:
            Dict containing:
            - customer_summary: Stats per customer
            - top_customers: Highest revenue customers
            - customer_trends: Order frequency and patterns
            - recommendations: Customer relationship actions
        """
        insights = []

        # Get customer summary
        if not hasattr(self.sync, 'customer_orders'):
            return {'error': 'Customer orders parser not available'}

        customer_summary = self.sync.customer_orders.get_customer_summary()

        # Identify top customers (by revenue)
        top_customers = sorted(
            customer_summary.items(),
            key=lambda x: x[1]['total_revenue'],
            reverse=True
        )[:5]

        # Calculate customer metrics
        total_customers = len(customer_summary)
        avg_orders_per_customer = sum(c['total_orders'] for c in customer_summary.values()) / total_customers if total_customers > 0 else 0
        avg_revenue_per_customer = sum(c['total_revenue'] for c in customer_summary.values()) / total_customers if total_customers > 0 else 0

        # Identify at-risk customers (low fulfillment rate)
        at_risk_customers = [
            (name, stats)
            for name, stats in customer_summary.items()
            if stats['fulfillment_rate'] < 0.8 and stats['total_orders'] > 1
        ]

        # Generate recommendations
        if top_customers:
            top_name, top_stats = top_customers[0]
            insights.append(f"⭐ Top customer: {top_name} (${top_stats['total_revenue']:.2f} revenue)")

        if at_risk_customers:
            insights.append(f"⚠️ {len(at_risk_customers)} customers have fulfillment issues. Improve service.")

        if avg_orders_per_customer < 2:
            insights.append("📈 Low repeat orders. Implement loyalty program.")

        return {
            'customer_summary': customer_summary,
            'top_customers': {name: stats for name, stats in top_customers},
            'customer_metrics': {
                'total_customers': total_customers,
                'avg_orders_per_customer': round(avg_orders_per_customer, 1),
                'avg_revenue_per_customer': round(avg_revenue_per_customer, 2),
            },
            'at_risk_customers': [name for name, _ in at_risk_customers],
            'insights': insights,
        }

    def generate_daily_brief(
        self,
        hourly_rate: float = 25.0,
        use_refinement: bool = True,
        refinement_provider: str = "anthropic"
    ) -> Dict:
        """
        Generate Executive Daily Brief

        Aggregates insights from all analysis methods and uses HoloLoom
        prompt refinement to transform raw metrics into executive-quality
        intelligence report.

        Features:
        - Comprehensive cross-parser analysis
        - Executive summary (refined for clarity)
        - Key metrics and trends
        - Prioritized action items
        - Performance indicators

        Args:
            hourly_rate: Hourly labor rate for profit calculations
            use_refinement: Use HoloLoom prompt refinement for executive quality
            refinement_provider: LLM provider ("anthropic", "google", "openai")

        Returns:
            Dict containing:
            - date: Brief generation date
            - summary: Executive summary (refined if enabled)
            - profit: Profit analysis results
            - efficiency: Efficiency insights
            - cost: Cost insights
            - production: Production efficiency
            - waste: Waste reduction recommendations
            - orders: Order fulfillment optimization
            - customers: Customer insights
            - action_items: Prioritized actions (top 5)
            - performance_indicators: Key metrics dashboard
            - refinement_used: Whether refinement was applied
        """

        self.logger.info(f"Generating daily brief (refinement: {use_refinement})")

        # Gather all insights
        try:
            profit = self.analyze_profit(hourly_rate=hourly_rate)
        except Exception as e:
            self.logger.error(f"Profit analysis failed: {e}")
            profit = {'error': str(e)}

        try:
            efficiency = self.get_efficiency_insights()
        except Exception as e:
            self.logger.error(f"Efficiency analysis failed: {e}")
            efficiency = {'error': str(e)}

        try:
            cost = self.get_cost_insights()
        except Exception as e:
            self.logger.error(f"Cost analysis failed: {e}")
            cost = {'error': str(e)}

        try:
            production = self.analyze_production_efficiency()
        except Exception as e:
            self.logger.error(f"Production analysis failed: {e}")
            production = {'error': str(e)}

        try:
            waste = self.get_waste_reduction_recommendations()
        except Exception as e:
            self.logger.error(f"Waste analysis failed: {e}")
            waste = {'error': str(e)}

        try:
            orders = self.optimize_order_fulfillment()
        except Exception as e:
            self.logger.error(f"Order optimization failed: {e}")
            orders = {'error': str(e)}

        try:
            customers = self.get_customer_insights()
        except Exception as e:
            self.logger.error(f"Customer analysis failed: {e}")
            customers = {'error': str(e)}

        # Aggregate action items (top 5 priority)
        all_recommendations = []

        if 'recommendations' in profit:
            all_recommendations.extend(profit['recommendations'])
        if 'insights' in efficiency:
            all_recommendations.extend(efficiency['insights'])
        if 'insights' in cost:
            all_recommendations.extend(cost['insights'])
        if 'recommendations' in production:
            all_recommendations.extend(production['recommendations'])
        if 'recommendations' in waste:
            all_recommendations.extend(waste['recommendations'])
        if 'recommendations' in orders:
            all_recommendations.extend(orders['recommendations'])
        if 'insights' in customers:
            all_recommendations.extend(customers['insights'])

        # Prioritize action items (take top 5)
        top_actions = all_recommendations[:5]

        # Build raw summary
        raw_summary = self._build_raw_summary(
            profit=profit,
            efficiency=efficiency,
            cost=cost,
            production=production,
            waste=waste,
            orders=orders,
            customers=customers
        )

        # Refine summary if enabled
        if use_refinement and REFINEMENT_AVAILABLE:
            refined_summary = self._refine_summary(
                raw_summary=raw_summary,
                provider=refinement_provider
            )
            summary = refined_summary
            refinement_used = True
        else:
            if use_refinement and not REFINEMENT_AVAILABLE:
                self.logger.warning(
                    "Refinement requested but HoloLoom not available. "
                    "Using raw summary."
                )
            summary = raw_summary
            refinement_used = False

        # Build performance indicators dashboard
        performance_indicators = {
            'profit_margin': profit.get('profit_margin', 0.0),
            'hourly_profit': profit.get('hourly_profit', 0.0),
            'task_efficiency': efficiency.get('overall_efficiency', 0.0) if 'overall_efficiency' in efficiency else 0.0,
            'waste_rate': waste.get('waste_rate', 0.0) if 'waste_rate' in waste else 0.0,
            'order_fulfillment_rate': orders.get('overall_fulfillment_rate', 0.0) if 'overall_fulfillment_rate' in orders else 0.0,
        }

        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'summary': summary,
            'profit': profit,
            'efficiency': efficiency,
            'cost': cost,
            'production': production,
            'waste': waste,
            'orders': orders,
            'customers': customers,
            'action_items': top_actions,
            'performance_indicators': performance_indicators,
            'refinement_used': refinement_used,
        }

    def _build_raw_summary(
        self,
        profit: Dict,
        efficiency: Dict,
        cost: Dict,
        production: Dict,
        waste: Dict,
        orders: Dict,
        customers: Dict
    ) -> str:
        """Build raw executive summary from all insights."""

        lines = [
            "# COZ Daily Intelligence Brief",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Financial Overview",
        ]

        # Profit summary
        if 'net_profit' in profit:
            lines.append(f"- **Net Profit**: ${profit['net_profit']:,.2f}")
            lines.append(f"- **Profit Margin**: {profit['profit_margin']:.1%}")
            lines.append(f"- **Hourly Profit**: ${profit['hourly_profit']:.2f}/hour")

        # Efficiency summary
        if 'overall_efficiency' in efficiency:
            lines.append("")
            lines.append("## Operational Efficiency")
            lines.append(f"- **Overall Efficiency**: {efficiency['overall_efficiency']:.1%}")

        # Production summary
        if 'sellthrough_rate' in production:
            lines.append("")
            lines.append("## Production Performance")
            lines.append(f"- **Sellthrough Rate**: {production['sellthrough_rate']:.1%}")
            lines.append(f"- **Waste Rate**: {production['waste_rate']:.1%}")

        # Order fulfillment summary
        if 'overall_fulfillment_rate' in orders:
            lines.append("")
            lines.append("## Order Fulfillment")
            lines.append(f"- **Fulfillment Rate**: {orders['overall_fulfillment_rate']:.1%}")

        # Customer summary
        if 'customer_metrics' in customers:
            cm = customers['customer_metrics']
            lines.append("")
            lines.append("## Customer Insights")
            lines.append(f"- **Total Customers**: {cm['total_customers']}")
            lines.append(f"- **Avg Orders/Customer**: {cm['avg_orders_per_customer']:.1f}")

        # Top recommendations
        lines.append("")
        lines.append("## Key Recommendations")
        all_recs = []
        for key in ['profit', 'efficiency', 'cost', 'production', 'waste', 'orders', 'customers']:
            data = locals()[key]
            if 'recommendations' in data:
                all_recs.extend(data['recommendations'])
            elif 'insights' in data:
                all_recs.extend(data['insights'])

        for i, rec in enumerate(all_recs[:5], 1):
            lines.append(f"{i}. {rec}")

        return "\n".join(lines)

    def _refine_summary(self, raw_summary: str, provider: str) -> str:
        """
        Refine raw summary using HoloLoom metaprompt framework.

        Transforms raw metrics into executive-quality intelligence brief.

        Args:
            raw_summary: Raw summary text
            provider: LLM provider ("anthropic", "google", "openai")

        Returns:
            Refined executive summary
        """

        try:
            self.logger.info(f"Refining summary using HoloLoom (provider: {provider})")

            # Create temporary config
            config = Config.fast()
            config.llm_provider = provider

            # Apply refinement with executive brief instructions
            refinement_instructions = (
                f"{raw_summary}\n\n"
                "Transform this raw data into an executive-quality daily intelligence brief:\n"
                "- Use clear, concise language\n"
                "- Highlight critical insights first\n"
                "- Provide context for metrics\n"
                "- Make recommendations actionable\n"
                "- Use professional tone (not overly formal)\n"
                "- Structure for quick scanning\n"
                "- Preserve all key metrics and numbers\n"
            )

            refined = create_metaprompt_auto(
                request=refinement_instructions,
                config=config,
                confidence_threshold=0.7
            )

            self.logger.info(
                f"Summary refined: {len(raw_summary)} → {len(refined)} chars "
                f"({round(len(refined) / len(raw_summary), 1)}x expansion)"
            )

            return refined

        except Exception as e:
            self.logger.error(
                f"Summary refinement failed: {e}. Using raw summary.",
                exc_info=True
            )
            return raw_summary


if __name__ == "__main__":
    print("Intelligence Engine module")
    print("Usage: Import and use with SyncManager instance")
    print("\nExample:")
    print("  from elle.coz.intelligence import IntelligenceEngine")
    print("  from elle.coz.sync_manager import SyncManager")
    print("")
    print("  sync = SyncManager()")
    print("  sync.parse_all()")
    print("  intelligence = IntelligenceEngine(sync)")
    print("  profit = intelligence.analyze_profit()")
