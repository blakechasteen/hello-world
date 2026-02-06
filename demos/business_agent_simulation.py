#!/usr/bin/env python3
"""
HoloLoom Business Agent Simulation (Dogfood Edition)
=====================================================
Date: 2026-02-06

Models a B2B AI SaaS company (HoloLoom itself) using HoloLoom's own
memory systems as the cognitive architecture for each business agent.

Each agent has:
- A Knowledge Graph (KG) — persistent memory of business entities/relationships
- Spring Dynamics — physics-based activation spreading across knowledge
- Thompson Sampling — Bayesian explore/exploit for strategic decisions
- Hot Pattern Tracking — learns which strategies work over time

Agents:
  1. Sales Agent      — pursues leads, closes deals, learns conversion patterns
  2. Marketing Agent  — generates leads, experiments with channels
  3. Product Agent    — ships features, responds to churn signals
  4. Support Agent    — handles tickets, detects churn risk

The simulation runs monthly ticks through 24 months of business operation.
"""

import math
import random
import json
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.memory.spring_dynamics import SpringDynamics, SpringConfig
from HoloLoom.config import Config
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# Thompson Sampling Bandit (from HoloLoom's core algorithm)
# ─────────────────────────────────────────────────────────────────────

class ThompsonBandit:
    """Bayesian bandit for explore/exploit decisions.

    Each arm tracks Beta(alpha, beta) posterior.
    On select: sample from posterior, pick highest.
    On update: success -> alpha += reward, failure -> beta += (1 - reward).
    """

    def __init__(self, arms: List[str]):
        self.arms = arms
        self.alpha = {arm: 1.0 for arm in arms}
        self.beta = {arm: 1.0 for arm in arms}
        self.pulls = {arm: 0 for arm in arms}
        self.total_reward = {arm: 0.0 for arm in arms}

    def select(self) -> str:
        """Sample from posterior, return best arm."""
        samples = {
            arm: np.random.beta(self.alpha[arm], self.beta[arm])
            for arm in self.arms
        }
        return max(samples, key=samples.get)

    def update(self, arm: str, reward: float):
        """Bayesian update: success strengthens alpha, failure strengthens beta."""
        self.pulls[arm] += 1
        self.total_reward[arm] += reward
        self.alpha[arm] += reward
        self.beta[arm] += (1.0 - reward)

    def expected_reward(self, arm: str) -> float:
        return self.alpha[arm] / (self.alpha[arm] + self.beta[arm])

    def stats(self) -> Dict:
        return {
            arm: {
                "pulls": self.pulls[arm],
                "expected": round(self.expected_reward(arm), 3),
                "alpha": round(self.alpha[arm], 1),
                "beta": round(self.beta[arm], 1),
            }
            for arm in self.arms
        }


# ─────────────────────────────────────────────────────────────────────
# Hot Pattern Tracker (from HoloLoom's recursive learning)
# ─────────────────────────────────────────────────────────────────────

class HotPatternTracker:
    """Tracks which strategies/patterns produce good outcomes.

    Heat = access_count * success_rate * avg_confidence * decay
    Decay = 0.95^months_since_last_success
    """

    def __init__(self, decay_rate: float = 0.95):
        self.patterns: Dict[str, dict] = {}
        self.decay_rate = decay_rate

    def record(self, pattern: str, success: bool, confidence: float, month: int):
        if pattern not in self.patterns:
            self.patterns[pattern] = {
                "accesses": 0, "successes": 0,
                "total_confidence": 0.0, "last_month": month
            }
        p = self.patterns[pattern]
        p["accesses"] += 1
        if success:
            p["successes"] += 1
            p["last_month"] = month
        p["total_confidence"] += confidence

    def get_heat(self, pattern: str, current_month: int) -> float:
        if pattern not in self.patterns:
            return 0.0
        p = self.patterns[pattern]
        if p["accesses"] == 0:
            return 0.0
        success_rate = p["successes"] / p["accesses"]
        avg_conf = p["total_confidence"] / p["accesses"]
        months_cold = current_month - p["last_month"]
        decay = self.decay_rate ** months_cold
        return p["accesses"] * success_rate * avg_conf * decay

    def top_patterns(self, current_month: int, n: int = 5) -> List[Tuple[str, float]]:
        heats = [(p, self.get_heat(p, current_month)) for p in self.patterns]
        return sorted(heats, key=lambda x: -x[1])[:n]


# ─────────────────────────────────────────────────────────────────────
# Business Agent (HoloLoom-brained)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class AgentProvenance:
    """Spacetime-lite: records what the agent decided and why."""
    month: int
    agent: str
    action: str
    strategy: str
    confidence: float
    outcome: dict
    reasoning: str


class BusinessAgent:
    """A business agent with HoloLoom cognitive architecture.

    Memory: KG (knowledge graph)
    Activation: SpringDynamics (physics-based spreading)
    Decisions: ThompsonBandit (Bayesian explore/exploit)
    Learning: HotPatternTracker (what works)
    Provenance: AgentProvenance log (audit trail)
    """

    def __init__(self, name: str, strategies: List[str]):
        self.name = name
        self.kg = KG()
        self.spring_config = SpringConfig(
            stiffness=0.8, damping=0.85, decay=0.99,
            max_iterations=100, convergence_epsilon=1e-3
        )
        self.bandit = ThompsonBandit(strategies)
        self.hot_patterns = HotPatternTracker()
        self.provenance: List[AgentProvenance] = []
        self.metrics: Dict[str, List[float]] = {}

    def record_knowledge(self, edges: List[KGEdge]):
        """Add business knowledge to the agent's memory."""
        self.kg.add_edges(edges)

    def spread_activation(self, seed: Dict[str, float]) -> Dict[str, float]:
        """Use SpringDynamics to find related knowledge."""
        if self.kg.G.number_of_nodes() < 2:
            return seed
        sd = SpringDynamics(self.kg, self.spring_config)
        sd.activate_nodes(seed)
        result = sd.propagate()
        return dict(zip(result.activated_nodes,
                        [result.node_activations[n] for n in result.activated_nodes]))

    def decide(self) -> str:
        """Thompson Sampling: choose strategy."""
        return self.bandit.select()

    def learn(self, strategy: str, reward: float, pattern: str, month: int):
        """Update from outcome."""
        self.bandit.update(strategy, reward)
        self.hot_patterns.record(
            pattern, success=(reward > 0.5),
            confidence=reward, month=month
        )

    def log_action(self, month: int, action: str, strategy: str,
                   confidence: float, outcome: dict, reasoning: str):
        """Spacetime provenance logging."""
        self.provenance.append(AgentProvenance(
            month=month, agent=self.name, action=action,
            strategy=strategy, confidence=confidence,
            outcome=outcome, reasoning=reasoning
        ))

    def track_metric(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)


# ─────────────────────────────────────────────────────────────────────
# Market Environment
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Company:
    """The simulated HoloLoom company state."""
    mrr: float = 0.0                 # Monthly Recurring Revenue
    customers: int = 0
    leads: int = 0
    churn_rate: float = 0.05         # 5% monthly churn
    avg_deal_size: float = 4900.0    # ~$49/user * 100 users avg
    enterprise_deal_size: float = 50000.0
    open_tickets: int = 0
    nps: float = 50.0                # Net Promoter Score (0-100)
    features_shipped: int = 0
    github_stars: int = 150
    brand_awareness: float = 0.01    # 0-1 scale

    # Cumulative
    total_revenue: float = 0.0
    total_customers_ever: int = 0
    month: int = 0


class MarketEnvironment:
    """Simulates market dynamics for a B2B AI SaaS."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.tam = 50000              # Total addressable market (companies)
        self.market_growth = 0.03     # 3% monthly TAM growth (AI is hot)
        self.competition_pressure = 0.02  # Increases over time

    def generate_organic_leads(self, company: Company) -> int:
        """Organic leads from brand awareness + GitHub stars."""
        base = company.brand_awareness * self.tam * 0.001
        star_bonus = math.log1p(company.github_stars) * 0.5
        noise = self.rng.gauss(0, max(1, base * 0.3))
        return max(0, int(base + star_bonus + noise))

    def lead_to_conversion_rate(self, company: Company) -> float:
        """How likely a lead converts. Depends on NPS, features, brand."""
        base = 0.05  # 5% base conversion
        nps_boost = (company.nps - 50) / 500  # NPS above 50 helps
        feature_boost = min(0.05, company.features_shipped * 0.002)
        competition = -self.competition_pressure * company.month * 0.1
        return max(0.01, min(0.25, base + nps_boost + feature_boost + competition))

    def churn_probability(self, company: Company) -> float:
        """Monthly churn rate. Bad support/NPS increases churn."""
        base = 0.04
        support_penalty = max(0, (company.open_tickets - 5) * 0.005)
        nps_effect = (50 - company.nps) / 1000
        return max(0.01, min(0.15, base + support_penalty + nps_effect))

    def tick(self, company: Company):
        """Advance market by one month."""
        self.tam = int(self.tam * (1 + self.market_growth))
        self.competition_pressure += 0.001
        company.month += 1


# ─────────────────────────────────────────────────────────────────────
# Simulation Engine
# ─────────────────────────────────────────────────────────────────────

class BusinessSimulation:
    """Runs 24-month business simulation with HoloLoom-brained agents."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.company = Company()
        self.market = MarketEnvironment(seed)

        # Create agents with HoloLoom cognitive architecture
        self.marketing = BusinessAgent("Marketing", [
            "content_marketing", "paid_ads", "conference_talks",
            "open_source_community", "partnerships"
        ])
        self.sales = BusinessAgent("Sales", [
            "product_led_growth", "outbound_enterprise",
            "inbound_demo", "free_trial_conversion"
        ])
        self.product = BusinessAgent("Product", [
            "ship_core_features", "improve_onboarding",
            "reduce_complexity", "add_integrations", "performance_optimization"
        ])
        self.support = BusinessAgent("Support", [
            "proactive_outreach", "knowledge_base",
            "fast_response", "escalation_to_engineering"
        ])

        self.agents = [self.marketing, self.sales, self.product, self.support]
        self.monthly_log: List[dict] = []

        # Seed initial knowledge graphs
        self._seed_knowledge()

    def _seed_knowledge(self):
        """Give each agent initial business knowledge."""
        # Marketing knows about channels and audience
        self.marketing.record_knowledge([
            KGEdge("content_marketing", "developer_trust", "LEADS_TO", 0.8),
            KGEdge("developer_trust", "organic_leads", "LEADS_TO", 0.9),
            KGEdge("paid_ads", "quick_leads", "LEADS_TO", 0.7),
            KGEdge("quick_leads", "low_quality", "LEADS_TO", 0.6),
            KGEdge("open_source_community", "github_stars", "LEADS_TO", 0.9),
            KGEdge("github_stars", "developer_trust", "LEADS_TO", 0.8),
            KGEdge("conference_talks", "brand_awareness", "LEADS_TO", 0.7),
            KGEdge("partnerships", "enterprise_leads", "LEADS_TO", 0.6),
        ])

        # Sales knows about deal dynamics
        self.sales.record_knowledge([
            KGEdge("product_led_growth", "self_serve", "USES", 0.9),
            KGEdge("self_serve", "small_deals", "LEADS_TO", 0.8),
            KGEdge("outbound_enterprise", "large_deals", "LEADS_TO", 0.7),
            KGEdge("large_deals", "long_sales_cycle", "LEADS_TO", 0.8),
            KGEdge("inbound_demo", "qualified_leads", "LEADS_TO", 0.85),
            KGEdge("free_trial_conversion", "product_experience", "USES", 0.9),
            KGEdge("product_experience", "conversion", "LEADS_TO", 0.7),
        ])

        # Product knows about feature impact
        self.product.record_knowledge([
            KGEdge("ship_core_features", "product_value", "LEADS_TO", 0.9),
            KGEdge("improve_onboarding", "activation_rate", "LEADS_TO", 0.85),
            KGEdge("reduce_complexity", "developer_experience", "LEADS_TO", 0.8),
            KGEdge("add_integrations", "market_reach", "LEADS_TO", 0.7),
            KGEdge("performance_optimization", "enterprise_readiness", "LEADS_TO", 0.8),
            KGEdge("product_value", "retention", "LEADS_TO", 0.9),
            KGEdge("activation_rate", "conversion", "LEADS_TO", 0.85),
        ])

        # Support knows about customer health
        self.support.record_knowledge([
            KGEdge("proactive_outreach", "early_warning", "LEADS_TO", 0.8),
            KGEdge("early_warning", "churn_prevention", "LEADS_TO", 0.7),
            KGEdge("knowledge_base", "self_service", "LEADS_TO", 0.75),
            KGEdge("self_service", "ticket_reduction", "LEADS_TO", 0.8),
            KGEdge("fast_response", "customer_satisfaction", "LEADS_TO", 0.9),
            KGEdge("customer_satisfaction", "nps_improvement", "LEADS_TO", 0.85),
            KGEdge("escalation_to_engineering", "bug_fixes", "LEADS_TO", 0.7),
        ])

    def _run_marketing(self, month: int) -> dict:
        """Marketing agent acts: choose channel, generate leads."""
        # Spread activation from current focus
        activation = self.marketing.spread_activation({"leads": 1.0, "brand_awareness": 0.8})

        # Thompson Sampling: pick channel
        channel = self.marketing.decide()

        # Simulate channel effectiveness
        effectiveness = {
            "content_marketing": lambda: self.rng.gauss(8, 3),
            "paid_ads": lambda: self.rng.gauss(15, 8),
            "conference_talks": lambda: self.rng.gauss(5, 4) + (3 if month > 6 else 0),
            "open_source_community": lambda: self.rng.gauss(4, 2) + math.log1p(self.company.github_stars) * 0.3,
            "partnerships": lambda: self.rng.gauss(3, 5) + (8 if month > 12 else 0),
        }

        new_leads = max(0, int(effectiveness[channel]()))
        organic = self.market.generate_organic_leads(self.company)
        total_leads = new_leads + organic

        # Brand awareness growth
        awareness_boost = {
            "content_marketing": 0.005,
            "paid_ads": 0.002,
            "conference_talks": 0.008,
            "open_source_community": 0.006,
            "partnerships": 0.004,
        }[channel]

        self.company.leads += total_leads
        self.company.brand_awareness = min(1.0, self.company.brand_awareness + awareness_boost)
        self.company.github_stars += self.rng.randint(5, 25) if channel == "open_source_community" else self.rng.randint(1, 5)

        # Reward based on cost-efficiency of leads
        reward = min(1.0, new_leads / 15)  # 15 leads = perfect score
        self.marketing.learn(channel, reward, f"channel:{channel}", month)

        # Add learned relationship to KG
        if reward > 0.7:
            self.marketing.record_knowledge([
                KGEdge(channel, f"month_{month}_success", "LEADS_TO", reward)
            ])

        result = {
            "channel": channel, "new_leads": new_leads,
            "organic_leads": organic, "total_leads": total_leads,
            "awareness": round(self.company.brand_awareness, 3),
        }

        self.marketing.log_action(
            month, "generate_leads", channel, reward, result,
            f"Thompson selected {channel} (E[r]={self.marketing.bandit.expected_reward(channel):.2f}). "
            f"Activated {len(activation)} knowledge nodes."
        )
        self.marketing.track_metric("leads_generated", total_leads)
        return result

    def _run_sales(self, month: int) -> dict:
        """Sales agent acts: pursue leads, close deals."""
        strategy = self.sales.decide()
        available_leads = self.company.leads
        conversion_rate = self.market.lead_to_conversion_rate(self.company)

        # Strategy modifiers
        modifiers = {
            "product_led_growth": {"rate_mult": 1.2, "deal_mult": 0.6, "leads_used": 0.5},
            "outbound_enterprise": {"rate_mult": 0.4, "deal_mult": 5.0, "leads_used": 0.2},
            "inbound_demo": {"rate_mult": 1.5, "deal_mult": 1.0, "leads_used": 0.7},
            "free_trial_conversion": {"rate_mult": 1.0, "deal_mult": 0.8, "leads_used": 0.6},
        }
        mod = modifiers[strategy]

        leads_worked = int(available_leads * mod["leads_used"])
        effective_rate = conversion_rate * mod["rate_mult"]
        effective_rate += self.rng.gauss(0, 0.02)
        effective_rate = max(0.01, min(0.30, effective_rate))

        new_customers = int(leads_worked * effective_rate)
        deal_size = self.company.avg_deal_size * mod["deal_mult"]
        deal_size *= self.rng.uniform(0.7, 1.3)

        new_mrr = new_customers * (deal_size / 12)  # Annual -> monthly
        self.company.customers += new_customers
        self.company.total_customers_ever += new_customers
        self.company.mrr += new_mrr
        self.company.leads = max(0, self.company.leads - leads_worked)

        # Reward: revenue efficiency
        reward = min(1.0, new_mrr / 5000) if leads_worked > 0 else 0.0
        self.sales.learn(strategy, reward, f"sales:{strategy}", month)

        if new_customers > 0:
            self.sales.record_knowledge([
                KGEdge(strategy, f"deal_m{month}", "LEADS_TO", reward),
                KGEdge(f"deal_m{month}", "revenue", "LEADS_TO", min(1.0, new_mrr / 10000)),
            ])

        result = {
            "strategy": strategy, "leads_worked": leads_worked,
            "new_customers": new_customers, "new_mrr": round(new_mrr, 2),
            "conversion_rate": round(effective_rate, 3),
        }

        self.sales.log_action(
            month, "close_deals", strategy, reward, result,
            f"Worked {leads_worked} leads at {effective_rate:.1%} conversion. "
            f"Won {new_customers} customers, ${new_mrr:.0f} new MRR."
        )
        self.sales.track_metric("new_mrr", new_mrr)
        self.sales.track_metric("customers_won", new_customers)
        return result

    def _run_product(self, month: int) -> dict:
        """Product agent acts: ship features, improve product."""
        focus = self.product.decide()

        # Feature shipping affects different metrics
        effects = {
            "ship_core_features": {"features": 2, "nps": 2, "churn": -0.005},
            "improve_onboarding": {"features": 1, "nps": 3, "churn": -0.003},
            "reduce_complexity": {"features": 0, "nps": 4, "churn": -0.008},
            "add_integrations": {"features": 1, "nps": 1, "churn": -0.002},
            "performance_optimization": {"features": 0, "nps": 2, "churn": -0.004},
        }
        effect = effects[focus]
        noise = self.rng.gauss(0, 1)

        self.company.features_shipped += effect["features"]
        self.company.nps = max(0, min(100, self.company.nps + effect["nps"] + noise))
        self.company.churn_rate = max(0.01, self.company.churn_rate + effect["churn"])

        # Reward: NPS improvement + churn reduction
        reward = min(1.0, (effect["nps"] + abs(effect["churn"]) * 100) / 10)
        self.product.learn(focus, reward, f"product:{focus}", month)

        result = {
            "focus": focus, "features_shipped": effect["features"],
            "nps_delta": round(effect["nps"] + noise, 1),
            "nps": round(self.company.nps, 1),
        }

        self.product.log_action(
            month, "ship_product", focus, reward, result,
            f"Focused on {focus}. NPS now {self.company.nps:.0f}."
        )
        self.product.track_metric("nps", self.company.nps)
        return result

    def _run_support(self, month: int) -> dict:
        """Support agent acts: handle tickets, prevent churn."""
        approach = self.support.decide()

        # Tickets generated by customer base
        new_tickets = max(0, int(self.company.customers * 0.15 + self.rng.gauss(0, 3)))
        self.company.open_tickets += new_tickets

        # Approach effectiveness
        resolution = {
            "proactive_outreach": {"resolved_pct": 0.6, "nps_boost": 2, "churn_prevention": 0.3},
            "knowledge_base": {"resolved_pct": 0.7, "nps_boost": 1, "churn_prevention": 0.1},
            "fast_response": {"resolved_pct": 0.8, "nps_boost": 3, "churn_prevention": 0.2},
            "escalation_to_engineering": {"resolved_pct": 0.5, "nps_boost": 1, "churn_prevention": 0.15},
        }
        eff = resolution[approach]

        resolved = int(self.company.open_tickets * eff["resolved_pct"])
        self.company.open_tickets = max(0, self.company.open_tickets - resolved)
        self.company.nps = min(100, self.company.nps + eff["nps_boost"] * 0.3)

        # Churn calculation
        base_churn = self.market.churn_probability(self.company)
        prevented = base_churn * eff["churn_prevention"]
        actual_churn_rate = base_churn - prevented
        churned = int(self.company.customers * actual_churn_rate)
        lost_mrr = churned * (self.company.mrr / max(1, self.company.customers))
        self.company.customers = max(0, self.company.customers - churned)
        self.company.mrr = max(0, self.company.mrr - lost_mrr)

        reward = min(1.0, eff["churn_prevention"] + (resolved / max(1, new_tickets + self.company.open_tickets)))
        self.support.learn(approach, reward, f"support:{approach}", month)

        result = {
            "approach": approach, "new_tickets": new_tickets,
            "resolved": resolved, "open_tickets": self.company.open_tickets,
            "churned": churned, "lost_mrr": round(lost_mrr, 2),
            "churn_rate": round(actual_churn_rate, 3),
        }

        self.support.log_action(
            month, "handle_support", approach, reward, result,
            f"Resolved {resolved}/{new_tickets + self.company.open_tickets} tickets. "
            f"Churned: {churned} customers (${lost_mrr:.0f} MRR lost)."
        )
        self.support.track_metric("churn_rate", actual_churn_rate)
        self.support.track_metric("open_tickets", self.company.open_tickets)
        return result

    def run_month(self) -> dict:
        """Execute one month of business operations."""
        self.market.tick(self.company)
        month = self.company.month

        # Each agent acts
        marketing_result = self._run_marketing(month)
        sales_result = self._run_sales(month)
        product_result = self._run_product(month)
        support_result = self._run_support(month)

        # Revenue accounting
        self.company.total_revenue += self.company.mrr

        snapshot = {
            "month": month,
            "mrr": round(self.company.mrr, 2),
            "arr": round(self.company.mrr * 12, 2),
            "customers": self.company.customers,
            "leads": self.company.leads,
            "nps": round(self.company.nps, 1),
            "churn_rate": round(self.company.churn_rate, 3),
            "features_shipped": self.company.features_shipped,
            "github_stars": self.company.github_stars,
            "brand_awareness": round(self.company.brand_awareness, 3),
            "total_revenue": round(self.company.total_revenue, 2),
            "marketing": marketing_result,
            "sales": sales_result,
            "product": product_result,
            "support": support_result,
        }
        self.monthly_log.append(snapshot)
        return snapshot

    def run(self, months: int = 24) -> List[dict]:
        """Run full simulation."""
        print(f"\n{'='*70}")
        print(f"  HOLOLOOM BUSINESS SIMULATION (DOGFOOD EDITION)")
        print(f"  {months}-Month Simulation with 4 HoloLoom-Brained Agents")
        print(f"{'='*70}\n")

        for m in range(months):
            snapshot = self.run_month()
            month = snapshot["month"]

            # Print monthly summary
            bar_len = min(50, int(snapshot["mrr"] / 200))
            mrr_bar = "█" * bar_len + "░" * (50 - bar_len)

            print(f"Month {month:2d} │ MRR ${snapshot['mrr']:>9,.0f} │{mrr_bar}│ "
                  f"Cust: {snapshot['customers']:>4d} │ NPS: {snapshot['nps']:>4.0f} │ "
                  f"Churn: {snapshot['churn_rate']:.1%}")

            # Quarterly deep dive
            if month % 6 == 0:
                self._print_quarterly_report(month)

        self._print_final_report(months)
        return self.monthly_log

    def _print_quarterly_report(self, month: int):
        """Print agent learning state every 6 months."""
        print(f"\n  ┌─── Agent Intelligence Report (Month {month}) ───┐")
        for agent in self.agents:
            print(f"  │")
            print(f"  │ {agent.name} Agent")
            print(f"  │   KG: {agent.kg.G.number_of_nodes()} nodes, "
                  f"{agent.kg.G.number_of_edges()} edges")
            print(f"  │   Thompson Sampling priors:")
            for arm, stats in agent.bandit.stats().items():
                if stats["pulls"] > 0:
                    print(f"  │     {arm:30s} E[r]={stats['expected']:.2f} "
                          f"(pulls={stats['pulls']}, α={stats['alpha']}, β={stats['beta']})")
            hot = agent.hot_patterns.top_patterns(month, n=3)
            if hot:
                print(f"  │   Hot patterns:")
                for pattern, heat in hot:
                    print(f"  │     {pattern:30s} heat={heat:.2f}")
        print(f"  └{'─'*50}┘\n")

    def _print_final_report(self, months: int):
        """Print comprehensive final report."""
        c = self.company
        print(f"\n{'='*70}")
        print(f"  FINAL REPORT — {months} Months Complete")
        print(f"{'='*70}")
        print(f"\n  Company Metrics:")
        print(f"    MRR:              ${c.mrr:>12,.0f}")
        print(f"    ARR:              ${c.mrr * 12:>12,.0f}")
        print(f"    Total Revenue:    ${c.total_revenue:>12,.0f}")
        print(f"    Customers:        {c.customers:>12,d}")
        print(f"    Total Ever:       {c.total_customers_ever:>12,d}")
        print(f"    NPS:              {c.nps:>12.0f}")
        print(f"    Churn Rate:       {c.churn_rate:>11.1%}")
        print(f"    GitHub Stars:     {c.github_stars:>12,d}")
        print(f"    Brand Awareness:  {c.brand_awareness:>11.1%}")
        print(f"    Features Shipped: {c.features_shipped:>12,d}")

        print(f"\n  Agent Knowledge Graphs (Learned Business Intelligence):")
        for agent in self.agents:
            print(f"    {agent.name:12s}: {agent.kg.G.number_of_nodes():3d} nodes, "
                  f"{agent.kg.G.number_of_edges():3d} edges, "
                  f"{len(agent.provenance):3d} decisions logged")

        print(f"\n  Thompson Sampling Convergence (What Each Agent Learned):")
        for agent in self.agents:
            print(f"\n    {agent.name}:")
            stats = agent.bandit.stats()
            ranked = sorted(stats.items(), key=lambda x: -x[1]["expected"])
            for arm, s in ranked:
                bar = "▓" * int(s["expected"] * 20) + "░" * (20 - int(s["expected"] * 20))
                print(f"      {arm:30s} {bar} {s['expected']:.2f} ({s['pulls']} pulls)")

        print(f"\n  Provenance: {sum(len(a.provenance) for a in self.agents)} "
              f"total decisions logged with full audit trail")
        print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sim = BusinessSimulation(seed=42)
    results = sim.run(months=24)

    # Save full results
    output_path = os.path.join(os.path.dirname(__file__), "simulation_results.json")
    serializable = []
    for r in results:
        s = dict(r)
        serializable.append(s)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"  Full results saved to: {output_path}")
