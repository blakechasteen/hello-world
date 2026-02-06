#!/usr/bin/env python3
"""
HoloLoom Business Agent Simulation v2 — Real Multi-Agent Framework
===================================================================
Date: 2026-02-06

Upgrades from v1 (hand-rolled Thompson Sampling + KG) to REAL HoloLoom
agent infrastructure:

  - AgentOrchestrator: Per-agent Trinity working memory (semantic 244D +
    graph activation + warp tension)
  - WorkingMemoryLearner: Persistent pattern learning from success
  - MessageBus: Async inter-agent communication
  - ConversationManager: Budgeted, safety-guardrailed dialogue
  - ZeroCopyMatryoshkaEmbeddings: Multi-scale embeddings (96/192/384/768)
  - MCTS-backed decision engine: UCT + breakthrough bias
  - Shared KG: Agents share a knowledge graph but focus on different regions

Each of the 4 business agents (Marketing, Sales, Product, Support) is a real
AgentOrchestrator with a custom AgentProfile, operating on the same shared
knowledge graph but attending to different semantic regions via Trinity WM.

The agents communicate via MessageBus to share insights (e.g., Support tells
Product about churn-causing bugs, Marketing shares lead quality data with
Sales).
"""

import asyncio
import math
import random
import json
import sys
import os
import warnings
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

# ─── HoloLoom imports (real agent infrastructure) ─────────────────────
from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.memory.spring_dynamics import SpringDynamics, SpringConfig
from HoloLoom.config import Config
from HoloLoom.agents.types import AgentProfile, AgentDomain, AgentStats
from HoloLoom.agents.orchestrator import AgentOrchestrator
from HoloLoom.agents.working_memory import AgentWorkingMemory
from HoloLoom.agents.learner import WorkingMemoryLearner
from HoloLoom.agents.multi_agent_communication import (
    MessageBus, ConversationManager, BudgetManager, SafetyGuardrails as CommSafetyGuardrails,
    Budget, MessageType, Message
)
from HoloLoom.agents.mcts_core import MCTSNode, MCTSEngine, MCTSStateSpace
from HoloLoom.protocols.types import Query
from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace
from HoloLoom.embedding.zero_copy import ZeroCopyMatryoshkaEmbeddings


# =====================================================================
# MCTS State Space for Business Strategy Selection
# =====================================================================

class StrategyStateSpace(MCTSStateSpace):
    """MCTS state space for selecting business strategies.

    State: (current_strategy, month, cumulative_reward)
    Actions: available strategies
    Evaluation: simulated reward based on historical performance
    """

    def __init__(self, strategies: List[str], reward_history: Dict[str, List[float]]):
        self.strategies = strategies
        self.reward_history = reward_history  # strategy -> [rewards]

    def get_legal_actions(self, state) -> List:
        return list(self.strategies)

    def apply_action(self, state, action):
        return (action, state[1], state[2])

    async def evaluate(self, state) -> float:
        strategy = state[0]
        history = self.reward_history.get(strategy, [])
        if not history:
            return 0.5  # Uninformed prior
        # Thompson-style: sample from Beta approximation
        mean = np.mean(history)
        std = max(0.05, np.std(history))
        return max(0.0, min(1.0, np.random.normal(mean, std)))

    def is_terminal(self, state) -> bool:
        return True  # Single-step decisions

    def copy_state(self, state):
        return state


# =====================================================================
# Custom Business Agent Profiles
# =====================================================================

def create_marketing_profile() -> AgentProfile:
    return AgentProfile(
        agent_id="marketing_agent",
        name="Marketing Agent",
        domain=AgentDomain.GENERAL,
        system_prompt=(
            "You are a B2B SaaS marketing strategist focused on developer tools. "
            "Optimize for lead generation, brand awareness, and developer trust. "
            "Use data-driven channel selection."
        ),
        priorities=["lead_generation", "brand_awareness", "cost_efficiency"],
        semantic_dimensions=["Urgency", "Novelty", "Scope"],
        preferred_tools=["answer", "research"],
        context_window_size=12,
        attention_radius=0.30,
        momentum=0.75,
        propagation_decay=0.85,
        enable_learning=True,
        success_threshold=0.70,
    )


def create_sales_profile() -> AgentProfile:
    return AgentProfile(
        agent_id="sales_agent",
        name="Sales Agent",
        domain=AgentDomain.GENERAL,
        system_prompt=(
            "You are a B2B enterprise sales agent for AI/ML developer tools. "
            "Optimize for revenue, conversion rate, and deal size. "
            "Balance product-led growth with enterprise outbound."
        ),
        priorities=["revenue", "conversion_rate", "deal_velocity"],
        semantic_dimensions=["Directness", "Specificity", "Actionability"],
        preferred_tools=["answer", "research"],
        context_window_size=10,
        attention_radius=0.25,
        momentum=0.80,
        propagation_decay=0.80,
        enable_learning=True,
        success_threshold=0.65,
    )


def create_product_profile() -> AgentProfile:
    return AgentProfile(
        agent_id="product_agent",
        name="Product Agent",
        domain=AgentDomain.ARCHITECTURE,
        system_prompt=(
            "You are a product manager for an AI infrastructure platform. "
            "Optimize for user retention, feature impact, and developer experience. "
            "Ship high-impact features, reduce complexity."
        ),
        priorities=["retention", "developer_experience", "feature_impact"],
        semantic_dimensions=["Complexity", "Abstraction", "Technicality"],
        preferred_tools=["answer", "research"],
        context_window_size=15,
        attention_radius=0.35,
        momentum=0.65,
        propagation_decay=0.85,
        enable_learning=True,
        success_threshold=0.70,
    )


def create_support_profile() -> AgentProfile:
    return AgentProfile(
        agent_id="support_agent",
        name="Support Agent",
        domain=AgentDomain.GENERAL,
        system_prompt=(
            "You are a customer success agent for a developer tools company. "
            "Optimize for customer satisfaction, churn prevention, and ticket resolution. "
            "Proactively identify at-risk customers."
        ),
        priorities=["customer_satisfaction", "churn_prevention", "resolution_speed"],
        semantic_dimensions=["Warmth", "Urgency", "Directness"],
        preferred_tools=["answer"],
        context_window_size=10,
        attention_radius=0.30,
        momentum=0.70,
        propagation_decay=0.90,
        enable_learning=True,
        success_threshold=0.70,
    )


# =====================================================================
# Market Environment (same as v1)
# =====================================================================

@dataclass
class Company:
    """The simulated HoloLoom company state."""
    mrr: float = 0.0
    customers: int = 0
    leads: int = 0
    churn_rate: float = 0.05
    avg_deal_size: float = 4900.0
    enterprise_deal_size: float = 50000.0
    open_tickets: int = 0
    nps: float = 50.0
    features_shipped: int = 0
    github_stars: int = 150
    brand_awareness: float = 0.01
    total_revenue: float = 0.0
    total_customers_ever: int = 0
    month: int = 0


class MarketEnvironment:
    """Simulates market dynamics for a B2B AI SaaS."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.tam = 50000
        self.market_growth = 0.03
        self.competition_pressure = 0.02

    def generate_organic_leads(self, company: Company) -> int:
        base = company.brand_awareness * self.tam * 0.001
        star_bonus = math.log1p(company.github_stars) * 0.5
        noise = self.rng.gauss(0, max(1, base * 0.3))
        return max(0, int(base + star_bonus + noise))

    def lead_to_conversion_rate(self, company: Company) -> float:
        base = 0.05
        nps_boost = (company.nps - 50) / 500
        feature_boost = min(0.05, company.features_shipped * 0.002)
        competition = -self.competition_pressure * company.month * 0.1
        return max(0.01, min(0.25, base + nps_boost + feature_boost + competition))

    def churn_probability(self, company: Company) -> float:
        base = 0.04
        support_penalty = max(0, (company.open_tickets - 5) * 0.005)
        nps_effect = (50 - company.nps) / 1000
        return max(0.01, min(0.15, base + support_penalty + nps_effect))

    def tick(self, company: Company):
        self.tam = int(self.tam * (1 + self.market_growth))
        self.competition_pressure += 0.001
        company.month += 1


# =====================================================================
# HoloLoom Multi-Agent Business Simulation (v2)
# =====================================================================

class HoloLoomBusinessSimulation:
    """24-month simulation with REAL HoloLoom multi-agent framework.

    Each agent has:
      - AgentOrchestrator with custom AgentProfile
      - Trinity Working Memory (semantic 244D + graph activation + warp tension)
      - WorkingMemoryLearner (pattern persistence)
      - MCTS-backed strategy selection
      - Communication via MessageBus + ConversationManager
      - Shared Knowledge Graph (focus on different regions)
    """

    MARKETING_STRATEGIES = [
        "content_marketing", "paid_ads", "conference_talks",
        "open_source_community", "partnerships"
    ]
    SALES_STRATEGIES = [
        "product_led_growth", "outbound_enterprise",
        "inbound_demo", "free_trial_conversion"
    ]
    PRODUCT_STRATEGIES = [
        "ship_core_features", "improve_onboarding",
        "reduce_complexity", "add_integrations", "performance_optimization"
    ]
    SUPPORT_STRATEGIES = [
        "proactive_outreach", "knowledge_base",
        "fast_response", "escalation_to_engineering"
    ]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        np.random.seed(seed)
        self.company = Company()
        self.market = MarketEnvironment(seed)

        # ── Shared infrastructure ──────────────────────────────────
        self.shared_kg = KG()
        self.embedding = ZeroCopyMatryoshkaEmbeddings()
        self.persist_dir = tempfile.mkdtemp(prefix="hololoom_agents_")

        # ── Create real HoloLoom agents ────────────────────────────
        self.marketing = AgentOrchestrator(
            profile=create_marketing_profile(),
            shared_knowledge=self.shared_kg,
            embedding_model=self.embedding,
            persist_dir=Path(self.persist_dir) / "marketing",
        )
        self.sales = AgentOrchestrator(
            profile=create_sales_profile(),
            shared_knowledge=self.shared_kg,
            embedding_model=self.embedding,
            persist_dir=Path(self.persist_dir) / "sales",
        )
        self.product = AgentOrchestrator(
            profile=create_product_profile(),
            shared_knowledge=self.shared_kg,
            embedding_model=self.embedding,
            persist_dir=Path(self.persist_dir) / "product",
        )
        self.support = AgentOrchestrator(
            profile=create_support_profile(),
            shared_knowledge=self.shared_kg,
            embedding_model=self.embedding,
            persist_dir=Path(self.persist_dir) / "support",
        )

        self.agents = {
            "marketing": self.marketing,
            "sales": self.sales,
            "product": self.product,
            "support": self.support,
        }

        # ── MCTS engines per agent ─────────────────────────────────
        self.reward_history: Dict[str, Dict[str, List[float]]] = {
            "marketing": {s: [] for s in self.MARKETING_STRATEGIES},
            "sales": {s: [] for s in self.SALES_STRATEGIES},
            "product": {s: [] for s in self.PRODUCT_STRATEGIES},
            "support": {s: [] for s in self.SUPPORT_STRATEGIES},
        }

        # ── MessageBus + ConversationManager ───────────────────────
        self.message_bus = MessageBus()
        self.budget_mgr = BudgetManager(Budget(
            max_messages=20,
            max_duration_seconds=60.0,
            max_depth=4,
            max_conversations_per_hour=50,
        ))
        self.comm_safety = CommSafetyGuardrails(
            min_insight_length=10,
            similarity_threshold=0.85,
        )
        self.conv_mgr = ConversationManager(
            self.message_bus, self.budget_mgr, self.comm_safety
        )

        # Track inter-agent messages for reporting
        self.messages_sent: List[dict] = []
        self.monthly_log: List[dict] = []

        # Seed knowledge graph
        self._seed_knowledge()

    def _seed_knowledge(self):
        """Populate shared KG with business domain knowledge."""
        edges = [
            # Marketing knowledge
            KGEdge("content_marketing", "developer_trust", "LEADS_TO", 0.8),
            KGEdge("developer_trust", "organic_leads", "LEADS_TO", 0.9),
            KGEdge("paid_ads", "quick_leads", "LEADS_TO", 0.7),
            KGEdge("quick_leads", "low_quality", "LEADS_TO", 0.6),
            KGEdge("open_source_community", "github_stars", "LEADS_TO", 0.9),
            KGEdge("github_stars", "developer_trust", "LEADS_TO", 0.8),
            KGEdge("conference_talks", "brand_awareness", "LEADS_TO", 0.7),
            KGEdge("partnerships", "enterprise_leads", "LEADS_TO", 0.6),
            # Sales knowledge
            KGEdge("product_led_growth", "self_serve", "USES", 0.9),
            KGEdge("self_serve", "small_deals", "LEADS_TO", 0.8),
            KGEdge("outbound_enterprise", "large_deals", "LEADS_TO", 0.7),
            KGEdge("large_deals", "long_sales_cycle", "LEADS_TO", 0.8),
            KGEdge("inbound_demo", "qualified_leads", "LEADS_TO", 0.85),
            KGEdge("free_trial_conversion", "product_experience", "USES", 0.9),
            KGEdge("product_experience", "conversion", "LEADS_TO", 0.7),
            # Product knowledge
            KGEdge("ship_core_features", "product_value", "LEADS_TO", 0.9),
            KGEdge("improve_onboarding", "activation_rate", "LEADS_TO", 0.85),
            KGEdge("reduce_complexity", "developer_experience", "LEADS_TO", 0.8),
            KGEdge("add_integrations", "market_reach", "LEADS_TO", 0.7),
            KGEdge("performance_optimization", "enterprise_readiness", "LEADS_TO", 0.8),
            KGEdge("product_value", "retention", "LEADS_TO", 0.9),
            KGEdge("activation_rate", "conversion", "LEADS_TO", 0.85),
            # Support knowledge
            KGEdge("proactive_outreach", "early_warning", "LEADS_TO", 0.8),
            KGEdge("early_warning", "churn_prevention", "LEADS_TO", 0.7),
            KGEdge("knowledge_base", "self_service", "LEADS_TO", 0.75),
            KGEdge("self_service", "ticket_reduction", "LEADS_TO", 0.8),
            KGEdge("fast_response", "customer_satisfaction", "LEADS_TO", 0.9),
            KGEdge("customer_satisfaction", "nps_improvement", "LEADS_TO", 0.85),
            KGEdge("escalation_to_engineering", "bug_fixes", "LEADS_TO", 0.7),
            # Cross-domain connections
            KGEdge("retention", "mrr_growth", "LEADS_TO", 0.9),
            KGEdge("conversion", "new_revenue", "LEADS_TO", 0.85),
            KGEdge("churn_prevention", "retention", "LEADS_TO", 0.8),
            KGEdge("developer_experience", "retention", "LEADS_TO", 0.7),
            KGEdge("nps_improvement", "organic_leads", "LEADS_TO", 0.6),
        ]
        self.shared_kg.add_edges(edges)

    # ── MCTS-backed strategy selection ──────────────────────────────

    async def _select_strategy(self, agent_name: str, strategies: List[str]) -> str:
        """Use MCTS to select best strategy for an agent."""
        history = self.reward_history[agent_name]
        state_space = StrategyStateSpace(strategies, history)
        engine = MCTSEngine(state_space, exploration_weight=1.414, max_depth=1)

        initial_state = (None, self.company.month, 0.0)
        best_action, root = await engine.search(initial_state, n_simulations=30)

        if best_action is None:
            return self.rng.choice(strategies)
        return best_action

    def _record_reward(self, agent_name: str, strategy: str, reward: float):
        """Record reward for MCTS learning."""
        self.reward_history[agent_name][strategy].append(reward)

    # ── Agent Working Memory Operations ─────────────────────────────

    async def _attend_and_decide(
        self, agent: AgentOrchestrator, query_text: str
    ) -> List:
        """Have agent attend to a query and return relevant context."""
        query = Query(text=query_text)
        context = await agent.working_memory.attend_to(query)
        return context

    # ── Inter-Agent Communication ───────────────────────────────────

    async def _share_insight(
        self, from_name: str, to_name: str, content: str, topic: str
    ):
        """One agent shares insight with another via MessageBus."""
        conv_id = await self.conv_mgr.start_conversation(
            topic=topic,
            initiator=from_name,
            participants=[from_name, to_name],
        )
        if conv_id:
            msg = await self.conv_mgr.send_message(
                conversation_id=conv_id,
                from_agent=from_name,
                to_agent=to_name,
                message_type=MessageType.INSIGHT,
                content=content,
            )
            await self.conv_mgr.end_conversation(conv_id, reason="insight_delivered")
            if msg:
                self.messages_sent.append({
                    "from": from_name, "to": to_name,
                    "type": "insight", "content": content[:80],
                    "month": self.company.month,
                })

    # ── Monthly Agent Actions ───────────────────────────────────────

    async def _run_marketing(self, month: int) -> dict:
        # Agent attends to current business context
        context = await self._attend_and_decide(
            self.marketing,
            f"Generate leads for month {month}. Brand awareness {self.company.brand_awareness:.1%}. "
            f"Current leads: {self.company.leads}. GitHub stars: {self.company.github_stars}."
        )

        # MCTS strategy selection
        channel = await self._select_strategy("marketing", self.MARKETING_STRATEGIES)

        # Pin the selected strategy in working memory
        self.marketing.pin(channel, weight=0.6)

        # Simulate channel effectiveness (same economics as v1)
        effectiveness = {
            "content_marketing": lambda: self.rng.gauss(8, 3),
            "paid_ads": lambda: self.rng.gauss(15, 8),
            "conference_talks": lambda: self.rng.gauss(5, 4) + (3 if month > 6 else 0),
            "open_source_community": lambda: (
                self.rng.gauss(4, 2) + math.log1p(self.company.github_stars) * 0.3
            ),
            "partnerships": lambda: self.rng.gauss(3, 5) + (8 if month > 12 else 0),
        }

        new_leads = max(0, int(effectiveness[channel]()))
        organic = self.market.generate_organic_leads(self.company)
        total_leads = new_leads + organic

        awareness_boost = {
            "content_marketing": 0.005, "paid_ads": 0.002,
            "conference_talks": 0.008, "open_source_community": 0.006,
            "partnerships": 0.004,
        }[channel]

        self.company.leads += total_leads
        self.company.brand_awareness = min(1.0, self.company.brand_awareness + awareness_boost)
        self.company.github_stars += (
            self.rng.randint(5, 25) if channel == "open_source_community"
            else self.rng.randint(1, 5)
        )

        reward = min(1.0, new_leads / 15)
        self._record_reward("marketing", channel, reward)

        # Learn: add successful pattern to shared KG
        if reward > 0.7:
            self.shared_kg.add_edges([
                KGEdge(channel, f"m{month}_mktg_win", "LEADS_TO", reward)
            ])

        # Share insight with Sales about lead quality
        if total_leads > 10:
            await self._share_insight(
                "marketing", "sales",
                f"Month {month}: {channel} generated {total_leads} leads "
                f"(quality={'high' if channel == 'content_marketing' else 'mixed'})",
                "lead_quality"
            )

        return {
            "channel": channel, "new_leads": new_leads,
            "organic_leads": organic, "total_leads": total_leads,
            "awareness": round(self.company.brand_awareness, 3),
            "reward": round(reward, 3),
            "wm_context_size": len(context),
        }

    async def _run_sales(self, month: int) -> dict:
        context = await self._attend_and_decide(
            self.sales,
            f"Close deals in month {month}. Available leads: {self.company.leads}. "
            f"Current MRR: ${self.company.mrr:,.0f}. Customers: {self.company.customers}."
        )

        strategy = await self._select_strategy("sales", self.SALES_STRATEGIES)
        self.sales.pin(strategy, weight=0.6)

        available_leads = self.company.leads
        conversion_rate = self.market.lead_to_conversion_rate(self.company)

        modifiers = {
            "product_led_growth": {"rate_mult": 1.2, "deal_mult": 0.6, "leads_used": 0.5},
            "outbound_enterprise": {"rate_mult": 0.4, "deal_mult": 5.0, "leads_used": 0.2},
            "inbound_demo": {"rate_mult": 1.5, "deal_mult": 1.0, "leads_used": 0.7},
            "free_trial_conversion": {"rate_mult": 1.0, "deal_mult": 0.8, "leads_used": 0.6},
        }
        mod = modifiers[strategy]

        leads_worked = int(available_leads * mod["leads_used"])
        effective_rate = max(0.01, min(0.30,
            conversion_rate * mod["rate_mult"] + self.rng.gauss(0, 0.02)))

        new_customers = int(leads_worked * effective_rate)
        deal_size = self.company.avg_deal_size * mod["deal_mult"] * self.rng.uniform(0.7, 1.3)
        new_mrr = new_customers * (deal_size / 12)

        self.company.customers += new_customers
        self.company.total_customers_ever += new_customers
        self.company.mrr += new_mrr
        self.company.leads = max(0, self.company.leads - leads_worked)

        reward = min(1.0, new_mrr / 5000) if leads_worked > 0 else 0.0
        self._record_reward("sales", strategy, reward)

        if new_customers > 0:
            self.shared_kg.add_edges([
                KGEdge(strategy, f"m{month}_deal", "LEADS_TO", reward),
                KGEdge(f"m{month}_deal", "revenue", "LEADS_TO", min(1.0, new_mrr / 10000)),
            ])

        return {
            "strategy": strategy, "leads_worked": leads_worked,
            "new_customers": new_customers, "new_mrr": round(new_mrr, 2),
            "conversion_rate": round(effective_rate, 3),
            "reward": round(reward, 3),
            "wm_context_size": len(context),
        }

    async def _run_product(self, month: int) -> dict:
        context = await self._attend_and_decide(
            self.product,
            f"Ship product for month {month}. NPS: {self.company.nps:.0f}. "
            f"Churn: {self.company.churn_rate:.1%}. Features shipped: {self.company.features_shipped}."
        )

        focus = await self._select_strategy("product", self.PRODUCT_STRATEGIES)
        self.product.pin(focus, weight=0.6)

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

        reward = min(1.0, (effect["nps"] + abs(effect["churn"]) * 100) / 10)
        self._record_reward("product", focus, reward)

        # Share churn insights with Support
        if self.company.churn_rate > 0.06:
            await self._share_insight(
                "product", "support",
                f"Month {month}: Churn at {self.company.churn_rate:.1%}. "
                f"Focus on {focus}. Recommend proactive outreach.",
                "churn_risk"
            )

        return {
            "focus": focus, "features_shipped": effect["features"],
            "nps_delta": round(effect["nps"] + noise, 1),
            "nps": round(self.company.nps, 1),
            "reward": round(reward, 3),
            "wm_context_size": len(context),
        }

    async def _run_support(self, month: int) -> dict:
        context = await self._attend_and_decide(
            self.support,
            f"Handle support for month {month}. Customers: {self.company.customers}. "
            f"Open tickets: {self.company.open_tickets}. NPS: {self.company.nps:.0f}."
        )

        approach = await self._select_strategy("support", self.SUPPORT_STRATEGIES)
        self.support.pin(approach, weight=0.6)

        new_tickets = max(0, int(self.company.customers * 0.15 + self.rng.gauss(0, 3)))
        self.company.open_tickets += new_tickets

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

        base_churn = self.market.churn_probability(self.company)
        prevented = base_churn * eff["churn_prevention"]
        actual_churn_rate = base_churn - prevented
        churned = int(self.company.customers * actual_churn_rate)
        lost_mrr = churned * (self.company.mrr / max(1, self.company.customers))
        self.company.customers = max(0, self.company.customers - churned)
        self.company.mrr = max(0, self.company.mrr - lost_mrr)

        reward = min(1.0, eff["churn_prevention"] + (resolved / max(1, new_tickets + self.company.open_tickets)))
        self._record_reward("support", approach, reward)

        # Share NPS concern with Product if declining
        if self.company.nps < 55:
            await self._share_insight(
                "support", "product",
                f"Month {month}: NPS dropped to {self.company.nps:.0f}. "
                f"Top complaint: complexity. Recommend reduce_complexity focus.",
                "nps_decline"
            )

        return {
            "approach": approach, "new_tickets": new_tickets,
            "resolved": resolved, "open_tickets": self.company.open_tickets,
            "churned": churned, "lost_mrr": round(lost_mrr, 2),
            "churn_rate": round(actual_churn_rate, 3),
            "reward": round(reward, 3),
            "wm_context_size": len(context),
        }

    # ── Month + Simulation Loop ─────────────────────────────────────

    async def run_month(self) -> dict:
        self.market.tick(self.company)
        month = self.company.month

        # Each agent acts (async)
        marketing_result = await self._run_marketing(month)
        sales_result = await self._run_sales(month)
        product_result = await self._run_product(month)
        support_result = await self._run_support(month)

        # Relax all working memories (decay activations between months)
        for agent in self.agents.values():
            await agent.relax()

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

    async def run(self, months: int = 24) -> List[dict]:
        """Run the full simulation."""
        print(f"\n{'='*72}")
        print(f"  HOLOLOOM BUSINESS SIMULATION v2 — REAL MULTI-AGENT FRAMEWORK")
        print(f"  {months}-Month Simulation with HoloLoom AgentOrchestrator + MCTS + MessageBus")
        print(f"{'='*72}")
        print(f"  Shared KG: {self.shared_kg.G.number_of_nodes()} nodes, "
              f"{self.shared_kg.G.number_of_edges()} edges")
        print(f"  Embedding: ZeroCopyMatryoshka (scales: {self.embedding.sizes})")
        print(f"  Agents: {', '.join(self.agents.keys())}")
        print(f"  Communication: MessageBus + ConversationManager (budget-limited)")
        print()

        # Start message bus
        await self.message_bus.start()

        for m in range(months):
            snapshot = await self.run_month()
            month = snapshot["month"]

            bar_len = min(50, int(snapshot["mrr"] / 200))
            mrr_bar = "█" * bar_len + "░" * (50 - bar_len)

            print(f"Month {month:2d} │ MRR ${snapshot['mrr']:>9,.0f} │{mrr_bar}│ "
                  f"Cust: {snapshot['customers']:>4d} │ NPS: {snapshot['nps']:>4.0f} │ "
                  f"Churn: {snapshot['churn_rate']:.1%}")

            if month % 6 == 0:
                self._print_agent_report(month)

        # Stop message bus
        await self.message_bus.stop()

        self._print_final_report(months)
        return self.monthly_log

    def _print_agent_report(self, month: int):
        print(f"\n  ┌─── Agent Intelligence Report (Month {month}) ───┐")

        for name, agent in self.agents.items():
            wm_state = agent.working_memory.get_state_summary()
            stats = agent.stats
            learner_stats = agent.get_learning_stats()

            print(f"  │")
            print(f"  │ {agent.profile.name}")
            print(f"  │   Trinity WM: focus_dim={wm_state.get('embedding_dim', '?')}, "
                  f"activated={wm_state.get('activated_nodes', 0)}, "
                  f"tensioned={wm_state.get('tensioned_count', 0)}")
            print(f"  │   Queries: {stats.total_queries}, "
                  f"Avg conf: {stats.avg_confidence:.2f}")

            # MCTS reward history
            history = self.reward_history[name]
            ranked = sorted(
                [(s, np.mean(r) if r else 0.0, len(r)) for s, r in history.items()],
                key=lambda x: -x[1]
            )
            print(f"  │   MCTS Strategy Rewards:")
            for strat, mean_r, pulls in ranked:
                if pulls > 0:
                    bar = "▓" * int(mean_r * 15) + "░" * (15 - int(mean_r * 15))
                    print(f"  │     {strat:30s} {bar} {mean_r:.2f} ({pulls} evals)")

            if not learner_stats.get('learning_disabled'):
                print(f"  │   Patterns learned: {learner_stats.get('patterns_count', 0)}, "
                      f"snapshots: {learner_stats.get('snapshot_count', 0)}")

        # Inter-agent communication stats
        msgs_this_period = [m for m in self.messages_sent if m["month"] > month - 6]
        if msgs_this_period:
            print(f"  │")
            print(f"  │ Inter-Agent Messages (last 6 months): {len(msgs_this_period)}")
            for m in msgs_this_period[-5:]:
                print(f"  │   {m['from']:>12s} → {m['to']:<12s}: {m['content'][:50]}")

        conv_stats = self.conv_mgr.get_statistics()
        print(f"  │ Conversations: {conv_stats['total_conversations']} total, "
              f"{conv_stats['total_messages']} messages")
        print(f"  └{'─'*54}┘\n")

    def _print_final_report(self, months: int):
        c = self.company
        print(f"\n{'='*72}")
        print(f"  FINAL REPORT — {months} Months (v2 Real Multi-Agent Framework)")
        print(f"{'='*72}")
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

        print(f"\n  Shared Knowledge Graph (All Agents):")
        print(f"    Nodes: {self.shared_kg.G.number_of_nodes()}")
        print(f"    Edges: {self.shared_kg.G.number_of_edges()}")

        print(f"\n  Agent Working Memory States:")
        for name, agent in self.agents.items():
            wm = agent.working_memory.get_state_summary()
            stats = agent.stats
            print(f"    {name:12s}: queries={stats.total_queries:3d}, "
                  f"avg_conf={stats.avg_confidence:.2f}, "
                  f"activated={wm.get('activated_nodes', 0):3d}, "
                  f"tensioned={wm.get('tensioned_count', 0):2d}, "
                  f"patterns={agent.get_learning_stats().get('patterns_count', 0)}")

        print(f"\n  MCTS Strategy Convergence:")
        for name in self.agents:
            print(f"\n    {name.title()}:")
            history = self.reward_history[name]
            ranked = sorted(
                [(s, np.mean(r) if r else 0.0, len(r)) for s, r in history.items()],
                key=lambda x: -x[1]
            )
            for strat, mean_r, pulls in ranked:
                bar = "▓" * int(mean_r * 20) + "░" * (20 - int(mean_r * 20))
                print(f"      {strat:30s} {bar} {mean_r:.2f} ({pulls} evals)")

        print(f"\n  Inter-Agent Communication:")
        print(f"    Total messages: {len(self.messages_sent)}")
        conv_stats = self.conv_mgr.get_statistics()
        print(f"    Total conversations: {conv_stats['total_conversations']}")
        print(f"    Avg msgs/conversation: {conv_stats['avg_messages_per_conversation']:.1f}")

        print(f"\n  HoloLoom Components Used:")
        print(f"    ✓ AgentOrchestrator (4 agents with custom AgentProfile)")
        print(f"    ✓ Trinity Working Memory (semantic 244D + graph + warp)")
        print(f"    ✓ ZeroCopyMatryoshkaEmbeddings (scales: 96/192/384/768)")
        print(f"    ✓ MCTS Strategy Selection (30 simulations/decision)")
        print(f"    ✓ MessageBus + ConversationManager (budget-limited)")
        print(f"    ✓ Shared KG with SpringDynamics-compatible structure")
        print(f"    ✓ WorkingMemoryLearner (pattern persistence)")
        print(f"{'='*72}\n")


# =====================================================================
# Run
# =====================================================================

async def main():
    sim = HoloLoomBusinessSimulation(seed=42)
    results = await sim.run(months=24)

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "simulation_results_v2.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Full results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
