#!/usr/bin/env python3
"""
Multi-Agent Collaboration Framework

Coordinate multiple specialized bots for complex tasks:
- Specialist agents (incident, code review, deployment, security)
- Agent orchestration and routing
- Collaborative workflows
- Knowledge sharing between agents
- Consensus building

Usage:
    from HoloLoom.chatops.multi_agent import MultiAgentSystem, AgentRole

    system = MultiAgentSystem()

    # Register specialized agents
    system.register_agent("incident_specialist", role=AgentRole.INCIDENT_RESPONSE)
    system.register_agent("code_reviewer", role=AgentRole.CODE_REVIEW)

    # Route query to best agent
    response = await system.route(query)

    # Collaborative task
    result = await system.collaborate(
        task="investigate_security_incident",
        agents=["incident_specialist", "security_analyst"]
    )
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

try:
    from promptly_integration import PromptlyEnhancedBot, UltrapromptConfig
    PROMPTLY_AVAILABLE = True
except ImportError:
    PROMPTLY_AVAILABLE = False


class AgentRole(Enum):
    """Specialized agent roles"""
    INCIDENT_RESPONSE = "incident_response"
    CODE_REVIEW = "code_review"
    DEPLOYMENT = "deployment"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    GENERAL = "general"


@dataclass
class AgentCapability:
    """Agent capability definition"""
    role: AgentRole
    expertise_areas: List[str]
    supported_tasks: List[str]
    quality_threshold: float = 0.75
    max_concurrent_tasks: int = 5

    # Performance metrics
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    total_tasks: int = 0


@dataclass
class Agent:
    """Individual agent"""
    agent_id: str
    name: str
    capability: AgentCapability
    bot: Any  # PromptlyEnhancedBot instance

    # Configuration
    ultraprompt_config: Dict[str, Any] = field(default_factory=dict)

    # State
    active_tasks: List[str] = field(default_factory=list)
    available: bool = True

    # Statistics
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_quality_score: float = 0.0


@dataclass
class CollaborationTask:
    """Multi-agent collaboration task"""
    task_id: str
    description: str
    task_type: str
    required_agents: List[str]

    # Execution
    status: str = "pending"  # pending, in_progress, completed, failed
    assigned_agents: List[str] = field(default_factory=list)
    subtasks: List[Dict[str, Any]] = field(default_factory=list)

    # Results
    agent_responses: Dict[str, Any] = field(default_factory=dict)
    consensus: Optional[str] = None
    final_response: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class MultiAgentSystem:
    """
    System for coordinating multiple specialized agents.

    Features:
    - Agent registration and management
    - Intelligent routing
    - Collaborative workflows
    - Consensus building
    - Knowledge sharing
    """

    def __init__(self, storage_path: Optional[Path] = None):
        if not PROMPTLY_AVAILABLE:
            raise ImportError("Promptly integration required")

        self.storage_path = storage_path or Path("./chatops_data/multi_agent")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Agents
        self.agents: Dict[str, Agent] = {}

        # Active collaborations
        self.collaborations: Dict[str, CollaborationTask] = {}

        # Routing rules
        self.routing_rules: Dict[str, str] = {}  # pattern -> agent_id

        # Knowledge sharing
        self.shared_knowledge: Dict[str, Any] = {}

        # Statistics
        self.stats = {
            "total_agents": 0,
            "total_tasks": 0,
            "collaborative_tasks": 0,
            "successful_routings": 0
        }

        # Initialize default agents
        self._initialize_default_agents()

        logging.info("MultiAgentSystem initialized")

    def _initialize_default_agents(self):
        """Create default specialized agents"""

        # Incident Response Specialist
        self.register_agent(
            agent_id="incident_specialist",
            name="Incident Response Specialist",
            role=AgentRole.INCIDENT_RESPONSE,
            expertise=["troubleshooting", "root cause analysis", "remediation"],
            ultraprompt_config={
                "sections": ["PLAN", "ANSWER", "VERIFY"],
                "use_verification": True,
                "temperature": 0.6  # More deterministic for incidents
            }
        )

        # Code Review Expert
        self.register_agent(
            agent_id="code_reviewer",
            name="Code Review Expert",
            role=AgentRole.CODE_REVIEW,
            expertise=["security", "best practices", "performance"],
            ultraprompt_config={
                "sections": ["ANSWER", "TL;DR"],
                "use_planning": True,
                "temperature": 0.7
            }
        )

        # Security Analyst
        self.register_agent(
            agent_id="security_analyst",
            name="Security Analyst",
            role=AgentRole.SECURITY,
            expertise=["vulnerabilities", "compliance", "threat analysis"],
            ultraprompt_config={
                "sections": ["PLAN", "ANSWER", "VERIFY"],
                "use_verification": True,
                "temperature": 0.5  # Very deterministic for security
            }
        )

        # Deployment Coordinator
        self.register_agent(
            agent_id="deployment_coordinator",
            name="Deployment Coordinator",
            role=AgentRole.DEPLOYMENT,
            expertise=["ci/cd", "rollback", "monitoring"],
            ultraprompt_config={
                "sections": ["PLAN", "ANSWER", "NEXT"],
                "use_planning": True,
                "temperature": 0.6
            }
        )

        # General Assistant
        self.register_agent(
            agent_id="general_assistant",
            name="General Assistant",
            role=AgentRole.GENERAL,
            expertise=["general queries", "documentation", "help"],
            ultraprompt_config={
                "temperature": 0.7
            }
        )

    def register_agent(
        self,
        agent_id: str,
        name: str,
        role: AgentRole,
        expertise: List[str],
        supported_tasks: Optional[List[str]] = None,
        ultraprompt_config: Optional[Dict] = None
    ):
        """
        Register a new specialized agent.

        Args:
            agent_id: Unique agent identifier
            name: Human-readable name
            role: Agent role
            expertise: Areas of expertise
            supported_tasks: Task types this agent can handle
            ultraprompt_config: Custom ultraprompt configuration
        """

        # Create capability
        capability = AgentCapability(
            role=role,
            expertise_areas=expertise,
            supported_tasks=supported_tasks or []
        )

        # Create bot with specialized config
        config = UltrapromptConfig(**(ultraprompt_config or {}))
        bot = PromptlyEnhancedBot(ultraprompt_config=config)

        # Create agent
        agent = Agent(
            agent_id=agent_id,
            name=name,
            capability=capability,
            bot=bot,
            ultraprompt_config=ultraprompt_config or {}
        )

        self.agents[agent_id] = agent
        self.stats["total_agents"] += 1

        logging.info(f"Registered agent: {agent_id} ({role.value})")

    async def route(
        self,
        query: str,
        context: Optional[Dict] = None,
        preferred_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Route query to best agent.

        Args:
            query: User query
            context: Additional context
            preferred_agent: Optionally specify agent

        Returns:
            Response from selected agent
        """

        # Select agent
        if preferred_agent and preferred_agent in self.agents:
            agent_id = preferred_agent
        else:
            agent_id = self._select_best_agent(query, context)

        agent = self.agents[agent_id]

        # Check availability
        if not agent.available or len(agent.active_tasks) >= agent.capability.max_concurrent_tasks:
            # Fallback to general assistant
            agent_id = "general_assistant"
            agent = self.agents[agent_id]

        # Generate task ID
        task_id = f"task_{agent_id}_{datetime.now().strftime('%H%M%S')}"

        # Add to active tasks
        agent.active_tasks.append(task_id)

        try:
            # Process with agent's bot
            result = await agent.bot.process_with_ultraprompt(query, context)

            # Update agent statistics
            agent.completed_tasks += 1
            self.stats["successful_routings"] += 1

            # Update quality tracking
            quality = result.get("quality_score", 0.0)
            agent.avg_quality_score = (
                (agent.avg_quality_score * (agent.completed_tasks - 1) + quality) /
                agent.completed_tasks
            )

            # Add agent metadata
            result["routed_to"] = agent_id
            result["agent_name"] = agent.name
            result["agent_role"] = agent.capability.role.value

            return result

        finally:
            # Remove from active tasks
            agent.active_tasks.remove(task_id)

    def _select_best_agent(
        self,
        query: str,
        context: Optional[Dict]
    ) -> str:
        """Select best agent for query"""

        query_lower = query.lower()

        # Rule-based routing
        if any(word in query_lower for word in ["incident", "error", "down", "failing"]):
            return "incident_specialist"
        elif any(word in query_lower for word in ["review", "pr", "code"]):
            return "code_reviewer"
        elif any(word in query_lower for word in ["security", "vulnerability", "exploit"]):
            return "security_analyst"
        elif any(word in query_lower for word in ["deploy", "release", "rollout"]):
            return "deployment_coordinator"
        else:
            return "general_assistant"

    async def collaborate(
        self,
        task_description: str,
        task_type: str,
        required_roles: Optional[List[AgentRole]] = None,
        agent_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute collaborative task with multiple agents.

        Args:
            task_description: Task description
            task_type: Type of task
            required_roles: Required agent roles
            agent_ids: Specific agent IDs (or auto-select by role)

        Returns:
            Collaboration results with consensus
        """

        # Select agents
        if agent_ids:
            agents = [self.agents[aid] for aid in agent_ids if aid in self.agents]
        elif required_roles:
            agents = [
                agent for agent in self.agents.values()
                if agent.capability.role in required_roles
            ]
        else:
            # Auto-select based on task type
            agents = self._auto_select_agents(task_type)

        if not agents:
            raise ValueError("No suitable agents found")

        # Create collaboration task
        task_id = f"collab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        collaboration = CollaborationTask(
            task_id=task_id,
            description=task_description,
            task_type=task_type,
            required_agents=[a.agent_id for a in agents],
            assigned_agents=[a.agent_id for a in agents],
            status="in_progress"
        )

        self.collaborations[task_id] = collaboration
        self.stats["collaborative_tasks"] += 1

        try:
            # Get responses from all agents
            agent_responses = await self._gather_agent_responses(
                agents,
                task_description,
                task_type
            )

            collaboration.agent_responses = agent_responses

            # Build consensus
            consensus = await self._build_consensus(agent_responses, task_type)
            collaboration.consensus = consensus

            # Generate final response
            final_response = await self._generate_final_response(
                task_description,
                agent_responses,
                consensus
            )

            collaboration.final_response = final_response
            collaboration.status = "completed"
            collaboration.completed_at = datetime.now()

            return {
                "task_id": task_id,
                "agents_involved": [a.name for a in agents],
                "individual_responses": agent_responses,
                "consensus": consensus,
                "final_response": final_response,
                "success": True
            }

        except Exception as e:
            collaboration.status = "failed"
            logging.error(f"Collaboration failed: {e}")

            return {
                "task_id": task_id,
                "error": str(e),
                "success": False
            }

    async def _gather_agent_responses(
        self,
        agents: List[Agent],
        task: str,
        task_type: str
    ) -> Dict[str, Any]:
        """Gather responses from multiple agents"""

        # Execute agents in parallel
        tasks = [
            agent.bot.process_with_ultraprompt(
                f"[{agent.capability.role.value.upper()}] {task}",
                context={"task_type": task_type}
            )
            for agent in agents
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Organize responses
        agent_responses = {}
        for agent, response in zip(agents, responses):
            if isinstance(response, Exception):
                agent_responses[agent.agent_id] = {
                    "error": str(response),
                    "success": False
                }
            else:
                agent_responses[agent.agent_id] = {
                    "response": response.get("answer", ""),
                    "quality_score": response.get("quality_score", 0.0),
                    "agent_name": agent.name,
                    "role": agent.capability.role.value,
                    "success": True
                }

        return agent_responses

    async def _build_consensus(
        self,
        agent_responses: Dict[str, Any],
        task_type: str
    ) -> str:
        """Build consensus from agent responses"""

        # Collect successful responses
        successful = [
            r for r in agent_responses.values()
            if r.get("success") and r.get("quality_score", 0) > 0.7
        ]

        if not successful:
            return "No consensus - insufficient quality responses"

        # Extract key points from each response
        all_points = []
        for response in successful:
            text = response.get("response", "")
            # Simple extraction - would use NLP in production
            points = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
            all_points.extend(points)

        # Find common themes (simplified)
        consensus_points = []

        # Priority based on agent role
        role_priority = {
            "incident_response": 1.0,
            "security": 0.9,
            "code_review": 0.8,
            "deployment": 0.7,
            "general": 0.5
        }

        # Weight by quality and role
        for response in successful:
            role = response.get("role", "general")
            quality = response.get("quality_score", 0.7)
            weight = role_priority.get(role, 0.5) * quality

            if weight > 0.6:
                text = response.get("response", "")
                consensus_points.append(f"[{role.upper()}] {text[:200]}...")

        return "\n\n".join(consensus_points[:3])  # Top 3 weighted responses

    async def _generate_final_response(
        self,
        task: str,
        agent_responses: Dict[str, Any],
        consensus: str
    ) -> str:
        """Generate final synthesized response"""

        # Use general assistant to synthesize
        general = self.agents["general_assistant"]

        synthesis_prompt = f"""
Task: {task}

Agent Responses:
{self._format_agent_responses(agent_responses)}

Consensus:
{consensus}

Synthesize a final response that:
1. Incorporates insights from all agents
2. Highlights areas of agreement
3. Notes any disagreements
4. Provides actionable recommendations
"""

        result = await general.bot.process_with_ultraprompt(synthesis_prompt)

        return result.get("answer", consensus)

    def _format_agent_responses(self, responses: Dict[str, Any]) -> str:
        """Format agent responses for synthesis"""

        formatted = []
        for agent_id, response in responses.items():
            if response.get("success"):
                agent_name = response.get("agent_name", agent_id)
                role = response.get("role", "unknown")
                text = response.get("response", "")[:300]

                formatted.append(f"**{agent_name} ({role}):**\n{text}...\n")

        return "\n".join(formatted)

    def _auto_select_agents(self, task_type: str) -> List[Agent]:
        """Auto-select agents based on task type"""

        if task_type == "security_incident":
            return [
                self.agents["incident_specialist"],
                self.agents["security_analyst"]
            ]
        elif task_type == "code_review_security":
            return [
                self.agents["code_reviewer"],
                self.agents["security_analyst"]
            ]
        elif task_type == "deployment_incident":
            return [
                self.agents["deployment_coordinator"],
                self.agents["incident_specialist"]
            ]
        else:
            return [self.agents["general_assistant"]]

    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent status"""

        agent = self.agents.get(agent_id)
        if not agent:
            return None

        return {
            "agent_id": agent_id,
            "name": agent.name,
            "role": agent.capability.role.value,
            "available": agent.available,
            "active_tasks": len(agent.active_tasks),
            "completed_tasks": agent.completed_tasks,
            "failed_tasks": agent.failed_tasks,
            "avg_quality_score": agent.avg_quality_score,
            "expertise": agent.capability.expertise_areas
        }

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents"""
        return [self.get_agent_status(aid) for aid in self.agents.keys()]

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = self.stats.copy()

        # Add agent breakdown
        stats["agents_by_role"] = {}
        for agent in self.agents.values():
            role = agent.capability.role.value
            stats["agents_by_role"][role] = stats["agents_by_role"].get(role, 0) + 1

        return stats


# Demo
async def demo_multi_agent():
    """Demonstrate multi-agent system"""

    print("👥 Multi-Agent Collaboration System Demo\n")

    system = MultiAgentSystem()

    # List agents
    print("1️⃣  Available Agents:")
    for agent_status in system.list_agents():
        print(f"   - {agent_status['name']} ({agent_status['role']})")
        print(f"     Expertise: {', '.join(agent_status['expertise'])}")
    print()

    # Route simple query
    print("2️⃣  Routing Query:")
    query = "The API is returning 500 errors"
    result = await system.route(query)
    print(f"   Query: {query}")
    print(f"   Routed to: {result['agent_name']} ({result['agent_role']})")
    print(f"   Quality: {result.get('quality_score', 0.0):.2f}")
    print()

    # Collaborative task
    print("3️⃣  Collaborative Task:")
    task = "Investigate potential security incident with database breach"
    collab_result = await system.collaborate(
        task_description=task,
        task_type="security_incident"
    )

    print(f"   Task: {task}")
    print(f"   Agents: {', '.join(collab_result['agents_involved'])}")
    print(f"   Success: {collab_result['success']}")
    if collab_result['success']:
        print(f"   Consensus:\n   {collab_result['consensus'][:200]}...")
    print()

    # Statistics
    print("4️⃣  System Statistics:")
    stats = system.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    asyncio.run(demo_multi_agent())
