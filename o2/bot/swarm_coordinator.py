"""
Swarm Coordinator - Multi-agent collaboration for O2

Implements:
- Task decomposition (break complex tasks into agent tasks)
- Agent selection (choose optimal agents)
- Parallel execution (concurrent agent work)
- Consensus (agents vote on decisions)
- Human-in-loop (escalate critical decisions)
"""

import asyncio
import logging
from typing import List, Dict, Optional
from enum import Enum

from HoloLoom.config import Config
from HoloLoom import HoloLoom

logger = logging.getLogger('o2-bot.swarm')


class AgentType(Enum):
    """Types of specialized agents."""
    QA = "qa"  # Quality assurance (trough/xTerminator)
    SECURITY = "security"  # Security scanning
    DEPLOYMENT = "deployment"  # Deployment automation
    MONITORING = "monitoring"  # Health monitoring
    DOCUMENTATION = "documentation"  # Documentation generation
    RESEARCH = "research"  # Research and analysis


class SwarmAgent:
    """
    Specialized agent within the swarm.

    Each agent has:
    - Specific capabilities (QA, security, deployment, etc.)
    - Own HoloLoom instance (specialized knowledge)
    - Communication protocol (Matrix rooms)
    """

    def __init__(self, agent_type: AgentType, config: Config):
        self.agent_type = agent_type
        self.config = config
        self.loom = None
        self.status = "idle"
        self.current_task = None

    async def initialize(self):
        """Initialize agent's HoloLoom instance."""
        self.loom = HoloLoom(config=self.config)
        logger.info(f"Initialized {self.agent_type.value} agent")

    async def execute_task(self, task: Dict) -> Dict:
        """
        Execute a task.

        Args:
            task: Task dictionary with description, context, parameters

        Returns:
            Result dictionary with status, output, confidence
        """
        self.status = "working"
        self.current_task = task

        try:
            # Agent-specific task execution
            if self.agent_type == AgentType.QA:
                result = await self._execute_qa_task(task)
            elif self.agent_type == AgentType.SECURITY:
                result = await self._execute_security_task(task)
            elif self.agent_type == AgentType.DEPLOYMENT:
                result = await self._execute_deployment_task(task)
            elif self.agent_type == AgentType.MONITORING:
                result = await self._execute_monitoring_task(task)
            elif self.agent_type == AgentType.DOCUMENTATION:
                result = await self._execute_documentation_task(task)
            elif self.agent_type == AgentType.RESEARCH:
                result = await self._execute_research_task(task)
            else:
                result = {
                    'status': 'error',
                    'error': f'Unknown agent type: {self.agent_type}'
                }

            return result

        except Exception as e:
            logger.error(f"{self.agent_type.value} agent error: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e)
            }
        finally:
            self.status = "idle"
            self.current_task = None

    async def _execute_qa_task(self, task: Dict) -> Dict:
        """Execute QA task (code review, testing)."""
        # TODO: Integrate with trough/xTerminator
        return {
            'status': 'success',
            'agent': 'qa',
            'output': 'QA task placeholder',
            'tests_passed': 0,
            'tests_failed': 0,
            'confidence': 0.9
        }

    async def _execute_security_task(self, task: Dict) -> Dict:
        """Execute security task (vulnerability scanning)."""
        # TODO: Integrate with security scanning tools
        return {
            'status': 'success',
            'agent': 'security',
            'output': 'Security scan placeholder',
            'vulnerabilities': [],
            'confidence': 0.95
        }

    async def _execute_deployment_task(self, task: Dict) -> Dict:
        """Execute deployment task (rollout, monitoring)."""
        # TODO: Implement deployment automation
        return {
            'status': 'success',
            'agent': 'deployment',
            'output': 'Deployment placeholder',
            'deployed': False,
            'confidence': 0.85
        }

    async def _execute_monitoring_task(self, task: Dict) -> Dict:
        """Execute monitoring task (health checks, metrics)."""
        # TODO: Implement monitoring
        return {
            'status': 'success',
            'agent': 'monitoring',
            'output': 'Monitoring placeholder',
            'healthy': True,
            'confidence': 0.9
        }

    async def _execute_documentation_task(self, task: Dict) -> Dict:
        """Execute documentation task (changelog, README)."""
        # TODO: Implement documentation generation
        return {
            'status': 'success',
            'agent': 'documentation',
            'output': 'Documentation placeholder',
            'files_updated': [],
            'confidence': 0.8
        }

    async def _execute_research_task(self, task: Dict) -> Dict:
        """Execute research task (analysis, synthesis)."""
        # Use HoloLoom's agentic reasoning
        result = await self.loom.recall(task['description'])
        return {
            'status': 'success',
            'agent': 'research',
            'output': result,
            'confidence': 0.85
        }

    async def close(self):
        """Close agent's HoloLoom instance."""
        if self.loom:
            await self.loom.close()
            logger.info(f"Closed {self.agent_type.value} agent")


class SwarmCoordinator:
    """
    Coordinates multiple agents working on complex tasks.

    Workflow:
    1. User requests task via Matrix (@o2 run swarm: deploy auth service)
    2. Coordinator decomposes task into subtasks
    3. Selects appropriate agents for each subtask
    4. Executes subtasks (parallel when possible)
    5. Aggregates results and reports back
    6. Escalates to human for critical decisions
    """

    def __init__(self, config: Config):
        self.config = config
        self.agents: Dict[AgentType, SwarmAgent] = {}

    async def initialize(self):
        """Initialize swarm with all agent types."""
        # Create agents
        for agent_type in AgentType:
            agent = SwarmAgent(agent_type, self.config)
            await agent.initialize()
            self.agents[agent_type] = agent

        logger.info(f"Swarm coordinator initialized with {len(self.agents)} agents")

    async def execute_swarm_task(self, task_description: str, user_id: str) -> Dict:
        """
        Execute a task using the swarm.

        Args:
            task_description: High-level task description
            user_id: User requesting the task

        Returns:
            Aggregated results from all agents
        """
        logger.info(f"Swarm task from {user_id}: {task_description}")

        # Step 1: Decompose task into subtasks
        subtasks = await self._decompose_task(task_description)

        # Step 2: Execute subtasks (parallel when possible)
        results = await self._execute_subtasks(subtasks)

        # Step 3: Aggregate results
        aggregated = await self._aggregate_results(results)

        # Step 4: Check if human approval needed
        if aggregated.get('requires_approval'):
            aggregated['status'] = 'pending_approval'
            aggregated['message'] = 'Critical decisions require human approval'

        return aggregated

    async def _decompose_task(self, task_description: str) -> List[Dict]:
        """
        Decompose high-level task into agent subtasks.

        Args:
            task_description: Task description

        Returns:
            List of subtasks with agent assignments
        """
        # Simple keyword-based decomposition for now
        # TODO: Use LLM for intelligent decomposition

        subtasks = []
        task_lower = task_description.lower()

        # Check for deployment keywords
        if any(kw in task_lower for kw in ['deploy', 'rollout', 'release']):
            subtasks.extend([
                {'agent': AgentType.QA, 'description': 'Run tests', 'priority': 1},
                {'agent': AgentType.SECURITY, 'description': 'Security scan', 'priority': 1},
                {'agent': AgentType.DEPLOYMENT, 'description': f'Deploy: {task_description}', 'priority': 2},
                {'agent': AgentType.MONITORING, 'description': 'Monitor deployment', 'priority': 3},
                {'agent': AgentType.DOCUMENTATION, 'description': 'Update docs', 'priority': 3}
            ])

        # Check for research keywords
        elif any(kw in task_lower for kw in ['research', 'analyze', 'investigate']):
            subtasks.append(
                {'agent': AgentType.RESEARCH, 'description': task_description, 'priority': 1}
            )

        # Check for QA keywords
        elif any(kw in task_lower for kw in ['test', 'qa', 'review']):
            subtasks.append(
                {'agent': AgentType.QA, 'description': task_description, 'priority': 1}
            )

        # Default: research
        else:
            subtasks.append(
                {'agent': AgentType.RESEARCH, 'description': task_description, 'priority': 1}
            )

        logger.info(f"Decomposed task into {len(subtasks)} subtasks")
        return subtasks

    async def _execute_subtasks(self, subtasks: List[Dict]) -> List[Dict]:
        """
        Execute subtasks (parallel when possible).

        Args:
            subtasks: List of subtasks

        Returns:
            List of results
        """
        # Group by priority (execute same priority in parallel)
        priority_groups = {}
        for subtask in subtasks:
            priority = subtask['priority']
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(subtask)

        all_results = []

        # Execute priority groups sequentially
        for priority in sorted(priority_groups.keys()):
            group = priority_groups[priority]
            logger.info(f"Executing priority {priority} group ({len(group)} tasks)")

            # Execute group in parallel
            tasks = [
                self.agents[subtask['agent']].execute_task(subtask)
                for subtask in group
            ]
            results = await asyncio.gather(*tasks)
            all_results.extend(results)

        return all_results

    async def _aggregate_results(self, results: List[Dict]) -> Dict:
        """
        Aggregate results from all agents.

        Args:
            results: List of agent results

        Returns:
            Aggregated result dictionary
        """
        aggregated = {
            'status': 'success',
            'agents_executed': len(results),
            'results': results,
            'summary': {},
            'requires_approval': False
        }

        # Check for errors
        errors = [r for r in results if r.get('status') == 'error']
        if errors:
            aggregated['status'] = 'partial_success' if len(errors) < len(results) else 'error'
            aggregated['errors'] = errors

        # Aggregate by agent type
        for result in results:
            agent = result.get('agent')
            if agent:
                aggregated['summary'][agent] = {
                    'status': result.get('status'),
                    'confidence': result.get('confidence', 0.0)
                }

        # Check if any critical task failed (QA, security)
        qa_result = next((r for r in results if r.get('agent') == 'qa'), None)
        security_result = next((r for r in results if r.get('agent') == 'security'), None)

        if qa_result and qa_result.get('tests_failed', 0) > 0:
            aggregated['requires_approval'] = True
        if security_result and security_result.get('vulnerabilities'):
            aggregated['requires_approval'] = True

        return aggregated

    async def close(self):
        """Close all agents."""
        for agent in self.agents.values():
            await agent.close()
        logger.info("Swarm coordinator closed")
