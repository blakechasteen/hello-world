#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collaborative Agents Integration
=================================

Integrate multi-agent communication with persistent background agents.

Created: 2025-01-20

Enables persistent agents to:
- Ask questions to each other
- Share insights
- Request help
- Collaborate on complex queries
"""

import asyncio
from typing import Optional, Dict, Any, List
import logging

from HoloLoom.agents.persistent_agent import (
    PersistentBackgroundAgent,
    AgentState
)
from HoloLoom.agents.multi_agent_communication import (
    MessageBus,
    ConversationManager,
    BudgetManager,
    SafetyGuardrails,
    Budget,
    Message,
    MessageType,
    Conversation
)
from HoloLoom.agents.policy_governance import (
    PolicyEngine,
    RoleBasedAccessControl,
    TopicGovernance,
    CommunicationRequest,
    PolicyDecision,
    Priority,
    AgentRole
)

logger = logging.getLogger(__name__)


class CollaborativeAgent(PersistentBackgroundAgent):
    """
    Persistent agent with multi-agent communication capabilities.

    Extends PersistentBackgroundAgent with ability to talk to other agents.
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        message_bus: MessageBus,
        conversation_manager: ConversationManager,
        loop_interval: float = 60.0,
        policy_engine: Optional[PolicyEngine] = None,
        **kwargs
    ):
        """
        Initialize collaborative agent.

        Args:
            agent_id: Unique agent ID
            agent_type: Agent type
            message_bus: Shared message bus
            conversation_manager: Shared conversation manager
            loop_interval: Learning loop interval
            policy_engine: Optional policy engine for governance
            **kwargs: Additional args for PersistentBackgroundAgent
        """
        super().__init__(
            agent_id=agent_id,
            agent_type=agent_type,
            loop_interval=loop_interval,
            **kwargs
        )

        self.message_bus = message_bus
        self.conversation_manager = conversation_manager
        self.policy_engine = policy_engine

        # Subscribe to messages
        self.message_bus.subscribe(agent_id, self._handle_message)

        # Pending conversations
        self.active_conversations: Dict[str, List[Message]] = {}

    async def _handle_message(self, message: Message) -> None:
        """
        Handle incoming message from another agent.

        Args:
            message: Received message
        """
        logger.info(f"📨 {self.agent_id} received {message.message_type.value} from {message.from_agent}")

        # Track conversation
        if message.conversation_id not in self.active_conversations:
            self.active_conversations[message.conversation_id] = []
        self.active_conversations[message.conversation_id].append(message)

        # Handle based on type
        if message.message_type == MessageType.QUESTION:
            await self._handle_question(message)
        elif message.message_type == MessageType.REQUEST_HELP:
            await self._handle_help_request(message)
        elif message.message_type == MessageType.INSIGHT:
            await self._handle_insight(message)
        elif message.message_type == MessageType.ANSWER:
            await self._handle_answer(message)

    async def _handle_question(self, message: Message) -> None:
        """Handle question from another agent."""
        # Use scratchpad to think about answer
        if not self.scratchpad:
            return

        # Think about the question
        thought = await self.scratchpad.think(
            f"{message.from_agent} asks: {message.content}",
            thought_type=ThoughtType.QUESTION
        )

        # Generate answer via dialogue
        tree = await self.scratchpad.dialogue_loop(
            initial_thought=thought,
            max_depth=2,
            mode="exploratory"
        )

        # Extract answer from dialogue
        answers = [t for t in tree.thoughts.values() if t.type == ThoughtType.ANSWER]
        if answers:
            answer_text = answers[0].text
        else:
            answer_text = "I don't have a clear answer yet."

        # Send answer
        await self.send_message(
            conversation_id=message.conversation_id,
            to_agent=message.from_agent,
            message_type=MessageType.ANSWER,
            content=answer_text,
            parent_message_id=message.id
        )

    async def _handle_help_request(self, message: Message) -> None:
        """Handle help request from another agent."""
        # Check if we can help based on agent type
        can_help = self._can_help_with(message.content)

        if can_help:
            # Offer help
            await self.send_message(
                conversation_id=message.conversation_id,
                to_agent=message.from_agent,
                message_type=MessageType.OFFER_HELP,
                content=f"I can help with that! Here's my approach: {can_help}",
                parent_message_id=message.id
            )
        else:
            # Acknowledge but can't help
            await self.send_message(
                conversation_id=message.conversation_id,
                to_agent=message.from_agent,
                message_type=MessageType.ACKNOWLEDGE,
                content="I see your request but this isn't my area of expertise.",
                parent_message_id=message.id
            )

    async def _handle_insight(self, message: Message) -> None:
        """Handle insight shared by another agent."""
        # Store insight
        self.state.insights.append(f"From {message.from_agent}: {message.content}")

        # Acknowledge
        await self.send_message(
            conversation_id=message.conversation_id,
            to_agent=message.from_agent,
            message_type=MessageType.ACKNOWLEDGE,
            content="Thanks for sharing that insight!",
            parent_message_id=message.id
        )

    async def _handle_answer(self, message: Message) -> None:
        """Handle answer from another agent."""
        # Store answer as insight
        self.state.insights.append(f"Answer from {message.from_agent}: {message.content}")

        logger.info(f"💡 {self.agent_id} received answer from {message.from_agent}")

    def _can_help_with(self, request: str) -> Optional[str]:
        """
        Check if agent can help with request based on agent type.

        Returns:
            Help description or None
        """
        request_lower = request.lower()

        if self.agent_type == "chain":
            if any(word in request_lower for word in ["sequence", "workflow", "steps", "order"]):
                return "I can help design sequential workflows with the Chain Orchestrator"

        elif self.agent_type == "recursive":
            if any(word in request_lower for word in ["improve", "refine", "optimize", "confidence"]):
                return "I can help with iterative refinement using Recursive Reasoner"

        elif self.agent_type == "workflow":
            if any(word in request_lower for word in ["complex", "parallel", "checkpoint", "visual"]):
                return "I can help design complex workflows with the Agentic Workflow System"

        elif self.agent_type == "scratchpad":
            if any(word in request_lower for word in ["think", "reflect", "meta", "reasoning"]):
                return "I can help with internal dialogue and meta-reasoning"

        return None

    async def ask_question(
        self,
        to_agent: str,
        question: str,
        topic: str,
        timeout: float = 30.0
    ) -> Optional[str]:
        """
        Ask question to another agent.

        Args:
            to_agent: Target agent ID
            question: Question text
            topic: Conversation topic
            timeout: Timeout for answer

        Returns:
            Answer text or None
        """
        # Start conversation
        conversation_id = await self.conversation_manager.start_conversation(
            topic=topic,
            initiator=self.agent_id,
            participants=[self.agent_id, to_agent]
        )

        if not conversation_id:
            logger.warning(f"⚠️  Could not start conversation with {to_agent}")
            return None

        # Send question
        message = await self.send_message(
            conversation_id=conversation_id,
            to_agent=to_agent,
            message_type=MessageType.QUESTION,
            content=question
        )

        if not message:
            return None

        # Wait for answer
        try:
            answer = await self._wait_for_answer(conversation_id, message.id, timeout)
            return answer
        finally:
            # End conversation
            await self.conversation_manager.end_conversation(conversation_id)

    async def request_help(
        self,
        from_agents: List[str],
        request: str,
        topic: str,
        timeout: float = 60.0
    ) -> List[tuple[str, str]]:
        """
        Request help from multiple agents.

        Args:
            from_agents: List of agent IDs to ask
            request: Help request
            topic: Conversation topic
            timeout: Timeout for responses

        Returns:
            List of (agent_id, response) tuples
        """
        # Start conversation
        conversation_id = await self.conversation_manager.start_conversation(
            topic=topic,
            initiator=self.agent_id,
            participants=[self.agent_id] + from_agents
        )

        if not conversation_id:
            return []

        # Send help requests
        for agent_id in from_agents:
            await self.send_message(
                conversation_id=conversation_id,
                to_agent=agent_id,
                message_type=MessageType.REQUEST_HELP,
                content=request
            )

        # Wait for responses
        await asyncio.sleep(timeout)

        # Collect responses
        responses = []
        if conversation_id in self.active_conversations:
            for msg in self.active_conversations[conversation_id]:
                if msg.message_type in [MessageType.OFFER_HELP, MessageType.ANSWER]:
                    responses.append((msg.from_agent, msg.content))

        # End conversation
        await self.conversation_manager.end_conversation(conversation_id)

        return responses

    async def share_insight(
        self,
        with_agents: List[str],
        insight: str,
        topic: str
    ) -> None:
        """
        Share insight with other agents.

        Args:
            with_agents: List of agent IDs
            insight: Insight to share
            topic: Topic
        """
        # Start conversation
        conversation_id = await self.conversation_manager.start_conversation(
            topic=topic,
            initiator=self.agent_id,
            participants=[self.agent_id] + with_agents
        )

        if not conversation_id:
            return

        # Share with each agent
        for agent_id in with_agents:
            await self.send_message(
                conversation_id=conversation_id,
                to_agent=agent_id,
                message_type=MessageType.INSIGHT,
                content=insight
            )

        # Give time for acknowledgments
        await asyncio.sleep(2.0)

        # End conversation
        await self.conversation_manager.end_conversation(conversation_id)

    def _check_policy(
        self,
        to_agent: str,
        message_type: MessageType,
        topic: str,
        content: str,
        priority: Priority = Priority.MEDIUM
    ) -> tuple[bool, str]:
        """
        Check policy before sending message.

        Args:
            to_agent: Target agent
            message_type: Message type
            topic: Conversation topic
            content: Message content
            priority: Message priority

        Returns:
            (allowed, reason)
        """
        # If no policy engine, allow
        if not self.policy_engine:
            return True, "No policy engine"

        # Create request
        request = CommunicationRequest(
            from_agent=self.agent_id,
            to_agent=to_agent,
            message_type=message_type.value,
            topic=topic,
            content=content,
            priority=priority
        )

        # Evaluate
        decision, reason = self.policy_engine.evaluate_communication_request(request)

        if decision == PolicyDecision.ALLOW:
            return True, reason
        elif decision == PolicyDecision.DENY:
            logger.warning(f"❌ Policy denied: {reason}")
            return False, reason
        elif decision == PolicyDecision.ESCALATE:
            logger.warning(f"⚠️  Policy escalated: {reason}")
            return False, f"Escalated: {reason}"
        else:
            return True, f"Policy decision: {decision.value}"

    async def send_message(
        self,
        conversation_id: str,
        to_agent: str,
        message_type: MessageType,
        content: str,
        parent_message_id: Optional[str] = None,
        topic: str = "",
        priority: Priority = Priority.MEDIUM
    ) -> Optional[Message]:
        """
        Send message in conversation.

        Args:
            conversation_id: Conversation ID
            to_agent: Recipient
            message_type: Message type
            content: Message content
            parent_message_id: Parent message ID
            topic: Topic (for policy check)
            priority: Priority (for policy check)

        Returns:
            Sent message or None
        """
        # Check policy first
        allowed, reason = self._check_policy(
            to_agent=to_agent,
            message_type=message_type,
            topic=topic,
            content=content,
            priority=priority
        )

        if not allowed:
            logger.warning(f"Policy blocked message: {reason}")
            return None

        return await self.conversation_manager.send_message(
            conversation_id=conversation_id,
            from_agent=self.agent_id,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            parent_message_id=parent_message_id
        )

    async def _wait_for_answer(
        self,
        conversation_id: str,
        question_message_id: str,
        timeout: float
    ) -> Optional[str]:
        """Wait for answer to question."""
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            # Check for answer
            if conversation_id in self.active_conversations:
                for msg in self.active_conversations[conversation_id]:
                    if (msg.message_type == MessageType.ANSWER and
                        msg.parent_message_id == question_message_id):
                        return msg.content

            await asyncio.sleep(0.5)

        return None


class CollaborativeAgentManager:
    """
    Manage multiple collaborative agents.

    Provides centralized communication infrastructure.
    """

    def __init__(
        self,
        loop_interval: float = 60.0,
        budget: Optional[Budget] = None,
        policy_engine: Optional[PolicyEngine] = None
    ):
        """
        Initialize collaborative agent manager.

        Args:
            loop_interval: Learning loop interval
            budget: Default budget for conversations
            policy_engine: Optional policy engine for governance
        """
        self.loop_interval = loop_interval
        self.policy_engine = policy_engine

        # Communication infrastructure
        self.message_bus = MessageBus()
        self.budget_manager = BudgetManager(
            default_budget=budget or Budget()
        )
        self.safety_guardrails = SafetyGuardrails()
        self.conversation_manager = ConversationManager(
            message_bus=self.message_bus,
            budget_manager=self.budget_manager,
            safety_guardrails=self.safety_guardrails
        )

        self.agents: Dict[str, CollaborativeAgent] = {}

    async def start(self) -> None:
        """Start message bus."""
        await self.message_bus.start()

    async def stop(self) -> None:
        """Stop all agents and message bus."""
        # Stop agents
        for agent in self.agents.values():
            await agent.stop()

        # Stop message bus
        await self.message_bus.stop()

        self.agents.clear()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def create_agent(
        self,
        agent_id: str,
        agent_type: str
    ) -> CollaborativeAgent:
        """
        Create and start collaborative agent.

        Args:
            agent_id: Unique agent ID
            agent_type: Agent type

        Returns:
            Started agent
        """
        if agent_id in self.agents:
            return self.agents[agent_id]

        agent = CollaborativeAgent(
            agent_id=agent_id,
            agent_type=agent_type,
            message_bus=self.message_bus,
            conversation_manager=self.conversation_manager,
            loop_interval=self.loop_interval,
            policy_engine=self.policy_engine
        )

        await agent.start()
        self.agents[agent_id] = agent

        return agent

    def get_agent(self, agent_id: str) -> Optional[CollaborativeAgent]:
        """Get agent by ID."""
        return self.agents.get(agent_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "agents": len(self.agents),
            "conversations": self.conversation_manager.get_statistics(),
            "message_bus_subscribers": len(self.message_bus.subscribers)
        }


# Import ThoughtType for type hints
from HoloLoom.scratchpad import ThoughtType
