#!/usr/bin/env python3
"""
Promptly Integration for ChatOps

Integrates Promptly's advanced prompt engineering framework with Matrix.org chatops:
- Ultraprompt system for high-quality responses
- LLM judge for response quality control
- Loop composition for complex workflows
- Skill system for reusable chatops functions
- A/B testing for prompt optimization

Usage:
    from HoloLoom.chatops.promptly_integration import PromptlyEnhancedBot

    bot = PromptlyEnhancedBot(bot_config)

    # Ultraprompt responses
    response = await bot.process_with_ultraprompt(message, context)

    # Judge response quality
    quality = await bot.evaluate_response(query, response)

    # Execute workflow loops
    result = await bot.execute_workflow("research_loop", {"topic": "AI"})

    # A/B test prompts
    winner = await bot.ab_test_prompts(prompt_a, prompt_b, test_cases)
"""

import asyncio
import os
import sys
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import yaml
import logging

# Add Promptly to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Promptly" / "promptly"))

try:
    from promptly import Promptly
    from tools.llm_judge_enhanced import EnhancedLLMJudge
    from tools.ab_testing import ABTester
    from loop_composition import LoopComposer
    from integrations.hololoom_bridge import HoloLoomBridge
    PROMPTLY_AVAILABLE = True
except ImportError:
    PROMPTLY_AVAILABLE = False
    logging.warning("Promptly not available - install from Promptly/ directory")


@dataclass
class UltrapromptConfig:
    """Configuration for ultraprompt system"""
    enabled: bool = True
    version: int = 2
    use_sources: bool = True
    use_verification: bool = True
    use_planning: bool = True
    max_tokens: int = 4096
    temperature: float = 0.7

    # Modular sections
    sections: List[str] = field(default_factory=lambda: [
        "PLAN",      # Planning section
        "ANSWER",    # Main answer
        "VERIFY",    # Verification
        "TL;DR"      # Summary
    ])

    # Advanced techniques
    use_prompt_chaining: bool = True
    use_latent_scaffolding: bool = True
    use_instruction_tagging: bool = True


@dataclass
class JudgeConfig:
    """Configuration for LLM judge"""
    enabled: bool = True
    criteria: List[str] = field(default_factory=lambda: [
        "accuracy",
        "clarity",
        "completeness",
        "relevance",
        "helpfulness"
    ])
    min_score_threshold: float = 0.7
    auto_retry_on_low_score: bool = True
    max_retries: int = 2


@dataclass
class WorkflowConfig:
    """Configuration for workflow loops"""
    enabled: bool = True
    max_loop_iterations: int = 10
    timeout_seconds: int = 300
    cache_results: bool = True

    # Built-in workflows
    workflows: Dict[str, str] = field(default_factory=dict)


@dataclass
class ABTestConfig:
    """Configuration for A/B testing"""
    enabled: bool = True
    min_test_cases: int = 3
    statistical_significance: float = 0.95
    auto_promote_winner: bool = True


class PromptlyEnhancedBot:
    """
    Matrix chatbot enhanced with Promptly framework features.

    Combines:
    - Ultraprompt for high-quality responses
    - LLM judge for quality control
    - Loop composition for workflows
    - A/B testing for optimization
    """

    def __init__(
        self,
        ultraprompt_config: Optional[UltrapromptConfig] = None,
        judge_config: Optional[JudgeConfig] = None,
        workflow_config: Optional[WorkflowConfig] = None,
        ab_test_config: Optional[ABTestConfig] = None
    ):
        if not PROMPTLY_AVAILABLE:
            raise ImportError("Promptly framework not available")

        self.ultraprompt_config = ultraprompt_config or UltrapromptConfig()
        self.judge_config = judge_config or JudgeConfig()
        self.workflow_config = workflow_config or WorkflowConfig()
        self.ab_test_config = ab_test_config or ABTestConfig()

        # Initialize Promptly components
        self.promptly = Promptly()
        self.judge = EnhancedLLMJudge() if judge_config.enabled else None
        self.ab_tester = ABTester() if ab_test_config.enabled else None
        self.loop_composer = LoopComposer() if workflow_config.enabled else None
        self.hololoom_bridge = HoloLoomBridge()

        # Load ultraprompt template
        self.ultraprompt_template = self._load_ultraprompt_template()

        # Workflow definitions
        self.workflows: Dict[str, Any] = {}
        self._register_default_workflows()

        # A/B test experiments
        self.active_experiments: Dict[str, Dict] = {}

        # Quality statistics
        self.quality_stats = {
            "total_responses": 0,
            "high_quality": 0,
            "low_quality": 0,
            "avg_score": 0.0,
            "retry_count": 0
        }

        logging.info("PromptlyEnhancedBot initialized")

    def _load_ultraprompt_template(self) -> Optional[Dict]:
        """Load ultraprompt template from YAML"""
        try:
            template_path = Path(__file__).parent.parent.parent / "Promptly" / "promptly" / ".promptly" / "prompts" / "ultraprompt-advanced.yaml"

            if template_path.exists():
                with open(template_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                logging.warning(f"Ultraprompt template not found at {template_path}")
                return None
        except Exception as e:
            logging.error(f"Failed to load ultraprompt template: {e}")
            return None

    def _register_default_workflows(self):
        """Register built-in workflow loops"""

        # Research workflow
        research_loop = """
LOOP research
  INPUT: topic
  STEPS:
    - search_knowledge = hololoom.search(topic, limit=5)
    - analyze_results = analyze(search_knowledge)
    - find_gaps = identify_gaps(analyze_results)
    - deep_dive = research_deeper(find_gaps)
    - synthesize = combine_findings(search_knowledge, deep_dive)
  OUTPUT: synthesize
END
"""

        # Incident investigation workflow
        incident_loop = """
LOOP investigate_incident
  INPUT: incident_id
  STEPS:
    - incident_data = get_incident(incident_id)
    - related_incidents = search_similar(incident_data)
    - root_cause = analyze_root_cause(incident_data, related_incidents)
    - remediation = suggest_remediation(root_cause)
    - verify_fix = validate_remediation(remediation)
  OUTPUT: {root_cause, remediation, verify_fix}
END
"""

        # Code review workflow
        code_review_loop = """
LOOP code_review
  INPUT: pr_number
  STEPS:
    - pr_data = fetch_pr(pr_number)
    - code_analysis = analyze_code(pr_data)
    - security_scan = check_security(code_analysis)
    - best_practices = check_best_practices(code_analysis)
    - suggestions = generate_suggestions(code_analysis, security_scan, best_practices)
  OUTPUT: suggestions
END
"""

        # Onboarding workflow
        onboarding_loop = """
LOOP onboard_user
  INPUT: user_id, role
  STEPS:
    - welcome = send_welcome_message(user_id)
    - docs = gather_relevant_docs(role)
    - team_intro = introduce_team(role)
    - setup_tasks = create_setup_checklist(role)
    - schedule_checkin = schedule_followup(user_id, days=7)
  OUTPUT: {welcome, docs, team_intro, setup_tasks}
END
"""

        if self.loop_composer:
            try:
                self.workflows["research"] = self.loop_composer.parse(research_loop)
                self.workflows["investigate_incident"] = self.loop_composer.parse(incident_loop)
                self.workflows["code_review"] = self.loop_composer.parse(code_review_loop)
                self.workflows["onboard_user"] = self.loop_composer.parse(onboarding_loop)

                logging.info(f"Registered {len(self.workflows)} default workflows")
            except Exception as e:
                logging.error(f"Failed to register workflows: {e}")

    async def process_with_ultraprompt(
        self,
        query: str,
        context: Optional[Dict] = None,
        use_hololoom: bool = True
    ) -> Dict[str, Any]:
        """
        Process a query using the ultraprompt framework.

        Args:
            query: User query
            context: Additional context
            use_hololoom: Use HoloLoom memory for context

        Returns:
            Dict with response, plan, verification, and metadata
        """
        if not self.ultraprompt_config.enabled:
            return {"response": await self._simple_response(query), "ultraprompt": False}

        # Build ultraprompt
        ultraprompt = self._build_ultraprompt(query, context, use_hololoom)

        # Execute with Promptly
        start_time = datetime.now()

        try:
            if self.ultraprompt_config.use_prompt_chaining:
                # Multi-stage execution
                result = await self._chained_execution(ultraprompt, query, context)
            else:
                # Single-shot execution
                result = await self.promptly.execute(ultraprompt)

            # Parse structured response
            parsed = self._parse_ultraprompt_response(result)

            # Quality check with judge
            if self.judge_config.enabled and self.judge:
                quality_score = await self.evaluate_response(query, parsed["answer"])
                parsed["quality_score"] = quality_score

                # Auto-retry on low quality
                if (quality_score < self.judge_config.min_score_threshold and
                    self.judge_config.auto_retry_on_low_score):

                    logging.warning(f"Low quality response (score={quality_score:.2f}), retrying...")
                    self.quality_stats["retry_count"] += 1

                    for retry in range(self.judge_config.max_retries):
                        result = await self.promptly.execute(ultraprompt)
                        parsed = self._parse_ultraprompt_response(result)
                        quality_score = await self.evaluate_response(query, parsed["answer"])

                        if quality_score >= self.judge_config.min_score_threshold:
                            parsed["quality_score"] = quality_score
                            parsed["retry_count"] = retry + 1
                            break

            # Update stats
            self._update_quality_stats(parsed.get("quality_score", 0.0))

            execution_time = (datetime.now() - start_time).total_seconds()
            parsed["execution_time"] = execution_time
            parsed["ultraprompt"] = True

            return parsed

        except Exception as e:
            logging.error(f"Ultraprompt execution failed: {e}")
            return {
                "response": f"Error processing query: {str(e)}",
                "ultraprompt": False,
                "error": str(e)
            }

    def _build_ultraprompt(
        self,
        query: str,
        context: Optional[Dict],
        use_hololoom: bool
    ) -> str:
        """Build ultraprompt with all directives"""

        sections = []

        # Role and directives (from template)
        if self.ultraprompt_template:
            sections.append(self.ultraprompt_template.get("content", ""))

        # HoloLoom context
        if use_hololoom:
            try:
                memory_context = self.hololoom_bridge.get_relevant_context(query, limit=3)
                if memory_context:
                    sections.append(f"\n## Relevant Context from Memory:\n{memory_context}\n")
            except:
                pass

        # Additional context
        if context:
            context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
            sections.append(f"\n## Additional Context:\n{context_str}\n")

        # User query with instruction tagging
        if self.ultraprompt_config.use_instruction_tagging:
            sections.append(f"\n<user_query>\n{query}\n</user_query>\n")
        else:
            sections.append(f"\n## User Query:\n{query}\n")

        # Request structured output
        if "PLAN" in self.ultraprompt_config.sections:
            sections.append("\n## Required Output Format:\n")
            sections.append("Please provide your response in the following sections:\n")
            for section in self.ultraprompt_config.sections:
                sections.append(f"- **{section}**")

        return "\n".join(sections)

    async def _chained_execution(
        self,
        ultraprompt: str,
        query: str,
        context: Optional[Dict]
    ) -> str:
        """Execute ultraprompt using prompt chaining"""

        # Stage 1: Planning
        plan_prompt = f"{ultraprompt}\n\nFirst, create a PLAN for how to answer this query. List the key steps."
        plan = await self.promptly.execute(plan_prompt)

        # Stage 2: Answer with plan
        answer_prompt = f"{ultraprompt}\n\n## PLAN:\n{plan}\n\nNow provide the detailed ANSWER following this plan."
        answer = await self.promptly.execute(answer_prompt)

        # Stage 3: Verification (if enabled)
        if self.ultraprompt_config.use_verification:
            verify_prompt = f"Query: {query}\n\nAnswer: {answer}\n\nVERIFY: Check if this answer fully addresses the query. List any issues or confirm it's complete."
            verification = await self.promptly.execute(verify_prompt)
        else:
            verification = "Verification skipped"

        # Stage 4: TL;DR
        tldr_prompt = f"Answer: {answer}\n\nProvide a TL;DR (1-2 sentence summary):"
        tldr = await self.promptly.execute(tldr_prompt)

        # Combine stages
        return f"## PLAN\n{plan}\n\n## ANSWER\n{answer}\n\n## VERIFY\n{verification}\n\n## TL;DR\n{tldr}"

    def _parse_ultraprompt_response(self, response: str) -> Dict[str, str]:
        """Parse structured ultraprompt response into sections"""

        sections = {
            "plan": "",
            "answer": "",
            "verify": "",
            "tldr": "",
            "assumptions": "",
            "next": ""
        }

        current_section = None
        current_content = []

        for line in response.split("\n"):
            # Check for section headers
            if line.strip().startswith("## PLAN"):
                current_section = "plan"
                current_content = []
            elif line.strip().startswith("## ANSWER"):
                current_section = "answer"
                current_content = []
            elif line.strip().startswith("## VERIFY"):
                current_section = "verify"
                current_content = []
            elif line.strip().startswith("## TL;DR"):
                current_section = "tldr"
                current_content = []
            elif line.strip().startswith("## ASSUMPTIONS"):
                current_section = "assumptions"
                current_content = []
            elif line.strip().startswith("## NEXT"):
                current_section = "next"
                current_content = []
            elif current_section:
                current_content.append(line)

            # Update section
            if current_section and (line.strip().startswith("## ") and len(current_content) > 0):
                sections[current_section] = "\n".join(current_content[:-1]).strip()
                current_content = []

        # Add last section
        if current_section and current_content:
            sections[current_section] = "\n".join(current_content).strip()

        # Default to full response if parsing failed
        if not sections["answer"]:
            sections["answer"] = response

        return sections

    async def evaluate_response(
        self,
        query: str,
        response: str,
        criteria: Optional[List[str]] = None
    ) -> float:
        """
        Evaluate response quality using LLM judge.

        Args:
            query: Original query
            response: Generated response
            criteria: Evaluation criteria (defaults to config)

        Returns:
            Quality score (0.0 to 1.0)
        """
        if not self.judge:
            return 1.0

        criteria = criteria or self.judge_config.criteria

        try:
            result = await self.judge.evaluate(
                query=query,
                response=response,
                criteria=criteria
            )

            # Extract score
            score = result.get("overall_score", 0.0)
            return score

        except Exception as e:
            logging.error(f"Judge evaluation failed: {e}")
            return 0.5  # Neutral score on error

    async def execute_workflow(
        self,
        workflow_name: str,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a named workflow loop.

        Args:
            workflow_name: Name of registered workflow
            inputs: Workflow input parameters

        Returns:
            Workflow execution results
        """
        if not self.loop_composer or workflow_name not in self.workflows:
            raise ValueError(f"Workflow '{workflow_name}' not found")

        workflow = self.workflows[workflow_name]

        try:
            start_time = datetime.now()
            result = await workflow.execute(inputs)
            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "workflow": workflow_name,
                "inputs": inputs,
                "result": result,
                "execution_time": execution_time,
                "success": True
            }

        except Exception as e:
            logging.error(f"Workflow '{workflow_name}' failed: {e}")
            return {
                "workflow": workflow_name,
                "inputs": inputs,
                "error": str(e),
                "success": False
            }

    async def ab_test_prompts(
        self,
        prompt_a: str,
        prompt_b: str,
        test_cases: List[Dict[str, str]],
        experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        A/B test two prompts to find the better one.

        Args:
            prompt_a: First prompt variant
            prompt_b: Second prompt variant
            test_cases: List of test inputs
            experiment_name: Optional experiment identifier

        Returns:
            A/B test results with winner
        """
        if not self.ab_tester:
            raise ValueError("A/B testing not enabled")

        if len(test_cases) < self.ab_test_config.min_test_cases:
            raise ValueError(f"Need at least {self.ab_test_config.min_test_cases} test cases")

        try:
            result = await self.ab_tester.compare(
                prompt_a=prompt_a,
                prompt_b=prompt_b,
                test_cases=test_cases
            )

            # Store experiment
            if experiment_name:
                self.active_experiments[experiment_name] = {
                    "result": result,
                    "timestamp": datetime.now(),
                    "test_cases": len(test_cases)
                }

            # Auto-promote winner
            if self.ab_test_config.auto_promote_winner:
                winner = result.get("winner")
                if winner:
                    logging.info(f"A/B test winner: {winner}")

            return result

        except Exception as e:
            logging.error(f"A/B test failed: {e}")
            return {"error": str(e), "success": False}

    def register_workflow(self, name: str, workflow_dsl: str):
        """Register a new workflow loop"""
        if not self.loop_composer:
            raise ValueError("Loop composition not enabled")

        try:
            workflow = self.loop_composer.parse(workflow_dsl)
            self.workflows[name] = workflow
            logging.info(f"Registered workflow: {name}")
        except Exception as e:
            logging.error(f"Failed to register workflow '{name}': {e}")
            raise

    def get_quality_statistics(self) -> Dict[str, Any]:
        """Get response quality statistics"""
        stats = self.quality_stats.copy()

        if stats["total_responses"] > 0:
            stats["high_quality_rate"] = stats["high_quality"] / stats["total_responses"]
            stats["low_quality_rate"] = stats["low_quality"] / stats["total_responses"]

        return stats

    def _update_quality_stats(self, score: float):
        """Update quality statistics"""
        self.quality_stats["total_responses"] += 1
        self.quality_stats["avg_score"] = (
            (self.quality_stats["avg_score"] * (self.quality_stats["total_responses"] - 1) + score) /
            self.quality_stats["total_responses"]
        )

        if score >= self.judge_config.min_score_threshold:
            self.quality_stats["high_quality"] += 1
        else:
            self.quality_stats["low_quality"] += 1

    async def _simple_response(self, query: str) -> str:
        """Fallback simple response"""
        return await self.promptly.execute(query)


# Example usage
async def demo_promptly_integration():
    """Demonstrate Promptly integration"""

    print("🎯 Promptly ChatOps Integration Demo\n")

    # Initialize
    bot = PromptlyEnhancedBot()

    # 1. Ultraprompt response
    print("1️⃣  Ultraprompt Response:")
    result = await bot.process_with_ultraprompt(
        query="Explain the benefits of using ChatOps for DevOps teams",
        use_hololoom=True
    )
    print(f"   TL;DR: {result.get('tldr', 'N/A')}")
    print(f"   Quality: {result.get('quality_score', 0.0):.2f}")
    print()

    # 2. Workflow execution
    print("2️⃣  Execute Research Workflow:")
    workflow_result = await bot.execute_workflow(
        "research",
        {"topic": "incident response automation"}
    )
    print(f"   Success: {workflow_result['success']}")
    print(f"   Time: {workflow_result.get('execution_time', 0):.2f}s")
    print()

    # 3. Quality statistics
    print("3️⃣  Quality Statistics:")
    stats = bot.get_quality_statistics()
    print(f"   Total responses: {stats['total_responses']}")
    print(f"   Avg quality: {stats['avg_score']:.2f}")
    print(f"   High quality rate: {stats.get('high_quality_rate', 0):.1%}")
    print()

    # 4. List workflows
    print("4️⃣  Available Workflows:")
    for name in bot.workflows.keys():
        print(f"   - {name}")
    print()


if __name__ == "__main__":
    asyncio.run(demo_promptly_integration())
