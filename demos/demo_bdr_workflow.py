"""
BDR Outbound Sequence - Demo Script

Demonstrates the complete 12-day outbound sales workflow with:
- Multi-query research (Day 0)
- Thompson Sampling subject line selection (Day 1)
- Conditional branching based on engagement (Days 3, 5, 7, 10)
- Background learning loop (continuous)

Run this to see the workflow in action without needing the full UI.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any

# HoloLoom imports
from hololoom.config import Config
from hololoom.agentic import AgenticOrchestrator, ReasoningMode
from hololoom.policy.unified import create_policy, BanditStrategy
from hololoom.documentation.types import Query, MemoryShard
from hololoom.alignment import SafetyGuardrails, AuditTrail
from hololoom.recursive import FullLearningEngine


class BDRWorkflow:
    """
    Complete BDR outbound sequence orchestrator.

    Simulates a 12-day sales cadence with Thompson Sampling optimization.
    """

    def __init__(self, config: Config):
        self.config = config
        self.guardrails = SafetyGuardrails(enable_human_in_loop=False)
        self.audit_trail = AuditTrail()

        # Thompson Sampling priors (will learn over time)
        self.subject_line_priors = {
            'funding_news': {'alpha': 45, 'beta': 12},
            'tech_stack': {'alpha': 38, 'beta': 15},
            'hiring_signal': {'alpha': 22, 'beta': 8},
            'pain_point': {'alpha': 15, 'beta': 20},
            'competitor_trigger': {'alpha': 10, 'beta': 5}
        }

        # Sequence state
        self.touchpoints: List[Dict] = []

    async def run_sequence(self, prospect_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the full 12-day BDR sequence.

        Args:
            prospect_data: Dict with keys:
                - prospect_name
                - company_name
                - persona_type
                - industry
                - company_size
                - our_category
                - trigger_event

        Returns:
            Dict with sequence outcome and analytics
        """
        print(f"\n{'='*70}")
        print(f"Starting BDR Sequence: {prospect_data['prospect_name']} @ {prospect_data['company_name']}")
        print(f"{'='*70}\n")

        # Create memory shards from prospect data
        shards = self._create_prospect_shards(prospect_data)

        # Day 0: Research Phase
        research_result = await self._day0_research(prospect_data, shards)

        # Day 1: Email Campaign
        email_result = await self._day1_email(prospect_data, research_result)

        # Day 3: Conditional Branch (opened?)
        day3_result = await self._day3_branch(prospect_data, email_result)

        # Day 5: Call Strategy
        day5_result = await self._day5_call(prospect_data, day3_result)

        # Day 7: Follow-up
        day7_result = await self._day7_followup(prospect_data, day5_result)

        # Day 10: Social Engagement
        day10_result = await self._day10_social(prospect_data, day7_result)

        # Day 12: Breakup Email
        day12_result = await self._day12_breakup(prospect_data, day10_result)

        # Generate Analytics
        analytics = self._generate_analytics(prospect_data)

        print(f"\n{'='*70}")
        print(f"Sequence Complete: {prospect_data['prospect_name']}")
        print(f"{'='*70}\n")

        return analytics

    async def _day0_research(self, prospect_data: Dict, shards: List[MemoryShard]) -> Dict:
        """Day 0: Deep research on prospect and company"""
        print("📚 DAY 0: RESEARCH PHASE")
        print("-" * 70)

        async with AgenticOrchestrator(cfg=self.config, shards=shards) as orchestrator:
            result = await orchestrator.reason(
                query=f"""
                Research {prospect_data['prospect_name']} at {prospect_data['company_name']}:
                - Recent news (funding, product launches, hiring)
                - Tech stack (from job posts, LinkedIn)
                - Pain points related to {prospect_data['our_category']}
                - Competitive intelligence
                """,
                mode=ReasoningMode.RESEARCH,
                max_steps=5
            )

        research = {
            'findings': result.response,
            'confidence': result.confidence,
            'steps': result.steps_taken
        }

        print(f"  ✓ Research complete ({len(result.steps_taken)} queries)")
        print(f"  ✓ Confidence: {result.confidence:.2f}")
        print(f"  ✓ Key findings: {result.response[:150]}...\n")

        self.touchpoints.append({
            'day': 0,
            'type': 'research',
            'outcome': 'success',
            'confidence': result.confidence
        })

        return research

    async def _day1_email(self, prospect_data: Dict, research: Dict) -> Dict:
        """Day 1: Thompson Sampling subject line selection + email generation"""
        print("📧 DAY 1: EMAIL CAMPAIGN")
        print("-" * 70)

        # Thompson Sampling: Select subject line variant
        selected_variant = self._thompson_sample(self.subject_line_priors)

        print(f"  🎲 Thompson Sampling selected: '{selected_variant}'")
        print(f"     Priors: α={self.subject_line_priors[selected_variant]['alpha']}, "
              f"β={self.subject_line_priors[selected_variant]['beta']}")

        expected_value = self.subject_line_priors[selected_variant]['alpha'] / (
            self.subject_line_priors[selected_variant]['alpha'] +
            self.subject_line_priors[selected_variant]['beta']
        )
        print(f"     Expected conversion: {expected_value:.1%}\n")

        # Generate personalized email
        shards = self._create_prospect_shards(prospect_data)

        async with AgenticOrchestrator(cfg=self.config, shards=shards) as orchestrator:
            result = await orchestrator.reason(
                query=f"""
                Write a personalized cold email to {prospect_data['prospect_name']} using:
                - Subject line variant: {selected_variant}
                - Context: {research['findings'][:300]}
                - Keep under 100 words
                - Value-first, specific CTA (15-min call)
                """,
                mode=ReasoningMode.DIRECT,
                max_steps=1
            )

        # Safety check
        gate_result = await self.guardrails.gate_action(
            'send_email',
            {'body': result.response}
        )

        if not gate_result.allowed:
            print(f"  ❌ Safety check FAILED: {gate_result.reason}")
            return {'sent': False, 'reason': gate_result.reason}

        print(f"  ✓ Email generated")
        print(f"  ✓ Safety check: PASSED (score {gate_result.safety_score:.2f})")

        # Simulate sending
        email_opened = self._simulate_email_open(expected_value)

        print(f"  ✓ Email sent")
        print(f"  {'✓' if email_opened else '✗'} Email {'opened' if email_opened else 'not opened'}\n")

        # Update Thompson priors
        if email_opened:
            self.subject_line_priors[selected_variant]['alpha'] += 1
        else:
            self.subject_line_priors[selected_variant]['beta'] += 1

        self.touchpoints.append({
            'day': 1,
            'type': 'email',
            'variant': selected_variant,
            'outcome': 'opened' if email_opened else 'not_opened',
            'confidence': result.confidence
        })

        return {
            'sent': True,
            'opened': email_opened,
            'variant': selected_variant,
            'body': result.response
        }

    async def _day3_branch(self, prospect_data: Dict, email_result: Dict) -> Dict:
        """Day 3: Conditional branch based on email engagement"""
        print("🔀 DAY 3: CONDITIONAL BRANCH")
        print("-" * 70)

        shards = self._create_prospect_shards(prospect_data)

        if email_result['opened']:
            # True path: LinkedIn connection
            print("  ✓ Email was opened → LinkedIn connection strategy\n")

            async with AgenticOrchestrator(cfg=self.config, shards=shards) as orchestrator:
                result = await orchestrator.reason(
                    query=f"""
                    Verify {prospect_data['prospect_name']} at {prospect_data['company_name']}
                    has pain point: scaling challenges post-{prospect_data['trigger_event']}.

                    Then write a 200-character LinkedIn connection note that:
                    - References their recent activity (if any)
                    - Mentions our expertise in {prospect_data['our_category']}
                    - No sales pitch
                    """,
                    mode=ReasoningMode.VERIFY,
                    max_steps=2
                )

            linkedin_accepted = self._simulate_linkedin_accept(0.4)  # 40% base rate

            print(f"  ✓ LinkedIn note generated")
            print(f"  {'✓' if linkedin_accepted else '✗'} Connection {'accepted' if linkedin_accepted else 'not accepted'}\n")

            self.touchpoints.append({
                'day': 3,
                'type': 'linkedin',
                'outcome': 'accepted' if linkedin_accepted else 'not_accepted',
                'confidence': result.confidence
            })

            return {'linkedin_accepted': linkedin_accepted}

        else:
            # False path: Retry with different subject
            print("  ✗ Email not opened → Retry with new subject line\n")

            # Select different variant
            used_variant = email_result['variant']
            remaining_variants = [v for v in self.subject_line_priors.keys() if v != used_variant]
            new_variant = self._thompson_sample({k: v for k, v in self.subject_line_priors.items() if k in remaining_variants})

            print(f"  🎲 New variant: '{new_variant}'\n")

            # Simulate retry (simplified - would actually send new email)
            retry_opened = self._simulate_email_open(0.15)  # Lower rate for retry

            self.touchpoints.append({
                'day': 3,
                'type': 'email_retry',
                'variant': new_variant,
                'outcome': 'opened' if retry_opened else 'not_opened'
            })

            return {'linkedin_accepted': False, 'retry_opened': retry_opened}

    async def _day5_call(self, prospect_data: Dict, day3_result: Dict) -> Dict:
        """Day 5: Call strategy (warm vs cold based on engagement)"""
        print("📞 DAY 5: PHONE OUTREACH")
        print("-" * 70)

        shards = self._create_prospect_shards(prospect_data)

        if day3_result.get('linkedin_accepted', False):
            # Warm call
            print("  ✓ LinkedIn accepted → Warm call strategy\n")

            async with AgenticOrchestrator(cfg=self.config, shards=shards) as orchestrator:
                result = await orchestrator.reason(
                    query=f"""
                    Plan a discovery call with {prospect_data['prospect_name']}:
                    - Context: Opened email, accepted LinkedIn connection
                    - Persona: {prospect_data['persona_type']} (analytical, risk-averse)
                    - Goal: Understand if scaling challenges are real, book demo

                    Generate:
                    1. Opening (reference specific touchpoints)
                    2. 3 discovery questions (persona-adapted)
                    3. Objection responses (pre-loaded from similar calls)
                    4. Next steps (demo vs follow-up email)
                    """,
                    mode=ReasoningMode.PLAN_EXECUTE,
                    max_steps=4
                )

            call_connected = self._simulate_call_connect(0.35)  # 35% for warm

        else:
            # Cold call
            print("  ✗ No engagement → Cold call strategy\n")

            async with AgenticOrchestrator(cfg=self.config, shards=shards) as orchestrator:
                result = await orchestrator.reason(
                    query=f"""
                    Plan a cold call to {prospect_data['prospect_name']}:
                    - Context: No engagement yet
                    - Pattern interrupt: Reference {prospect_data['trigger_event']}
                    - Persona: {prospect_data['persona_type']}

                    Generate cold call script with hook and discovery questions.
                    """,
                    mode=ReasoningMode.PLAN_EXECUTE,
                    max_steps=3
                )

            call_connected = self._simulate_call_connect(0.08)  # 8% for cold

        print(f"  ✓ Call script generated")
        print(f"  {'✓' if call_connected else '✗'} Call {'connected' if call_connected else 'went to voicemail'}\n")

        self.touchpoints.append({
            'day': 5,
            'type': 'call',
            'strategy': 'warm' if day3_result.get('linkedin_accepted') else 'cold',
            'outcome': 'connected' if call_connected else 'voicemail',
            'confidence': result.confidence
        })

        return {'call_connected': call_connected}

    async def _day7_followup(self, prospect_data: Dict, day5_result: Dict) -> Dict:
        """Day 7: Follow-up strategy based on engagement"""
        print("📬 DAY 7: FOLLOW-UP STRATEGY")
        print("-" * 70)

        # Check if any engagement (email opened, LinkedIn accepted, or call connected)
        any_engagement = any(
            t.get('outcome') in ['opened', 'accepted', 'connected']
            for t in self.touchpoints
        )

        shards = self._create_prospect_shards(prospect_data)

        if any_engagement:
            # Value-add content
            print("  ✓ Has engaged → Send value-add content\n")

            async with AgenticOrchestrator(cfg=self.config, shards=shards) as orchestrator:
                result = await orchestrator.reason(
                    query=f"""
                    Write a follow-up email to {prospect_data['prospect_name']} sharing:
                    - Case study or competitive intel relevant to {prospect_data['industry']}
                    - Reference their engagement (opened email/accepted LinkedIn)
                    - No ask, just value
                    - Include link to resource
                    """,
                    mode=ReasoningMode.DIRECT,
                    max_steps=1
                )

            print(f"  ✓ Value-add email sent (case study)\n")

            self.touchpoints.append({
                'day': 7,
                'type': 'followup',
                'strategy': 'value_add',
                'confidence': result.confidence
            })

            return {'strategy': 'value_add'}

        else:
            # Multi-thread: Find other contacts
            print("  ✗ No engagement → Multi-thread strategy\n")

            async with AgenticOrchestrator(cfg=self.config, shards=shards) as orchestrator:
                result = await orchestrator.reason(
                    query=f"""
                    Find alternative contacts at {prospect_data['company_name']}:
                    - Peers of {prospect_data['prospect_name']} in {prospect_data['persona_type']} role
                    - Decision-maker above {prospect_data['prospect_name']}
                    - Champions (users of {prospect_data['our_category']})
                    """,
                    mode=ReasoningMode.RESEARCH,
                    max_steps=3
                )

            print(f"  ✓ Found alternative contacts: {result.response[:100]}...\n")
            print(f"  → Starting fresh sequence with new contact\n")

            self.touchpoints.append({
                'day': 7,
                'type': 'followup',
                'strategy': 'multi_thread',
                'confidence': result.confidence
            })

            return {'strategy': 'multi_thread', 'new_contacts': result.response}

    async def _day10_social(self, prospect_data: Dict, day7_result: Dict) -> Dict:
        """Day 10: LinkedIn social engagement"""
        print("💬 DAY 10: SOCIAL ENGAGEMENT")
        print("-" * 70)

        # Simulate: Did prospect post on LinkedIn?
        posted_recently = self._simulate_linkedin_post(0.3)  # 30% chance

        if posted_recently:
            print("  ✓ Prospect posted on LinkedIn\n")

            shards = self._create_prospect_shards(prospect_data)

            async with AgenticOrchestrator(cfg=self.config, shards=shards) as orchestrator:
                result = await orchestrator.reason(
                    query=f"""
                    Prospect {prospect_data['prospect_name']} posted on LinkedIn:
                    "Just hit 100 engineers! Scaling is hard but worth it 🚀"

                    Write a genuine, insightful comment (not salesy) that:
                    - Shows you actually read it
                    - Adds insight (not just "great post!")
                    - Subtly relates to {prospect_data['our_category']}
                    """,
                    mode=ReasoningMode.DIRECT,
                    max_steps=1
                )

            print(f"  ✓ Comment posted: {result.response[:100]}...\n")

            self.touchpoints.append({
                'day': 10,
                'type': 'social',
                'outcome': 'comment_posted',
                'confidence': result.confidence
            })

            return {'engaged': True}

        else:
            print("  ✗ No recent LinkedIn activity → Skip to breakup\n")

            return {'engaged': False}

    async def _day12_breakup(self, prospect_data: Dict, day10_result: Dict) -> Dict:
        """Day 12: Graceful breakup email"""
        print("💔 DAY 12: BREAKUP EMAIL")
        print("-" * 70)

        shards = self._create_prospect_shards(prospect_data)

        async with AgenticOrchestrator(cfg=self.config, shards=shards) as orchestrator:
            result = await orchestrator.reason(
                query=f"""
                Write a breakup email to {prospect_data['prospect_name']}:
                - Context: {len(self.touchpoints)} touchpoints, limited engagement
                - Respect their time
                - Leave door open with trigger: "If you hit 200 engineers or raise Series C"
                - Offer opt-out: "Reply 'NOT A FIT' to never hear from me"
                - Keep friendly, professional
                """,
                mode=ReasoningMode.DIRECT,
                max_steps=1
            )

        print(f"  ✓ Breakup email sent\n")
        print(f"  Email preview:\n")
        print(f"  {'-'*66}")
        print(f"  {result.response[:300]}...")
        print(f"  {'-'*66}\n")

        # Simulate reply (18% reply rate for breakup emails!)
        breakup_reply = self._simulate_breakup_reply(0.18)

        if breakup_reply == 'lets_talk':
            print(f"  ✓ Prospect replied: 'Let's talk!' → MEETING BOOKED! 🎉\n")
            outcome = 'meeting_booked'
        elif breakup_reply == 'not_a_fit':
            print(f"  ✗ Prospect replied: 'Not a fit' → Archive forever\n")
            outcome = 'archived'
        else:
            print(f"  - No reply → Add to nurture campaign\n")
            outcome = 'nurture'

        self.touchpoints.append({
            'day': 12,
            'type': 'breakup',
            'outcome': outcome,
            'confidence': result.confidence
        })

        return {'outcome': outcome}

    def _generate_analytics(self, prospect_data: Dict) -> Dict:
        """Generate sequence analytics report"""
        print("\n" + "="*70)
        print("📊 SEQUENCE ANALYTICS")
        print("="*70 + "\n")

        # Calculate metrics
        total_touchpoints = len(self.touchpoints)
        engagement_rate = len([t for t in self.touchpoints if t.get('outcome') in
                               ['opened', 'accepted', 'connected', 'comment_posted']]) / total_touchpoints

        meeting_booked = any(t.get('outcome') == 'meeting_booked' for t in self.touchpoints)

        # Winning variants
        email_touchpoints = [t for t in self.touchpoints if t.get('type') == 'email']
        winning_variant = max(self.subject_line_priors.items(),
                             key=lambda x: x[1]['alpha'] / (x[1]['alpha'] + x[1]['beta']))

        analytics = {
            'prospect_name': prospect_data['prospect_name'],
            'company_name': prospect_data['company_name'],
            'duration_days': 12,
            'touchpoints_sent': total_touchpoints,
            'engagement_rate': engagement_rate,
            'meeting_booked': meeting_booked,
            'winning_variant': {
                'name': winning_variant[0],
                'conversion_rate': winning_variant[1]['alpha'] / (winning_variant[1]['alpha'] + winning_variant[1]['beta'])
            },
            'touchpoints': self.touchpoints
        }

        # Print report
        print(f"Prospect: {prospect_data['prospect_name']} @ {prospect_data['company_name']}")
        print(f"Persona: {prospect_data['persona_type']} | Industry: {prospect_data['industry']}")
        print(f"\nMetrics:")
        print(f"  - Duration: 12 days")
        print(f"  - Touchpoints: {total_touchpoints}")
        print(f"  - Engagement Rate: {engagement_rate:.1%}")
        print(f"  - Meeting Booked: {'YES ✓' if meeting_booked else 'NO ✗'}")
        print(f"\nWinning Variant (Thompson Sampling):")
        print(f"  - Subject: {winning_variant[0]}")
        print(f"  - Conversion: {analytics['winning_variant']['conversion_rate']:.1%}")
        print(f"  - Priors: α={winning_variant[1]['alpha']}, β={winning_variant[1]['beta']}")

        print(f"\nTouchpoint Timeline:")
        for t in self.touchpoints:
            outcome_emoji = {
                'opened': '✓',
                'not_opened': '✗',
                'accepted': '✓',
                'not_accepted': '✗',
                'connected': '✓',
                'voicemail': '✗',
                'comment_posted': '✓',
                'meeting_booked': '🎉',
                'archived': '🗄️',
                'nurture': '🌱'
            }.get(t.get('outcome', ''), '-')

            print(f"  Day {t['day']:2d}: {t['type']:15s} → {t.get('outcome', 'n/a'):20s} {outcome_emoji}")

        print("\n" + "="*70 + "\n")

        return analytics

    # Helper methods

    def _create_prospect_shards(self, prospect_data: Dict) -> List[MemoryShard]:
        """Create memory shards from prospect data"""
        return [
            MemoryShard(
                content=f"{prospect_data['prospect_name']} is a {prospect_data['persona_type']} at {prospect_data['company_name']}",
                source='prospect_profile',
                timestamp=datetime.now()
            ),
            MemoryShard(
                content=f"{prospect_data['company_name']} is in {prospect_data['industry']} industry, size {prospect_data['company_size']}",
                source='company_profile',
                timestamp=datetime.now()
            ),
            MemoryShard(
                content=f"Recent trigger event: {prospect_data['trigger_event']}",
                source='trigger_events',
                timestamp=datetime.now()
            )
        ]

    def _thompson_sample(self, priors: Dict) -> str:
        """Thompson Sampling: Sample from Beta distributions"""
        import random
        import numpy as np

        samples = {}
        for variant, params in priors.items():
            # Sample from Beta(α, β)
            samples[variant] = np.random.beta(params['alpha'], params['beta'])

        # Return variant with highest sample
        return max(samples.items(), key=lambda x: x[1])[0]

    def _simulate_email_open(self, base_rate: float) -> bool:
        """Simulate email open (with some randomness)"""
        import random
        return random.random() < base_rate

    def _simulate_linkedin_accept(self, base_rate: float) -> bool:
        """Simulate LinkedIn connection accept"""
        import random
        return random.random() < base_rate

    def _simulate_call_connect(self, base_rate: float) -> bool:
        """Simulate call connection"""
        import random
        return random.random() < base_rate

    def _simulate_linkedin_post(self, base_rate: float) -> bool:
        """Simulate LinkedIn posting activity"""
        import random
        return random.random() < base_rate

    def _simulate_breakup_reply(self, base_rate: float) -> str:
        """Simulate breakup email reply"""
        import random
        rand = random.random()
        if rand < base_rate * 0.5:  # 9% "let's talk"
            return 'lets_talk'
        elif rand < base_rate:  # 9% "not a fit"
            return 'not_a_fit'
        else:  # 82% no reply
            return None


async def main():
    """Run BDR workflow demo"""

    # Configure HoloLoom
    config = Config.fast()
    config.enable_alignment = True

    # Prospect data
    prospect = {
        'prospect_name': 'Sarah Chen',
        'company_name': 'Acme Fintech',
        'persona_type': 'engineering_vp',
        'industry': 'fintech',
        'company_size': '100-500',
        'our_category': 'API observability',
        'trigger_event': 'Series B funding announced 6 weeks ago'
    }

    # Run workflow
    workflow = BDRWorkflow(config)
    analytics = await workflow.run_sequence(prospect)

    # Export analytics
    with open('bdr_sequence_result.json', 'w') as f:
        json.dump(analytics, f, indent=2, default=str)

    print("✓ Analytics exported to bdr_sequence_result.json")


if __name__ == '__main__':
    asyncio.run(main())