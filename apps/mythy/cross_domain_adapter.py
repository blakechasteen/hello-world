#!/usr/bin/env python3
"""
🌐 CROSS-DOMAIN NARRATIVE ADAPTER
=================================
Extends narrative intelligence beyond mythology to universal domains.

Domains Supported:
1. BUSINESS - Startup journeys, corporate transformations, entrepreneurship
2. SCIENCE - Research breakthroughs, discovery narratives, paradigm shifts
3. PERSONAL - Therapy, coaching, self-improvement, life transitions
4. PRODUCT - Innovation stories, design thinking, product development
5. HISTORY - Political movements, revolutions, social change

Each domain has:
- Custom character archetypes
- Domain-specific Campbell stage mappings
- Specialized vocabulary and patterns
- Contextual truth database
- Adaptive complexity scoring

Universal Pattern Recognition:
The Hero's Journey is universal - whether you're:
- Founding a startup (crossing threshold = quitting day job)
- Making a scientific discovery (ordeal = failed experiments)
- Overcoming trauma (mentor = therapist)
- Creating a product (elixir = successful launch)
- Leading a revolution (return = new world order)
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from hololoom_narrative.intelligence import CampbellStage, ArchetypeType
from hololoom_narrative.matryoshka_depth import MatryoshkaNarrativeDepth, DepthLevel


class NarrativeDomain(Enum):
    """Universal narrative domains."""
    MYTHOLOGY = "mythology"        # Original: Gods, heroes, epic quests
    BUSINESS = "business"          # Startups, entrepreneurship, corporate
    SCIENCE = "science"            # Research, discovery, breakthroughs
    PERSONAL = "personal"          # Therapy, growth, transformation
    PRODUCT = "product"            # Innovation, design, development
    HISTORY = "history"            # Political, social, cultural change


@dataclass
class DomainCharacter:
    """Character archetype adapted to specific domain."""
    name: str
    archetype: str
    domain_role: str
    keywords: List[str]
    emotional_signature: Dict[str, float]


@dataclass
class DomainMapping:
    """Complete domain adaptation mapping."""
    domain: NarrativeDomain
    characters: List[DomainCharacter]
    stage_translations: Dict[CampbellStage, str]
    truth_database: List[str]
    pattern_keywords: Dict[str, List[str]]
    complexity_modifiers: Dict[str, float]


class CrossDomainAdapter:
    """
    Extensible narrative intelligence adapter for any domain.
    
    Translates universal Hero's Journey patterns into domain-specific language
    while maintaining the underlying archetypal structure.
    
    PLUGIN ARCHITECTURE:
    - Register custom domains at runtime
    - Extend with new character archetypes
    - Add domain-specific truth databases
    - Override complexity modifiers
    - Support hybrid/multi-domain analysis
    """
    
    def __init__(self):
        self.domains = self._initialize_domain_mappings()
        self.depth_analyzer = MatryoshkaNarrativeDepth()
        self._custom_domains: Dict[str, DomainMapping] = {}
    
    def register_domain(
        self,
        domain_name: str,
        mapping: DomainMapping
    ):
        """
        Register a custom domain for analysis.
        
        Args:
            domain_name: Unique identifier for the domain
            mapping: Complete domain mapping configuration
            
        Example:
            >>> adapter = CrossDomainAdapter()
            >>> medical_mapping = DomainMapping(...)
            >>> adapter.register_domain('medical', medical_mapping)
        """
        # Store in custom domains dict (not enum-based)
        self._custom_domains[domain_name] = mapping
        print(f"✅ Registered custom domain: {domain_name}")
    
    def list_domains(self) -> List[str]:
        """List all available domains (built-in + custom)."""
        return [d.value for d in self.domains.keys()] + list(self._custom_domains.keys())
    
    def get_domain_info(self, domain_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a specific domain."""
        # Check built-in domains
        for domain, mapping in self.domains.items():
            if domain.value == domain_name:
                return {
                    'name': domain.value,
                    'characters': len(mapping.characters),
                    'truths': len(mapping.truth_database),
                    'patterns': list(mapping.pattern_keywords.keys()),
                    'character_archetypes': [c.archetype for c in mapping.characters]
                }
        
        # Check custom domains
        if domain_name in self._custom_domains:
            mapping = self._custom_domains[domain_name]
            return {
                'name': domain_name,
                'characters': len(mapping.characters),
                'truths': len(mapping.truth_database),
                'patterns': list(mapping.pattern_keywords.keys()),
                'character_archetypes': [c.archetype for c in mapping.characters],
                'custom': True
            }
        
        return None
    
    def _initialize_domain_mappings(self) -> Dict[NarrativeDomain, DomainMapping]:
        """Initialize all domain-specific mappings."""
        return {
            NarrativeDomain.BUSINESS: self._create_business_mapping(),
            NarrativeDomain.SCIENCE: self._create_science_mapping(),
            NarrativeDomain.PERSONAL: self._create_personal_mapping(),
            NarrativeDomain.PRODUCT: self._create_product_mapping(),
            NarrativeDomain.HISTORY: self._create_history_mapping()
        }
    
    def _create_business_mapping(self) -> DomainMapping:
        """Business narrative domain."""
        characters = [
            DomainCharacter(
                name="Founder",
                archetype="hero",
                domain_role="Entrepreneur with vision",
                keywords=["founder", "ceo", "entrepreneur", "startup", "vision"],
                emotional_signature={"ambition": 1.0, "uncertainty": 0.8, "determination": 0.9}
            ),
            DomainCharacter(
                name="Mentor/Advisor",
                archetype="mentor",
                domain_role="Experienced advisor or investor",
                keywords=["advisor", "mentor", "investor", "board", "coach", "consultant"],
                emotional_signature={"wisdom": 1.0, "guidance": 0.9, "patience": 0.8}
            ),
            DomainCharacter(
                name="Market/Customer",
                archetype="threshold_guardian",
                domain_role="Customer needs and market forces",
                keywords=["customer", "market", "user", "demand", "competition"],
                emotional_signature={"skepticism": 0.7, "need": 0.8, "opportunity": 0.6}
            ),
            DomainCharacter(
                name="Co-founder",
                archetype="ally",
                domain_role="Partner in the journey",
                keywords=["co-founder", "partner", "team", "cto", "coo"],
                emotional_signature={"loyalty": 1.0, "support": 0.9, "collaboration": 1.0}
            ),
            DomainCharacter(
                name="Competitor",
                archetype="shadow",
                domain_role="Rival threatening success",
                keywords=["competitor", "rival", "threat", "disruption"],
                emotional_signature={"threat": 0.9, "challenge": 0.8, "pressure": 0.7}
            )
        ]
        
        stage_translations = {
            CampbellStage.ORDINARY_WORLD: "Pre-startup comfort zone (day job, corporate life)",
            CampbellStage.CALL_TO_ADVENTURE: "Business idea strikes / market opportunity identified",
            CampbellStage.REFUSAL_OF_CALL: "Fear of leaving stable job / self-doubt",
            CampbellStage.MEETING_MENTOR: "Finding advisor, investor, or business coach",
            CampbellStage.CROSSING_THRESHOLD: "Quitting day job / first customer / incorporation",
            CampbellStage.TESTS_ALLIES_ENEMIES: "Building team, testing product, facing competition",
            CampbellStage.APPROACH_INMOST_CAVE: "Approaching critical milestone (funding, launch)",
            CampbellStage.ORDEAL: "Near-death experience (running out of money, pivot)",
            CampbellStage.REWARD: "Product-market fit / funding secured / traction achieved",
            CampbellStage.ROAD_BACK: "Scaling challenges / maintaining growth",
            CampbellStage.RESURRECTION: "Final crisis overcome (acquisition offer, IPO, pivot)",
            CampbellStage.RETURN_WITH_ELIXIR: "Success achieved, lessons shared, ecosystem enriched"
        }
        
        truths = [
            "Startups are marathons disguised as sprints",
            "Product-market fit is found through iteration, not inspiration",
            "The best companies solve problems founders personally experienced",
            "Your first idea will pivot, your team will evolve, but vision endures",
            "Failure is tuition for entrepreneurial wisdom",
            "Markets don't care about your technology, only their problems",
            "The entrepreneur's journey transforms the founder more than the company",
            "Timing beats perfection; momentum beats planning",
            "Capital follows conviction, customers follow value",
            "Exit is not the goal; impact is the journey"
        ]
        
        patterns = {
            "crisis": ["pivot", "runway", "burn rate", "cash", "crisis", "failure", "layoff"],
            "growth": ["scale", "traction", "growth", "revenue", "users", "viral", "hockey stick"],
            "transformation": ["pivot", "rebrand", "restructure", "repositioning", "evolution"],
            "wisdom": ["lesson learned", "retrospective", "post-mortem", "reflection", "insight"]
        }
        
        return DomainMapping(
            domain=NarrativeDomain.BUSINESS,
            characters=characters,
            stage_translations=stage_translations,
            truth_database=truths,
            pattern_keywords=patterns,
            complexity_modifiers={"financial": 1.2, "team": 1.1, "market": 1.15}
        )
    
    def _create_science_mapping(self) -> DomainMapping:
        """Scientific discovery domain."""
        characters = [
            DomainCharacter(
                name="Researcher",
                archetype="hero",
                domain_role="Scientist pursuing truth",
                keywords=["scientist", "researcher", "professor", "phd", "investigator"],
                emotional_signature={"curiosity": 1.0, "rigor": 0.9, "persistence": 0.95}
            ),
            DomainCharacter(
                name="Advisor/PI",
                archetype="mentor",
                domain_role="Principal investigator or senior scientist",
                keywords=["advisor", "pi", "supervisor", "mentor", "senior scientist"],
                emotional_signature={"wisdom": 1.0, "experience": 0.9, "guidance": 0.85}
            ),
            DomainCharacter(
                name="Failed Experiment",
                archetype="threshold_guardian",
                domain_role="Nature's resistance to discovery",
                keywords=["failed", "negative result", "null hypothesis", "setback"],
                emotional_signature={"frustration": 0.8, "challenge": 0.9, "learning": 0.7}
            ),
            DomainCharacter(
                name="Collaborator",
                archetype="ally",
                domain_role="Research partner or lab mate",
                keywords=["collaborator", "co-author", "lab mate", "colleague"],
                emotional_signature={"support": 1.0, "collaboration": 0.95, "synergy": 0.9}
            )
        ]
        
        stage_translations = {
            CampbellStage.ORDINARY_WORLD: "Established scientific consensus / standard paradigm",
            CampbellStage.CALL_TO_ADVENTURE: "Anomalous result / unexplained phenomenon observed",
            CampbellStage.REFUSAL_OF_CALL: "Dismissing anomaly as error / fear of paradigm challenge",
            CampbellStage.MEETING_MENTOR: "Advisor encourages investigation / literature review",
            CampbellStage.CROSSING_THRESHOLD: "First experiment designed / hypothesis formulated",
            CampbellStage.TESTS_ALLIES_ENEMIES: "Experimental trials / peer review / replication attempts",
            CampbellStage.APPROACH_INMOST_CAVE: "Critical experiment designed / high-stakes trial",
            CampbellStage.ORDEAL: "Repeated failures / funding crisis / paradigm resistance",
            CampbellStage.REWARD: "Breakthrough data / unexpected insight / eureka moment",
            CampbellStage.ROAD_BACK: "Writing paper / facing peer review / defending findings",
            CampbellStage.RESURRECTION: "Publication / conference presentation / paradigm shift",
            CampbellStage.RETURN_WITH_ELIXIR: "New knowledge shared / field transformed / legacy created"
        }
        
        truths = [
            "Discovery requires the courage to be wrong publicly",
            "Failed experiments teach more than successful ones",
            "Paradigm shifts begin with anomalies dismissed by experts",
            "Science advances one funeral at a time (old paradigms die hard)",
            "The most interesting discoveries contradict expectations",
            "Replication is respect; skepticism is science",
            "Breakthroughs come from asking 'what if we're wrong?'",
            "Nature reveals truth to those who listen without agenda",
            "Every experiment is conversation with reality",
            "Knowledge grows through collaborative humility"
        ]
        
        patterns = {
            "discovery": ["breakthrough", "discovery", "eureka", "insight", "revelation", "found"],
            "struggle": ["failed", "negative result", "rejected", "setback", "obstacle"],
            "validation": ["replicated", "confirmed", "validated", "peer-reviewed", "published"],
            "paradigm": ["paradigm shift", "revolutionary", "overturned", "challenged"]
        }
        
        return DomainMapping(
            domain=NarrativeDomain.SCIENCE,
            characters=characters,
            stage_translations=stage_translations,
            truth_database=truths,
            pattern_keywords=patterns,
            complexity_modifiers={"experimental": 1.3, "theoretical": 1.2, "paradigm": 1.4}
        )
    
    def _create_personal_mapping(self) -> DomainMapping:
        """Personal development/therapy domain."""
        characters = [
            DomainCharacter(
                name="Self",
                archetype="hero",
                domain_role="Individual seeking growth",
                keywords=["i", "me", "myself", "my journey", "personal"],
                emotional_signature={"vulnerability": 0.9, "courage": 0.8, "growth": 1.0}
            ),
            DomainCharacter(
                name="Therapist/Coach",
                archetype="mentor",
                domain_role="Guide through transformation",
                keywords=["therapist", "coach", "counselor", "guide", "teacher"],
                emotional_signature={"compassion": 1.0, "wisdom": 0.9, "safety": 0.95}
            ),
            DomainCharacter(
                name="Inner Critic",
                archetype="shadow",
                domain_role="Internal resistance and self-doubt",
                keywords=["fear", "doubt", "shame", "guilt", "inner critic"],
                emotional_signature={"fear": 1.0, "resistance": 0.9, "protection": 0.7}
            ),
            DomainCharacter(
                name="Support System",
                archetype="ally",
                domain_role="Friends, family, community",
                keywords=["friend", "family", "support", "community", "loved ones"],
                emotional_signature={"love": 1.0, "support": 0.95, "acceptance": 0.9}
            )
        ]
        
        stage_translations = {
            CampbellStage.ORDINARY_WORLD: "Life before awareness / unconscious patterns",
            CampbellStage.CALL_TO_ADVENTURE: "Crisis, pain, or dissatisfaction calls for change",
            CampbellStage.REFUSAL_OF_CALL: "Denial, avoidance, or fear of facing truth",
            CampbellStage.MEETING_MENTOR: "Finding therapist, coach, or wise guide",
            CampbellStage.CROSSING_THRESHOLD: "First therapy session / commitment to change",
            CampbellStage.TESTS_ALLIES_ENEMIES: "Confronting patterns / building new habits",
            CampbellStage.APPROACH_INMOST_CAVE: "Approaching core wound or trauma",
            CampbellStage.ORDEAL: "Dark night of the soul / facing deepest pain",
            CampbellStage.REWARD: "Insight, healing, or breakthrough understanding",
            CampbellStage.ROAD_BACK: "Integrating changes / practicing new ways",
            CampbellStage.RESURRECTION: "Final test of transformation / choosing new self",
            CampbellStage.RETURN_WITH_ELIXIR: "Living authentically / helping others heal"
        }
        
        truths = [
            "Healing begins when we stop running from pain",
            "The wound is where light enters (Rumi)",
            "You cannot heal what you refuse to feel",
            "Transformation requires letting go of who you were",
            "Vulnerability is strength, not weakness",
            "The inner critic protects an inner child",
            "Growth happens at the edge of comfort",
            "Authenticity is the courage to be imperfect",
            "We teach what we most needed to learn",
            "The journey inward is the journey home"
        ]
        
        patterns = {
            "breakthrough": ["insight", "realization", "understanding", "clarity", "aha"],
            "struggle": ["pain", "suffering", "difficulty", "resistance", "fear"],
            "healing": ["healing", "recovery", "growth", "transformation", "integration"],
            "wisdom": ["learned", "realized", "understood", "accepted", "embraced"]
        }
        
        return DomainMapping(
            domain=NarrativeDomain.PERSONAL,
            characters=characters,
            stage_translations=stage_translations,
            truth_database=truths,
            pattern_keywords=patterns,
            complexity_modifiers={"trauma": 1.4, "breakthrough": 1.3, "integration": 1.2}
        )
    
    def _create_product_mapping(self) -> DomainMapping:
        """Product development/innovation domain."""
        characters = [
            DomainCharacter(
                name="Product Manager",
                archetype="hero",
                domain_role="Champion of user needs",
                keywords=["pm", "product manager", "product owner", "designer"],
                emotional_signature={"vision": 1.0, "empathy": 0.9, "determination": 0.85}
            ),
            DomainCharacter(
                name="User/Customer",
                archetype="mentor",
                domain_role="Source of truth and validation",
                keywords=["user", "customer", "feedback", "interview", "research"],
                emotional_signature={"need": 1.0, "pain": 0.8, "desire": 0.9}
            ),
            DomainCharacter(
                name="Technical Constraints",
                archetype="threshold_guardian",
                domain_role="Feasibility challenges",
                keywords=["constraint", "limitation", "technical debt", "feasibility"],
                emotional_signature={"challenge": 0.8, "reality": 0.9, "boundary": 0.7}
            ),
            DomainCharacter(
                name="Team",
                archetype="ally",
                domain_role="Engineers, designers, stakeholders",
                keywords=["team", "engineer", "designer", "developer", "stakeholder"],
                emotional_signature={"collaboration": 1.0, "creativity": 0.9, "execution": 0.85}
            )
        ]
        
        stage_translations = {
            CampbellStage.ORDINARY_WORLD: "Existing solution / status quo product",
            CampbellStage.CALL_TO_ADVENTURE: "User pain point discovered / market gap identified",
            CampbellStage.REFUSAL_OF_CALL: "Too hard to build / stakeholder resistance",
            CampbellStage.MEETING_MENTOR: "User research / customer discovery / empathy",
            CampbellStage.CROSSING_THRESHOLD: "First prototype / MVP decision",
            CampbellStage.TESTS_ALLIES_ENEMIES: "User testing / iterations / feedback loops",
            CampbellStage.APPROACH_INMOST_CAVE: "Beta launch / early adopter access",
            CampbellStage.ORDEAL: "Critical bug / user rejection / redesign needed",
            CampbellStage.REWARD: "Product-market fit / positive metrics / user love",
            CampbellStage.ROAD_BACK: "Scaling product / broader launch / optimization",
            CampbellStage.RESURRECTION: "Final pivot or refinement / v1.0 launch",
            CampbellStage.RETURN_WITH_ELIXIR: "Product solves problem / users transformed"
        }
        
        truths = [
            "Users don't want your product; they want their problem solved",
            "Fall in love with the problem, not the solution",
            "The best feature is the one you don't build",
            "Early feedback is gift; ignore it at your peril",
            "Perfect is the enemy of shipped",
            "Every product is a hypothesis awaiting validation",
            "Design for one user, optimize for thousands",
            "Constraints breed creativity; freedom breeds confusion",
            "Iterate until the solution feels obvious in hindsight",
            "Great products emerge from deep user empathy"
        ]
        
        patterns = {
            "discovery": ["insight", "research", "learned", "discovered", "uncovered"],
            "iteration": ["iterate", "pivot", "redesign", "refine", "improve"],
            "validation": ["tested", "validated", "confirmed", "metrics", "adoption"],
            "launch": ["launch", "ship", "release", "deploy", "beta"]
        }
        
        return DomainMapping(
            domain=NarrativeDomain.PRODUCT,
            characters=characters,
            stage_translations=stage_translations,
            truth_database=truths,
            pattern_keywords=patterns,
            complexity_modifiers={"user_research": 1.2, "iteration": 1.15, "launch": 1.3}
        )
    
    def _create_history_mapping(self) -> DomainMapping:
        """Historical/political movements domain."""
        characters = [
            DomainCharacter(
                name="Revolutionary Leader",
                archetype="hero",
                domain_role="Agent of social change",
                keywords=["leader", "activist", "revolutionary", "movement", "organizer"],
                emotional_signature={"courage": 1.0, "conviction": 0.95, "sacrifice": 0.9}
            ),
            DomainCharacter(
                name="Predecessor/Inspiration",
                archetype="mentor",
                domain_role="Historical precedent or ideology",
                keywords=["inspired by", "following", "tradition", "legacy", "precedent"],
                emotional_signature={"wisdom": 1.0, "legacy": 0.9, "guidance": 0.85}
            ),
            DomainCharacter(
                name="Establishment/Status Quo",
                archetype="shadow",
                domain_role="Existing power structure",
                keywords=["establishment", "regime", "system", "authority", "opposition"],
                emotional_signature={"resistance": 1.0, "power": 0.9, "threat": 0.85}
            ),
            DomainCharacter(
                name="The People",
                archetype="ally",
                domain_role="Mass movement or supporters",
                keywords=["people", "masses", "movement", "supporters", "citizens"],
                emotional_signature={"hope": 1.0, "unity": 0.9, "momentum": 0.85}
            )
        ]
        
        stage_translations = {
            CampbellStage.ORDINARY_WORLD: "Pre-revolutionary society / stable but unjust order",
            CampbellStage.CALL_TO_ADVENTURE: "Inciting incident / injustice becomes unbearable",
            CampbellStage.REFUSAL_OF_CALL: "Fear of consequences / initial hesitation",
            CampbellStage.MEETING_MENTOR: "Discovering ideology / historical precedent / manifesto",
            CampbellStage.CROSSING_THRESHOLD: "First public action / protest / declaration",
            CampbellStage.TESTS_ALLIES_ENEMIES: "Building coalition / facing opposition / organizing",
            CampbellStage.APPROACH_INMOST_CAVE: "Confronting power directly / major demonstration",
            CampbellStage.ORDEAL: "Violent suppression / crisis moment / martyrdom",
            CampbellStage.REWARD: "Victory in battle / public support / momentum shifts",
            CampbellStage.ROAD_BACK: "Consolidating gains / building new institutions",
            CampbellStage.RESURRECTION: "Final confrontation / decisive moment",
            CampbellStage.RETURN_WITH_ELIXIR: "New social order / transformed society / legacy"
        }
        
        truths = [
            "Revolutions begin when suffering outweighs fear",
            "History is written by those who show up",
            "Power concedes nothing without a demand",
            "The arc of moral universe bends toward justice, but slowly",
            "Every revolution was once an impossible dream",
            "Martyrs create movements; movements create change",
            "Old orders die not from opposition but obsolescence",
            "Liberation requires solidarity across differences",
            "The people united cannot be divided",
            "Change comes from collective courage, not individual heroism"
        ]
        
        patterns = {
            "uprising": ["protest", "revolution", "uprising", "resistance", "rebellion"],
            "oppression": ["tyranny", "injustice", "oppression", "persecution", "suppression"],
            "liberation": ["freedom", "liberation", "independence", "rights", "justice"],
            "transformation": ["paradigm shift", "new order", "transformed", "reformed"]
        }
        
        return DomainMapping(
            domain=NarrativeDomain.HISTORY,
            characters=characters,
            stage_translations=stage_translations,
            truth_database=truths,
            pattern_keywords=patterns,
            complexity_modifiers={"revolution": 1.4, "social_change": 1.3, "paradigm": 1.35}
        )
    
    async def analyze_with_domain(
        self,
        text: str,
        domain: Optional[NarrativeDomain] = None,
        domain_name: Optional[str] = None,
        auto_detect: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze narrative with domain-specific adaptation.
        
        Args:
            text: Text to analyze
            domain: Target narrative domain (enum)
            domain_name: Domain name (string, for custom domains)
            auto_detect: Auto-detect best domain from text
            
        Returns:
            Domain-adapted analysis with translated insights
        """
        # Auto-detect domain if requested
        if auto_detect:
            domain_name = self._auto_detect_domain(text)
            print(f"🔍 Auto-detected domain: {domain_name}")
        
        # Get domain mapping
        mapping = None
        if domain:
            mapping = self.domains.get(domain)
        elif domain_name:
            # Check custom domains first
            if domain_name in self._custom_domains:
                mapping = self._custom_domains[domain_name]
            else:
                # Try to find in built-in domains
                for d, m in self.domains.items():
                    if d.value == domain_name:
                        mapping = m
                        domain = d
                        break
        
        if not mapping:
            # Default to mythology
            domain = NarrativeDomain.MYTHOLOGY
            mapping = self.domains[domain]
        
        # Perform base depth analysis
        depth_result = await self.depth_analyzer.analyze_depth(text)
        
        # Detect domain-specific characters
        detected_characters = self._detect_domain_characters(text, mapping)
        
        # Translate Campbell stage to domain language
        campbell_stage = self._infer_campbell_stage(depth_result, detected_characters)
        domain_stage = mapping.stage_translations.get(
            campbell_stage,
            "Unknown stage in this domain"
        )
        
        # Find relevant domain truths
        relevant_truths = self._find_relevant_truths(text, mapping)
        
        # Calculate domain-adjusted complexity
        adjusted_complexity = self._adjust_complexity_for_domain(
            depth_result.total_complexity,
            text,
            mapping
        )
        
        return {
            'domain': domain.value if domain else domain_name,
            'base_analysis': {
                'max_depth': depth_result.max_depth_achieved.name,
                'complexity': depth_result.total_complexity,
                'adjusted_complexity': adjusted_complexity,
                'confidence': depth_result.bayesian_confidence
            },
            'domain_translation': {
                'campbell_stage': campbell_stage.value if campbell_stage else None,
                'domain_interpretation': domain_stage,
                'characters_detected': detected_characters,
                'relevant_truths': relevant_truths
            },
            'insights': {
                'deepest_meaning': depth_result.deepest_meaning,
                'transformation_journey': depth_result.transformation_journey,
                'cosmic_truth': depth_result.cosmic_truth
            }
        }
    
    def _auto_detect_domain(self, text: str) -> str:
        """Auto-detect most likely domain from text content."""
        text_lower = text.lower()
        
        # Score each domain
        scores = {}
        for domain, mapping in self.domains.items():
            score = 0
            # Check character keywords
            for char in mapping.characters:
                for kw in char.keywords:
                    if kw in text_lower:
                        score += 1
            # Check pattern keywords
            for pattern, keywords in mapping.pattern_keywords.items():
                for kw in keywords:
                    if kw in text_lower:
                        score += 0.5
            scores[domain.value] = score
        
        # Check custom domains
        for domain_name, mapping in self._custom_domains.items():
            score = 0
            for char in mapping.characters:
                for kw in char.keywords:
                    if kw in text_lower:
                        score += 1
            for pattern, keywords in mapping.pattern_keywords.items():
                for kw in keywords:
                    if kw in text_lower:
                        score += 0.5
            scores[domain_name] = score
        
        # Return highest scoring domain
        if not scores:
            return 'mythology'
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _detect_domain_characters(
        self,
        text: str,
        mapping: DomainMapping
    ) -> List[Dict[str, Any]]:
        """Detect domain-specific characters in text."""
        text_lower = text.lower()
        detected = []
        
        for character in mapping.characters:
            # Check if any keywords match
            matches = [kw for kw in character.keywords if kw in text_lower]
            if matches:
                detected.append({
                    'name': character.name,
                    'archetype': character.archetype,
                    'role': character.domain_role,
                    'matched_keywords': matches,
                    'emotional_signature': character.emotional_signature
                })
        
        return detected
    
    def _infer_campbell_stage(
        self,
        depth_result,
        characters: List[Dict]
    ) -> Optional[CampbellStage]:
        """Infer Campbell stage from analysis and characters."""
        # Simple heuristic based on characters and patterns
        if any(c['archetype'] == 'mentor' for c in characters):
            return CampbellStage.MEETING_MENTOR
        elif any(c['archetype'] == 'threshold_guardian' for c in characters):
            return CampbellStage.CROSSING_THRESHOLD
        elif any(c['archetype'] == 'shadow' for c in characters):
            return CampbellStage.ORDEAL
        else:
            return CampbellStage.CALL_TO_ADVENTURE
    
    def _find_relevant_truths(
        self,
        text: str,
        mapping: DomainMapping
    ) -> List[str]:
        """Find domain truths relevant to the text."""
        # Simple keyword matching for now
        # Could be enhanced with semantic similarity
        return mapping.truth_database[:3]  # Return top 3 for now
    
    def _adjust_complexity_for_domain(
        self,
        base_complexity: float,
        text: str,
        mapping: DomainMapping
    ) -> float:
        """Adjust complexity score based on domain-specific patterns."""
        text_lower = text.lower()
        adjustment = 1.0
        
        # Check for domain-specific complexity modifiers
        for pattern, keywords in mapping.pattern_keywords.items():
            if any(kw in text_lower for kw in keywords):
                modifier = mapping.complexity_modifiers.get(pattern, 1.0)
                adjustment = max(adjustment, modifier)
        
        return min(base_complexity * adjustment, 1.0)


async def demonstrate_cross_domain():
    """Demonstrate cross-domain narrative analysis."""
    print("🌐 CROSS-DOMAIN NARRATIVE ADAPTATION")
    print("=" * 80)
    print()
    
    adapter = CrossDomainAdapter()
    
    # Test cases for each domain
    test_cases = [
        {
            'domain': NarrativeDomain.BUSINESS,
            'text': '''Sarah quit her corporate job to build a startup solving a problem she 
            personally experienced. Her advisor, an experienced entrepreneur, warned her: "You'll 
            pivot three times before finding product-market fit." Months of failed experiments 
            followed, cash dwindling. Then one customer email changed everything: "This solved my 
            biggest pain point." The journey had just begun, but Sarah now understood - startups 
            aren't about the idea, but the transformation of the founder.'''
        },
        {
            'domain': NarrativeDomain.SCIENCE,
            'text': '''Dr. Chen's experiment contradicted 50 years of established theory. Her PI 
            initially dismissed it as error. But after the third replication, they couldn't ignore 
            the anomaly. What followed were months of failed attempts to explain it with existing 
            models. Then, during a sleepless night, Chen realized: what if the entire paradigm 
            was wrong? The breakthrough came not from new data, but from the courage to question 
            everything. Publication changed the field forever.'''
        },
        {
            'domain': NarrativeDomain.PERSONAL,
            'text': '''I sat in the therapist's office, finally facing what I'd avoided for years. 
            "The pain you're running from," she said gently, "is trying to teach you something." 
            The journey inward was harder than any external challenge. My inner critic screamed 
            with every step toward truth. But as I learned to sit with discomfort, something shifted. 
            The wound became a doorway. Healing meant accepting all of myself - shadow and light. 
            Now I help others find their way home to themselves.'''
        },
        {
            'domain': NarrativeDomain.PRODUCT,
            'text': '''The user interviews revealed a pain point we never anticipated. Our beautiful 
            design solved the wrong problem. The team resisted: "But we already built it!" The PM 
            had a choice - defend the solution or return to the problem. We scrapped everything and 
            started over. The new MVP was ugly but useful. Early adopters loved it. Iteration by 
            iteration, guided by feedback, the product emerged. Launch day proved the truth: fall 
            in love with the problem, and users will fall in love with your solution.'''
        },
        {
            'domain': NarrativeDomain.HISTORY,
            'text': '''The protestors gathered despite warnings. The regime had suppressed dissent 
            for decades, but this injustice was unbearable. Leaders emerged from ordinary citizens - 
            teachers, students, workers united. The first demonstration was small. They expected 
            violence; they received it. But martyrdom created momentum. Each crackdown spawned 
            ten more protests. The people, once divided, found solidarity. When the masses finally 
            flooded the capital, the old order crumbled. A new society emerged, built on the 
            courage of those who said "enough."'''
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        domain = test['domain']
        text = test['text']
        
        print(f"🎬 Domain {i}/5: {domain.value.upper()}")
        print("=" * 80)
        print(f"Text: {text[:100]}...")
        print()
        
        result = await adapter.analyze_with_domain(text, domain)
        
        print(f"📊 BASE ANALYSIS:")
        print(f"   Max Depth: {result['base_analysis']['max_depth']}")
        print(f"   Base Complexity: {result['base_analysis']['complexity']:.3f}")
        print(f"   Domain-Adjusted: {result['base_analysis']['adjusted_complexity']:.3f}")
        print(f"   Confidence: {result['base_analysis']['confidence']:.3f}")
        print()
        
        print(f"🌐 DOMAIN TRANSLATION:")
        print(f"   Campbell Stage: {result['domain_translation']['campbell_stage']}")
        print(f"   In {domain.value}: {result['domain_translation']['domain_interpretation']}")
        print()
        
        if result['domain_translation']['characters_detected']:
            print(f"👥 CHARACTERS DETECTED:")
            for char in result['domain_translation']['characters_detected'][:3]:
                print(f"   • {char['name']} ({char['archetype']})")
                print(f"     Role: {char['role']}")
                print(f"     Keywords: {', '.join(char['matched_keywords'][:3])}")
            print()
        
        print(f"💎 DOMAIN TRUTHS:")
        for truth in result['domain_translation']['relevant_truths']:
            print(f"   • {truth}")
        print()
        
        if result['insights']['cosmic_truth']:
            print(f"🌌 UNIVERSAL TRUTH:")
            print(f"   {result['insights']['cosmic_truth']}")
            print()
        
        print("=" * 80)
        print()
    
    print("✅ CROSS-DOMAIN ADAPTATION COMPLETE!")
    print()
    print("🎯 KEY INSIGHT:")
    print("   The Hero's Journey is UNIVERSAL - whether you're:")
    print("   • Building a startup")
    print("   • Making a discovery")
    print("   • Healing trauma")
    print("   • Creating a product")
    print("   • Leading a revolution")
    print()
    print("   The pattern remains: Call → Journey → Transformation → Return")
    print("   Only the language changes to match the domain!")
    print("=" * 80)


class DomainPluginBuilder:
    """
    Builder pattern for creating custom domain plugins.
    
    Makes it easy to extend the adapter with new domains.
    """
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.characters = []
        self.stage_translations = {}
        self.truth_database = []
        self.pattern_keywords = {}
        self.complexity_modifiers = {}
    
    def add_character(
        self,
        name: str,
        archetype: str,
        domain_role: str,
        keywords: List[str],
        emotional_signature: Optional[Dict[str, float]] = None
    ) -> 'DomainPluginBuilder':
        """Add a character archetype to the domain."""
        self.characters.append(DomainCharacter(
            name=name,
            archetype=archetype,
            domain_role=domain_role,
            keywords=keywords,
            emotional_signature=emotional_signature or {}
        ))
        return self
    
    def map_stage(
        self,
        campbell_stage: CampbellStage,
        domain_translation: str
    ) -> 'DomainPluginBuilder':
        """Map a Campbell stage to domain-specific language."""
        self.stage_translations[campbell_stage] = domain_translation
        return self
    
    def add_truth(self, truth: str) -> 'DomainPluginBuilder':
        """Add a domain-specific truth."""
        self.truth_database.append(truth)
        return self
    
    def add_pattern(
        self,
        pattern_name: str,
        keywords: List[str],
        complexity_modifier: float = 1.0
    ) -> 'DomainPluginBuilder':
        """Add a narrative pattern with keywords."""
        self.pattern_keywords[pattern_name] = keywords
        self.complexity_modifiers[pattern_name] = complexity_modifier
        return self
    
    def build(self) -> DomainMapping:
        """Build the complete domain mapping."""
        # Create a custom domain enum (this is a hack but works)
        try:
            custom_domain = NarrativeDomain(self.domain_name)
        except ValueError:
            # Domain doesn't exist in enum, create synthetic one
            custom_domain = type('DynamicDomain', (), {'value': self.domain_name})()
        
        return DomainMapping(
            domain=custom_domain,
            characters=self.characters,
            stage_translations=self.stage_translations,
            truth_database=self.truth_database,
            pattern_keywords=self.pattern_keywords,
            complexity_modifiers=self.complexity_modifiers
        )


async def demonstrate_plugin_system():
    """Demonstrate the plugin system with a custom domain."""
    print()
    print("🔌 EXTENSIBLE PLUGIN SYSTEM DEMO")
    print("=" * 80)
    print()
    
    adapter = CrossDomainAdapter()
    
    print("📦 Building custom domain: MEDICAL/HEALTHCARE")
    print()
    
    # Build a medical/healthcare domain plugin
    medical_domain = (
        DomainPluginBuilder('medical')
        .add_character(
            name='Patient',
            archetype='hero',
            domain_role='Individual facing health challenge',
            keywords=['patient', 'diagnosed', 'illness', 'treatment', 'recovery'],
            emotional_signature={'fear': 0.9, 'hope': 0.7, 'vulnerability': 0.95}
        )
        .add_character(
            name='Doctor/Healer',
            archetype='mentor',
            domain_role='Medical guide and expert',
            keywords=['doctor', 'physician', 'healer', 'specialist', 'surgeon'],
            emotional_signature={'wisdom': 0.9, 'compassion': 0.85, 'skill': 0.95}
        )
        .add_character(
            name='Disease/Condition',
            archetype='shadow',
            domain_role='The adversary to overcome',
            keywords=['cancer', 'disease', 'illness', 'condition', 'diagnosis'],
            emotional_signature={'threat': 1.0, 'fear': 0.9, 'challenge': 0.95}
        )
        .add_character(
            name='Support System',
            archetype='ally',
            domain_role='Family, friends, caregivers',
            keywords=['family', 'caregiver', 'support', 'loved ones', 'nurse'],
            emotional_signature={'love': 1.0, 'support': 0.95, 'strength': 0.8}
        )
        .map_stage(CampbellStage.ORDINARY_WORLD, "Life before diagnosis / healthy baseline")
        .map_stage(CampbellStage.CALL_TO_ADVENTURE, "Symptoms appear / diagnosis received")
        .map_stage(CampbellStage.REFUSAL_OF_CALL, "Denial / avoiding treatment")
        .map_stage(CampbellStage.MEETING_MENTOR, "Finding the right doctor / specialist")
        .map_stage(CampbellStage.CROSSING_THRESHOLD, "Beginning treatment / first procedure")
        .map_stage(CampbellStage.TESTS_ALLIES_ENEMIES, "Treatment cycles / side effects / progress")
        .map_stage(CampbellStage.ORDEAL, "Crisis point / complications / near death")
        .map_stage(CampbellStage.REWARD, "Remission / healing begins / turning point")
        .map_stage(CampbellStage.ROAD_BACK, "Recovery / rehabilitation / new normal")
        .map_stage(CampbellStage.RESURRECTION, "Final tests / confirmation of healing")
        .map_stage(CampbellStage.RETURN_WITH_ELIXIR, "Survivor / advocate / wisdom shared")
        .add_truth("Healing is a journey, not a destination")
        .add_truth("The body knows how to heal; medicine removes obstacles")
        .add_truth("Every patient's journey is unique; protocols are guidelines")
        .add_truth("Hope is medicine; fear is poison")
        .add_truth("Survival transforms; you cannot return to who you were")
        .add_pattern('diagnosis', ['diagnosed', 'test results', 'biopsy', 'scan'], 1.3)
        .add_pattern('treatment', ['chemotherapy', 'surgery', 'radiation', 'therapy'], 1.2)
        .add_pattern('recovery', ['remission', 'healed', 'recovered', 'survivor'], 1.4)
        .build()
    )
    
    # Register the custom domain
    adapter.register_domain('medical', medical_domain)
    
    print("✅ Medical domain registered!")
    print()
    print("📋 Available domains:", ', '.join(adapter.list_domains()))
    print()
    
    # Test with a medical narrative
    medical_story = """
    When the doctor said 'cancer', time stopped. I was 42, healthy, with two kids. 
    The diagnosis felt impossible. At first, I refused to believe it - more tests, 
    second opinions, denial. But Dr. Sarah, my oncologist, sat with me and explained: 
    "This is your journey now, and I'll guide you through it."
    
    Treatment began - chemotherapy every two weeks. My family became my strength, holding 
    me through the worst days. Each cycle was a test: nausea, fatigue, fear. But slowly, 
    the tumors shrank. Then came the ordeal - complications, infection, hospitalization. 
    I touched the edge of death and pulled back.
    
    Six months later, the word I'd dreamed of: remission. The journey back to life was 
    strange - nothing felt the same. I wasn't who I was before. The final tests confirmed: 
    cancer-free. Now I volunteer at the cancer center, sharing hope with newly diagnosed 
    patients. The gift I received wasn't just survival - it was wisdom about what truly 
    matters in life.
    """
    
    print("🏥 Analyzing medical narrative...")
    print()
    
    result = await adapter.analyze_with_domain(medical_story, domain_name='medical')
    
    print(f"Domain: {result['domain'].upper()}")
    print(f"Max Depth: {result['base_analysis']['max_depth']}")
    print(f"Complexity: {result['base_analysis']['complexity']:.3f}")
    print(f"Adjusted: {result['base_analysis']['adjusted_complexity']:.3f}")
    print()
    
    print(f"Campbell Stage: {result['domain_translation']['campbell_stage']}")
    print(f"Medical Translation: {result['domain_translation']['domain_interpretation']}")
    print()
    
    print("Characters Detected:")
    for char in result['domain_translation']['characters_detected']:
        print(f"  • {char['name']} ({char['role']})")
    print()
    
    print("Relevant Truths:")
    for truth in result['domain_translation']['relevant_truths']:
        print(f"  • {truth}")
    print()
    
    if result['insights']['cosmic_truth']:
        print(f"Universal Truth: {result['insights']['cosmic_truth']}")
        print()
    
    # Test auto-detection
    print("=" * 80)
    print("🔍 AUTO-DETECTION TEST")
    print()
    
    test_texts = [
        ("Our startup pivoted three times before finding product-market fit.", "business"),
        ("The experiment failed, but the anomaly revealed a new theory.", "science"),
        ("In therapy, I finally faced the trauma I'd buried for years.", "personal"),
        ("The diagnosis changed everything, but healing taught me to live.", "medical"),
    ]
    
    for text, expected in test_texts:
        result = await adapter.analyze_with_domain(text, auto_detect=True)
        detected = result['domain']
        status = "✅" if detected == expected else "❌"
        print(f"{status} Expected: {expected:10} | Detected: {detected:10} | '{text[:50]}...'")
    
    print()
    print("=" * 80)


async def demonstrate_all():
    """Run all demonstrations."""
    await demonstrate_cross_domain()
    await demonstrate_plugin_system()


if __name__ == "__main__":
    asyncio.run(demonstrate_all())
