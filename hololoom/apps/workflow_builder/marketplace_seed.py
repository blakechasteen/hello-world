"""
Marketplace Seed Data and Migration Script.

Seeds the marketplace with:
- Default category hierarchy
- Marketplace metadata for all 13 existing templates
- Initial featured templates

Usage:
    python -m HoloLoom.apps.workflow_builder.marketplace_seed [--db-path ./workflows.db]
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import uuid

# Default categories hierarchy
DEFAULT_CATEGORIES = [
    {
        "id": "research",
        "name": "Research",
        "parent_id": None,
        "description": "Multi-query research and exploration workflows",
        "icon": "search",
        "display_order": 1
    },
    {
        "id": "safety",
        "name": "Safety & Alignment",
        "parent_id": None,
        "description": "Safety-gated and alignment-aware workflows",
        "icon": "shield",
        "display_order": 2
    },
    {
        "id": "crm",
        "name": "CRM & Sales",
        "parent_id": None,
        "description": "Customer relationship management and sales automation",
        "icon": "users",
        "display_order": 3
    },
    {
        "id": "llm",
        "name": "LLM Applications",
        "parent_id": None,
        "description": "Content generation and customer support workflows",
        "icon": "message-square",
        "display_order": 4
    },
    {
        "id": "advanced",
        "name": "Advanced",
        "parent_id": None,
        "description": "Complex multi-agent and self-improving workflows",
        "icon": "cpu",
        "display_order": 5
    },
    {
        "id": "data",
        "name": "Data & Integration",
        "parent_id": None,
        "description": "ETL pipelines, data processing, and integration workflows",
        "icon": "database",
        "display_order": 6
    },
    {
        "id": "testing",
        "name": "Testing & Validation",
        "parent_id": None,
        "description": "Benchmarking, evaluation, and quality assurance workflows",
        "icon": "check-circle",
        "display_order": 7
    },
    {
        "id": "monitoring",
        "name": "Real-time & Monitoring",
        "parent_id": None,
        "description": "Event processing, alerting, and observability workflows",
        "icon": "activity",
        "display_order": 8
    }
]

# Template marketplace metadata
# Maps template file paths to marketplace info
TEMPLATE_MARKETPLACE_DATA = {
    # Research templates
    "research_pipeline.json": {
        "category": "research",
        "difficulty": "intermediate",
        "author": "HoloLoom Team",
        "license": "MIT",
        "is_featured": True,
        "tags": ["research", "multi-query", "synthesis", "refinement"],
        "description": "Multi-query research with synthesis and elegance refinement",
        "use_cases": [
            "Deep research on complex topics",
            "Literature review synthesis",
            "Comprehensive answer generation"
        ],
        "prerequisites": ["HoloLoom memory system with content"],
        "estimated_tokens": 800
    },

    # Safety templates
    "safety_gated_query.json": {
        "category": "safety",
        "difficulty": "intermediate",
        "author": "HoloLoom Team",
        "license": "MIT",
        "is_featured": True,
        "tags": ["safety", "guardrails", "conditional", "verification"],
        "description": "Safety-gated query with confidence-based routing",
        "use_cases": [
            "Production deployments requiring safety checks",
            "Enterprise AI with risk management",
            "Compliance-aware AI workflows"
        ],
        "prerequisites": ["SafetyGuardrails configured"],
        "estimated_tokens": 400
    },

    # Business/Sales templates
    "bdr_outbound_sequence.json": {
        "category": "crm",
        "difficulty": "intermediate",
        "author": "HoloLoom Team",
        "license": "MIT",
        "is_featured": False,
        "tags": ["sales", "outbound", "bdr", "sequences", "automation"],
        "description": "BDR outbound sequence automation workflow",
        "use_cases": [
            "Automated sales outreach",
            "Lead engagement sequences",
            "Sales development workflows"
        ],
        "prerequisites": ["CRM integration", "Email templates"],
        "estimated_tokens": 500
    },

    # CRM templates
    "crm/daily_action_list.json": {
        "category": "crm",
        "difficulty": "beginner",
        "author": "HoloLoom CRM",
        "license": "MIT",
        "is_featured": False,
        "tags": ["crm", "tasks", "daily", "prioritization"],
        "description": "Generate prioritized daily action list from CRM data",
        "use_cases": [
            "Sales rep daily planning",
            "Account manager task prioritization",
            "CRM-driven productivity"
        ],
        "prerequisites": ["CRM data in HoloLoom memory"],
        "estimated_tokens": 350
    },
    "crm/lead_scoring_simple.json": {
        "category": "crm",
        "difficulty": "beginner",
        "author": "HoloLoom CRM",
        "license": "MIT",
        "is_featured": True,
        "tags": ["crm", "lead_scoring", "automation", "bayesian"],
        "description": "Simple lead scoring with Bayesian blend decision engine",
        "use_cases": [
            "Lead qualification automation",
            "Sales prioritization",
            "Marketing qualified lead detection"
        ],
        "prerequisites": ["Contact and activity data"],
        "estimated_tokens": 450
    },
    "crm/multi_factor_scoring.json": {
        "category": "crm",
        "difficulty": "intermediate",
        "author": "HoloLoom CRM",
        "license": "MIT",
        "is_featured": False,
        "tags": ["crm", "scoring", "multi-factor", "parallel"],
        "description": "Multi-factor scoring with parallel signal extraction",
        "use_cases": [
            "Complex lead qualification",
            "Account health scoring",
            "Churn prediction"
        ],
        "prerequisites": ["Multiple data sources in memory"],
        "estimated_tokens": 600
    },

    # LLM templates
    "llm/content_creation.json": {
        "category": "llm",
        "difficulty": "beginner",
        "author": "HoloLoom Team",
        "license": "MIT",
        "is_featured": False,
        "tags": ["content", "generation", "marketing", "writing"],
        "description": "Content creation workflow with refinement",
        "use_cases": [
            "Blog post generation",
            "Marketing copy creation",
            "Documentation writing"
        ],
        "prerequisites": ["LLM backend configured"],
        "estimated_tokens": 400
    },
    "llm/customer_support_triage.json": {
        "category": "llm",
        "difficulty": "intermediate",
        "author": "HoloLoom Team",
        "license": "MIT",
        "is_featured": False,
        "tags": ["support", "triage", "classification", "routing"],
        "description": "Customer support ticket triage and routing",
        "use_cases": [
            "Support ticket classification",
            "Priority assignment",
            "Agent routing"
        ],
        "prerequisites": ["Support knowledge base"],
        "estimated_tokens": 500
    },

    # Advanced templates
    "advanced/self_healing_research.json": {
        "category": "advanced",
        "difficulty": "advanced",
        "author": "HoloLoom Team",
        "license": "MIT",
        "is_featured": True,
        "tags": ["research", "adaptive", "thompson-sampling", "self-healing"],
        "description": "Self-healing research with Thompson Sampling strategy selection",
        "use_cases": [
            "Adaptive research pipelines",
            "Self-improving AI systems",
            "Complex query handling"
        ],
        "prerequisites": ["Full HoloLoom stack"],
        "estimated_tokens": 1000
    },
    "advanced/adversarial_verification.json": {
        "category": "advanced",
        "difficulty": "advanced",
        "author": "HoloLoom Team",
        "license": "MIT",
        "is_featured": False,
        "tags": ["verification", "adversarial", "red-team", "robustness"],
        "description": "Adversarial verification for response robustness",
        "use_cases": [
            "Response quality assurance",
            "Adversarial testing",
            "Hallucination detection"
        ],
        "prerequisites": ["Verification tools configured"],
        "estimated_tokens": 800
    },
    "advanced/recursive_confidence_loop.json": {
        "category": "advanced",
        "difficulty": "advanced",
        "author": "HoloLoom Team",
        "license": "MIT",
        "is_featured": False,
        "tags": ["recursive", "confidence", "loop", "iteration"],
        "description": "Recursive confidence improvement loop",
        "use_cases": [
            "High-stakes decision making",
            "Multi-pass quality improvement",
            "Confidence calibration"
        ],
        "prerequisites": ["Reflection buffer enabled"],
        "estimated_tokens": 700
    },
    "advanced/ensemble_decision_maker.json": {
        "category": "advanced",
        "difficulty": "advanced",
        "author": "HoloLoom Team",
        "license": "MIT",
        "is_featured": False,
        "tags": ["ensemble", "parallel", "voting", "consensus"],
        "description": "Ensemble decision making with multiple strategy voting",
        "use_cases": [
            "High-reliability decisions",
            "Multi-model consensus",
            "Risk-averse AI applications"
        ],
        "prerequisites": ["Multiple memory strategies"],
        "estimated_tokens": 900
    },
    "advanced/knowledge_graph_builder.json": {
        "category": "advanced",
        "difficulty": "advanced",
        "author": "HoloLoom Team",
        "license": "MIT",
        "is_featured": True,
        "tags": ["knowledge-graph", "extraction", "entities", "relationships"],
        "description": "Automated knowledge graph construction from documents",
        "use_cases": [
            "Domain knowledge extraction",
            "Document understanding",
            "Entity relationship mapping"
        ],
        "prerequisites": ["Document ingestion pipeline"],
        "estimated_tokens": 1200
    },

    # New comprehensive advanced workflows (December 2025)
    "advanced/research_council.json": {
        "category": "advanced",
        "difficulty": "advanced",
        "author": "HoloLoom Team",
        "license": "MIT",
        "is_featured": True,
        "tags": ["multi-agent", "debate", "research", "consensus", "collaborative"],
        "description": "Multi-agent research council with debate, consensus building, and minority opinion preservation",
        "use_cases": [
            "Complex research requiring multiple perspectives",
            "Fact-checking with adversarial verification",
            "Policy analysis with diverse viewpoints",
            "Academic research synthesis"
        ],
        "prerequisites": ["Full HoloLoom stack", "LLM backend"],
        "estimated_tokens": 1500
    },
    "advanced/document_intelligence.json": {
        "category": "advanced",
        "difficulty": "advanced",
        "author": "HoloLoom Team",
        "license": "MIT",
        "is_featured": True,
        "tags": ["document", "extraction", "summarization", "knowledge-graph", "nlp"],
        "description": "End-to-end document processing with entity extraction, summarization, Q&A generation, and knowledge graph construction",
        "use_cases": [
            "Document understanding and analysis",
            "Legal document processing",
            "Research paper analysis",
            "Knowledge base construction from documents"
        ],
        "prerequisites": ["Document ingestion pipeline", "Knowledge graph storage"],
        "estimated_tokens": 1800
    },
    "advanced/rag_enhanced_qa.json": {
        "category": "advanced",
        "difficulty": "advanced",
        "author": "HoloLoom Team",
        "license": "MIT",
        "is_featured": True,
        "tags": ["rag", "retrieval", "qa", "hybrid-search", "reranking", "citations"],
        "description": "Sophisticated RAG with multi-query expansion, hybrid search, cross-encoder reranking, and confidence-aware answering",
        "use_cases": [
            "Enterprise knowledge base Q&A",
            "Customer support automation",
            "Research assistant with citations",
            "Documentation search and synthesis"
        ],
        "prerequisites": ["Vector store", "Knowledge graph", "LLM backend"],
        "estimated_tokens": 1400
    },
    "advanced/strategic_goal_planner.json": {
        "category": "advanced",
        "difficulty": "advanced",
        "author": "HoloLoom Team",
        "license": "MIT",
        "is_featured": False,
        "tags": ["planning", "goals", "strategy", "resource-allocation", "risk-assessment", "optimization"],
        "description": "Hierarchical goal decomposition with constraint satisfaction, resource allocation, and Thompson Sampling optimization",
        "use_cases": [
            "Project planning and scheduling",
            "Strategic initiative planning",
            "Resource-constrained optimization",
            "Risk-aware decision making"
        ],
        "prerequisites": ["Full HoloLoom stack"],
        "estimated_tokens": 1600
    },
    "advanced/automated_code_review.json": {
        "category": "advanced",
        "difficulty": "advanced",
        "author": "HoloLoom Team",
        "license": "MIT",
        "is_featured": True,
        "tags": ["code-review", "static-analysis", "security", "quality", "automation", "ci-cd"],
        "description": "Comprehensive code review with static analysis, security scanning, style checking, and AI-powered fix suggestions",
        "use_cases": [
            "Pull request automation",
            "CI/CD code quality gates",
            "Security vulnerability detection",
            "Code style enforcement"
        ],
        "prerequisites": ["Code parsing tools", "Security scanning rules"],
        "estimated_tokens": 1700
    },

    # New workflow templates (December 2025 - Marketplace Expansion)
    "testing/batch_evaluation.json": {
        "category": "testing",
        "difficulty": "advanced",
        "author": "HoloLoom Team",
        "license": "MIT",
        "is_featured": True,
        "tags": ["testing", "evaluation", "benchmarking", "quality", "metrics", "golden-dataset", "statistical-analysis"],
        "description": "Comprehensive batch testing against golden datasets with parallel semantic/factual/completeness scoring and statistical analysis",
        "use_cases": [
            "AI response quality benchmarking",
            "Regression testing for AI systems",
            "Golden dataset evaluation",
            "Multi-dimensional quality scoring"
        ],
        "prerequisites": ["Golden dataset with expected answers", "LLM backend configured"],
        "estimated_tokens": 1400
    },
    "llm/conversational_agent.json": {
        "category": "llm",
        "difficulty": "advanced",
        "author": "HoloLoom Team",
        "license": "MIT",
        "is_featured": True,
        "tags": ["chatbot", "conversation", "multi-turn", "memory", "intent", "session", "dialogue"],
        "description": "Stateful multi-turn chatbot with session persistence, intent routing, context compression, and confidence-based fallback",
        "use_cases": [
            "Customer support chatbots",
            "Virtual assistants",
            "Interactive Q&A systems",
            "Conversational interfaces"
        ],
        "prerequisites": ["Session storage", "LLM backend"],
        "estimated_tokens": 1600
    },
    "data/etl_pipeline.json": {
        "category": "data",
        "difficulty": "advanced",
        "author": "HoloLoom Team",
        "license": "MIT",
        "is_featured": True,
        "tags": ["etl", "data-pipeline", "ingestion", "transformation", "quality", "lineage", "batch-processing"],
        "description": "Intelligent ETL with schema detection, quality gates, parallel transforms, error quarantine, and complete data lineage",
        "use_cases": [
            "Data warehouse ingestion",
            "Data quality enforcement",
            "ETL pipeline automation",
            "Data migration workflows"
        ],
        "prerequisites": ["Source and destination database connections"],
        "estimated_tokens": 1700
    },
    "advanced/adaptive_learning.json": {
        "category": "advanced",
        "difficulty": "advanced",
        "author": "HoloLoom Team",
        "license": "MIT",
        "is_featured": True,
        "tags": ["learning", "thompson-sampling", "feedback", "adaptive", "bayesian", "self-improving", "optimization"],
        "description": "Self-improving workflow with Thompson Sampling strategy selection, user feedback integration, and Bayesian prior updates",
        "use_cases": [
            "Adaptive AI systems",
            "Continuous learning pipelines",
            "A/B testing optimization",
            "Personalization engines"
        ],
        "prerequisites": ["Full HoloLoom stack", "Feedback collection mechanism"],
        "estimated_tokens": 1500
    },
    "monitoring/event_stream.json": {
        "category": "monitoring",
        "difficulty": "expert",
        "author": "HoloLoom Team",
        "license": "MIT",
        "is_featured": True,
        "tags": ["streaming", "events", "alerting", "monitoring", "anomaly-detection", "real-time", "observability"],
        "description": "Real-time event processing with sliding windows, parallel anomaly detection, alert deduplication, and multi-channel notifications",
        "use_cases": [
            "Infrastructure monitoring",
            "Application observability",
            "Anomaly detection pipelines",
            "Alert management systems"
        ],
        "prerequisites": ["Event stream source (Kafka/Kinesis/etc)", "Notification channels configured"],
        "estimated_tokens": 1800
    }
}


async def seed_categories(persistence) -> Dict[str, str]:
    """Seed default categories and return mapping of id -> id."""
    category_map = {}

    for cat_data in DEFAULT_CATEGORIES:
        from .workflow_persistence import TemplateCategoryRecord

        category = TemplateCategoryRecord(
            id=cat_data["id"],
            name=cat_data["name"],
            parent_id=cat_data["parent_id"],
            description=cat_data["description"],
            icon=cat_data["icon"],
            display_order=cat_data["display_order"]
        )

        await persistence.save_category(category)
        category_map[cat_data["id"]] = cat_data["id"]
        print(f"  Created category: {cat_data['name']}")

    return category_map


async def seed_templates(persistence, templates_dir: Path) -> int:
    """Seed marketplace templates from existing workflow files."""
    from .workflow_persistence import (
        WorkflowRecord,
        MarketplaceTemplateRecord
    )

    count = 0
    now = datetime.utcnow()

    for rel_path, meta in TEMPLATE_MARKETPLACE_DATA.items():
        file_path = templates_dir / rel_path

        if not file_path.exists():
            print(f"  Warning: Template not found: {rel_path}")
            continue

        # Load the workflow JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)

        # Generate IDs
        workflow_id = f"template-{rel_path.replace('/', '-').replace('.json', '')}"
        marketplace_id = f"mp-{workflow_id}"

        # Create workflow record
        # Extract nodes and connections from workflow data
        nodes = workflow_data.get("nodes", [])
        connections = workflow_data.get("connections", [])

        # Build metadata from remaining fields
        metadata = {
            k: v for k, v in workflow_data.items()
            if k not in ("nodes", "connections", "name", "description", "version")
        }
        metadata["version"] = workflow_data.get("version", "1.0")

        workflow = WorkflowRecord(
            id=workflow_id,
            name=workflow_data.get("name", rel_path),
            description=meta.get("description", workflow_data.get("description", "")),
            nodes=nodes,
            connections=connections,
            metadata=metadata,
            created_at=now,
            updated_at=now,
            owner_id="hololoom-team"
        )

        # Save workflow
        await persistence.save_workflow(workflow)

        # Create marketplace listing
        marketplace_template = MarketplaceTemplateRecord(
            id=marketplace_id,
            workflow_id=workflow_id,
            category=meta["category"],
            difficulty=meta["difficulty"],
            author=meta["author"],
            license=meta["license"],
            is_featured=meta["is_featured"],
            usage_count=0,
            avg_rating=0.0,
            rating_count=0,
            tags=meta["tags"],
            published_at=now,
            updated_at=now
        )

        # Save marketplace listing
        await persistence.save_marketplace_template(marketplace_template)

        status = "[FEATURED]" if meta["is_featured"] else ""
        print(f"  Added: {workflow_data.get('name', rel_path)} ({meta['category']}/{meta['difficulty']}) {status}")
        count += 1

    return count


async def run_seed(db_path: str = "./workflows.db"):
    """Run the complete seed process."""
    from .workflow_persistence import WorkflowPersistence

    print("=" * 60)
    print("HoloLoom Workflow Marketplace - Seed Script")
    print("=" * 60)
    print(f"\nDatabase: {db_path}")

    # Initialize persistence
    persistence = WorkflowPersistence(db_path)
    await persistence.initialize()

    # Determine templates directory
    script_dir = Path(__file__).parent
    templates_dir = script_dir / "example_workflows"

    if not templates_dir.exists():
        print(f"\nError: Templates directory not found: {templates_dir}")
        return

    print(f"Templates: {templates_dir}")
    print("\n" + "-" * 60)

    # Seed categories
    print("\n[1/2] Seeding categories...")
    await seed_categories(persistence)
    print(f"  -> {len(DEFAULT_CATEGORIES)} categories created")

    # Seed templates
    print("\n[2/2] Seeding marketplace templates...")
    count = await seed_templates(persistence, templates_dir)
    print(f"  -> {count} templates added to marketplace")

    # Summary
    print("\n" + "-" * 60)
    print("\nSeed complete!")

    # Show featured templates
    featured = await persistence.list_marketplace_templates(featured_only=True)
    print(f"\nFeatured templates ({len(featured)}):")
    for t in featured:
        workflow = await persistence.get_workflow(t.workflow_id)
        if workflow:
            print(f"  * {workflow.name} [{t.category}/{t.difficulty}]")

    print("\n" + "=" * 60)


def main():
    """CLI entry point."""
    import sys

    db_path = "./workflows.db"

    # Parse args
    if "--db-path" in sys.argv:
        idx = sys.argv.index("--db-path")
        if idx + 1 < len(sys.argv):
            db_path = sys.argv[idx + 1]

    if "-h" in sys.argv or "--help" in sys.argv:
        print(__doc__)
        return

    asyncio.run(run_seed(db_path))


if __name__ == "__main__":
    main()
