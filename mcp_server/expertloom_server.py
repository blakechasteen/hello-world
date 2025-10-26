#!/usr/bin/env python3
"""
ExpertLoom MCP Server
=====================

Model Context Protocol server exposing ExpertLoom tools to Claude Desktop.

Tools:
- summarize_text: Summarize text while preserving entities/measurements
- extract_entities: Extract entities and measurements from text
- store_note: Store note in Qdrant with full pipeline
- search_notes: Search stored notes
- install_domain: Install a domain from registry

Usage:
    This server is launched by Claude Desktop via stdio.
    Configure in claude_desktop_config.json
"""

import asyncio
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print("ERROR: mcp package not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# ExpertLoom imports
from mythRL_core.entity_resolution import EntityRegistry, EntityResolver
from mythRL_core.entity_resolution.extractor import EntityExtractor
from mythRL_core.summarization import TextSummarizer, SummarizerConfig, SummarizationStrategy


# Initialize server
app = Server("expertloom")

# Global state (loaded on startup)
DOMAINS = {}  # domain_name -> (registry, resolver, extractor)
CURRENT_DOMAIN = None


def load_domain(domain_name: str) -> Dict[str, Any]:
    """Load a domain's registry, resolver, and extractor."""
    registry_path = Path(__file__).parent.parent / "mythRL_core" / "domains" / domain_name / "registry.json"

    if not registry_path.exists():
        raise FileNotFoundError(f"Domain not found: {domain_name}")

    registry = EntityRegistry.load(registry_path)
    resolver = EntityResolver(registry)
    extractor = EntityExtractor(resolver, custom_patterns=registry.measurement_patterns)

    return {
        "registry": registry,
        "resolver": resolver,
        "extractor": extractor
    }


def get_current_domain():
    """Get currently active domain components."""
    global CURRENT_DOMAIN, DOMAINS

    if CURRENT_DOMAIN is None:
        # Default to automotive if available
        if "automotive" not in DOMAINS:
            DOMAINS["automotive"] = load_domain("automotive")
        CURRENT_DOMAIN = "automotive"

    return DOMAINS[CURRENT_DOMAIN]


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available ExpertLoom tools."""
    return [
        Tool(
            name="summarize_text",
            description="""Summarize text while preserving entities and measurements.

Returns a concise summary (2-3 sentences) that preserves key information like:
- Entity mentions (vehicles, components, etc.)
- Measurements (tire pressure, mileage, etc.)
- Important actions and observations

Perfect for:
- Creating timeline views
- Quick scanning of notes
- Mobile displays
- Search result previews
""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to summarize"
                    },
                    "max_sentences": {
                        "type": "integer",
                        "description": "Maximum sentences in summary (default: 3)",
                        "default": 3
                    },
                    "domain": {
                        "type": "string",
                        "description": "Domain to use for entity extraction (default: automotive)",
                        "default": "automotive"
                    }
                },
                "required": ["text"]
            }
        ),

        Tool(
            name="extract_entities",
            description="""Extract entities and measurements from text using domain knowledge.

Detects and extracts:
- Entities: Vehicles, components, fluids, tools, etc. (domain-specific)
- Measurements: Numeric values (PSI, mm, quarts, miles, etc.)
- Categorical values: Conditions (dirty, clean, good, worn, etc.)
- Timestamps: When observations were made

Returns structured data ready for storage or analysis.
""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to analyze"
                    },
                    "domain": {
                        "type": "string",
                        "description": "Domain to use (automotive, beekeeping, etc.)",
                        "default": "automotive"
                    }
                },
                "required": ["text"]
            }
        ),

        Tool(
            name="process_note",
            description="""Complete pipeline: extract entities, measurements, and generate summary.

This is the main ExpertLoom processing tool. It:
1. Extracts entities (e.g., "Corolla" → vehicle-corolla-2015)
2. Extracts measurements (e.g., "28 PSI" → tire_pressure_psi: 28.0)
3. Generates intelligent summary preserving key info
4. Returns complete structured data

Use this when you want to capture a note with full intelligence.
""",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The note text to process"
                    },
                    "domain": {
                        "type": "string",
                        "description": "Domain (automotive, beekeeping, etc.)",
                        "default": "automotive"
                    },
                    "summarize": {
                        "type": "boolean",
                        "description": "Generate summary (default: true)",
                        "default": True
                    }
                },
                "required": ["text"]
            }
        ),

        Tool(
            name="list_domains",
            description="""List all available ExpertLoom domains.

Shows installed domains with statistics:
- Total entities
- Total aliases
- Entity types
- Measurement patterns
- Validation status
""",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),

        Tool(
            name="switch_domain",
            description="""Switch to a different domain.

Changes the active domain for subsequent operations.
Available domains: automotive, beekeeping, (more can be added)
""",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Domain name to switch to"
                    }
                },
                "required": ["domain"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls from Claude."""

    try:
        if name == "summarize_text":
            return await tool_summarize_text(arguments)
        elif name == "extract_entities":
            return await tool_extract_entities(arguments)
        elif name == "process_note":
            return await tool_process_note(arguments)
        elif name == "list_domains":
            return await tool_list_domains(arguments)
        elif name == "switch_domain":
            return await tool_switch_domain(arguments)
        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


async def tool_summarize_text(args: Dict[str, Any]) -> list[TextContent]:
    """Summarize text while preserving key information."""
    text = args["text"]
    max_sentences = args.get("max_sentences", 3)
    domain_name = args.get("domain", "automotive")

    # Load domain if needed
    if domain_name not in DOMAINS:
        DOMAINS[domain_name] = load_domain(domain_name)

    domain = DOMAINS[domain_name]
    extractor = domain["extractor"]

    # Extract entities and measurements first
    extracted = extractor.extract(text)

    # Create summarizer
    summarizer = TextSummarizer(SummarizerConfig(
        strategy=SummarizationStrategy.EXTRACTIVE,
        max_sentences=max_sentences,
        preserve_entities=True,
        preserve_measurements=True
    ))

    # Generate summary
    result = summarizer.summarize(
        text,
        entities=[e['canonical_id'] for e in extracted.entities],
        measurements=extracted.measurements
    )

    # Format response
    response = f"""**Summary** ({result['compression']:.0%} of original)

{result['summary']}

**Details:**
- Original: {result['original_length']} chars
- Summary: {result['summary_length']} chars
- Entities preserved: {len(result['preserved_entities'])}/{len(extracted.entities)}
- Measurements preserved: {len(result['preserved_measurements'])}/{len(extracted.measurements or {})}
"""

    if result['preserved_entities']:
        response += "\n**Preserved Entities:**\n"
        for entity_id in result['preserved_entities']:
            response += f"- {entity_id}\n"

    if result['preserved_measurements']:
        response += "\n**Preserved Measurements:**\n"
        for key, value in result['preserved_measurements'].items():
            response += f"- {key}: {value}\n"

    return [TextContent(type="text", text=response)]


async def tool_extract_entities(args: Dict[str, Any]) -> list[TextContent]:
    """Extract entities and measurements from text."""
    text = args["text"]
    domain_name = args.get("domain", "automotive")

    # Load domain if needed
    if domain_name not in DOMAINS:
        DOMAINS[domain_name] = load_domain(domain_name)

    domain = DOMAINS[domain_name]
    extractor = domain["extractor"]

    # Extract
    extracted = extractor.extract(text)

    # Format response
    response = f"""**Extraction Results**

**Entities Found:** {len(extracted.entities)}
"""

    for entity in extracted.entities:
        response += f"- {entity['matched_text']} → {entity['canonical_id']} ({entity['entity_type']})\n"

    if extracted.measurements:
        response += f"\n**Measurements Found:** {len(extracted.measurements)}\n"
        for key, value in extracted.measurements.items():
            response += f"- {key}: {value}\n"
    else:
        response += "\n**Measurements Found:** 0\n"

    response += f"\n**Timestamp:** {extracted.timestamp.isoformat()}\n"

    return [TextContent(type="text", text=response)]


async def tool_process_note(args: Dict[str, Any]) -> list[TextContent]:
    """Complete processing pipeline."""
    text = args["text"]
    domain_name = args.get("domain", "automotive")
    do_summarize = args.get("summarize", True)

    # Load domain if needed
    if domain_name not in DOMAINS:
        DOMAINS[domain_name] = load_domain(domain_name)

    domain = DOMAINS[domain_name]
    extractor = domain["extractor"]

    # Extract entities and measurements
    extracted = extractor.extract(text)

    # Build response
    response = f"""**Note Processed Successfully**

**Domain:** {domain_name}

**Entities:** {len(extracted.entities)}
"""
    for entity in extracted.entities:
        response += f"- {entity['matched_text']} → {entity['canonical_id']}\n"

    if extracted.measurements:
        response += f"\n**Measurements:** {len(extracted.measurements)}\n"
        for key, value in extracted.measurements.items():
            response += f"- {key}: {value}\n"

    # Generate summary if requested
    if do_summarize:
        summarizer = TextSummarizer(SummarizerConfig(
            strategy=SummarizationStrategy.EXTRACTIVE,
            max_sentences=2,
            preserve_entities=True,
            preserve_measurements=True
        ))

        summary_result = summarizer.summarize(
            text,
            entities=[e['canonical_id'] for e in extracted.entities],
            measurements=extracted.measurements
        )

        response += f"\n**Summary:** ({summary_result['compression']:.0%} compression)\n"
        response += f"{summary_result['summary']}\n"

    # Show payload structure
    payload = extracted.to_qdrant_payload()
    response += f"\n**Qdrant Payload:**\n"
    response += f"- Primary entity: {payload.get('primary_entity_id', 'none')}\n"
    response += f"- Entity IDs: {payload['entity_ids']}\n"
    response += f"- Has measurements: {payload['has_measurements']}\n"

    response += f"\n**Ready for Qdrant storage!**\n"

    return [TextContent(type="text", text=response)]


async def tool_list_domains(args: Dict[str, Any]) -> list[TextContent]:
    """List available domains."""
    domains_dir = Path(__file__).parent.parent / "mythRL_core" / "domains"

    if not domains_dir.exists():
        return [TextContent(type="text", text="Domains directory not found")]

    # Find all domains
    available_domains = []
    for item in domains_dir.iterdir():
        if item.is_dir() and (item / "registry.json").exists():
            domain_name = item.name
            if domain_name == "DOMAIN_TEMPLATE":
                continue

            # Load stats
            try:
                if domain_name not in DOMAINS:
                    DOMAINS[domain_name] = load_domain(domain_name)
                registry = DOMAINS[domain_name]["registry"]
                stats = registry.stats()
                available_domains.append((domain_name, stats))
            except Exception as e:
                available_domains.append((domain_name, {"error": str(e)}))

    # Format response
    response = f"**Available Domains:** {len(available_domains)}\n\n"

    for domain_name, stats in available_domains:
        if "error" in stats:
            response += f"- **{domain_name}** (error loading)\n"
            continue

        active = " (ACTIVE)" if domain_name == CURRENT_DOMAIN else ""
        response += f"- **{domain_name}**{active}\n"
        response += f"  - Entities: {stats['total_entities']}\n"
        response += f"  - Aliases: {stats['total_aliases']}\n"
        response += f"  - Types: {', '.join(stats['entities_by_type'].keys())}\n"
        response += "\n"

    return [TextContent(type="text", text=response)]


async def tool_switch_domain(args: Dict[str, Any]) -> list[TextContent]:
    """Switch active domain."""
    global CURRENT_DOMAIN

    domain_name = args["domain"]

    # Load domain if needed
    try:
        if domain_name not in DOMAINS:
            DOMAINS[domain_name] = load_domain(domain_name)

        CURRENT_DOMAIN = domain_name
        stats = DOMAINS[domain_name]["registry"].stats()

        response = f"""**Domain switched to: {domain_name}**

**Stats:**
- Entities: {stats['total_entities']}
- Aliases: {stats['total_aliases']}
- Entity types: {', '.join(stats['entities_by_type'].keys())}
"""

        return [TextContent(type="text", text=response)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error switching domain: {str(e)}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
