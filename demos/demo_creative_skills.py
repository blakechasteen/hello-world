"""
Demo: Creative Skills (LaTeX Compiler + Graphviz)
==================================================
Demonstrates the two newly implemented Tier 3 creative skills:
- LaTeX Compiler: Professional document generation
- Graphviz: Architecture diagram generation
"""

import asyncio
from HoloLoom.skills.base import SkillInput
from HoloLoom.skills.creative.latex_compiler import LaTeXCompilerSkill
from HoloLoom.skills.creative.graphviz_skill import GraphvizSkill


async def demo_latex_compiler():
    """Demonstrate LaTeX Compiler skill."""
    print("=" * 80)
    print("[1] DEMO: LaTeX Compiler Skill")
    print("=" * 80)
    print()

    skill = LaTeXCompilerSkill()

    # Demo 1: Compile PDF
    print("[1.1] Compile LaTeX to PDF")
    print("-" * 40)
    result = await skill.execute(SkillInput(
        operation="compile_pdf",
        parameters={
            "source": "thesis.tex",
            "output": "thesis.pdf",
            "passes": 2
        }
    ))
    print(f"Status: {result.status.value}")
    print(f"Message: {result.message}")
    print(f"Execution Time: {result.execution_time_ms:.1f}ms")
    if result.result:
        print(f"Pages: {result.result['pages']}")
        print(f"Size: {result.result['size_kb']} KB")
        print(f"Warnings: {result.result['warnings']}")
        print(f"Errors: {result.result['errors']}")
    print()

    # Demo 2: Compile with Bibliography
    print("[1.2] Compile with BibTeX")
    print("-" * 40)
    result = await skill.execute(SkillInput(
        operation="compile_with_bib",
        parameters={
            "source": "paper.tex",
            "bib_file": "references.bib",
            "output": "paper.pdf"
        }
    ))
    print(f"Status: {result.status.value}")
    print(f"Message: {result.message}")
    if result.result:
        print(f"Passes: {result.result['passes']} (pdflatex -> bibtex -> pdflatex x 2)")
        print(f"Citations: {result.result['citations']}")
        print(f"Pages: {result.result['pages']}")
    print()

    # Demo 3: From Template
    print("[1.3] Generate from Template")
    print("-" * 40)
    result = await skill.execute(SkillInput(
        operation="from_template",
        parameters={
            "template": "beamer",
            "title": "Machine Learning Presentation",
            "author": "HoloLoom Team",
            "variables": {"theme": "Madrid", "color": "blue"}
        }
    ))
    print(f"Status: {result.status.value}")
    print(f"Template: {result.result['template']}")
    print(f"Title: {result.result['title']}")
    print(f"Generated Source: {result.result['generated_source']}")
    print(f"Output: {result.result['output']}")
    print()

    # Demo 4: Diagnostics
    print("[1.4] LaTeX Diagnostics")
    print("-" * 40)
    result = await skill.execute(SkillInput(
        operation="diagnostics",
        parameters={"log_file": "document.log"}
    ))
    print(f"Status: {result.status.value}")
    if result.result:
        print(f"Errors: {result.result['error_count']}")
        for error in result.result['errors']:
            print(f"  Line {error['line']}: {error['message']}")
        print(f"Warnings: {result.result['warning_count']}")
        for warning in result.result['warnings']:
            print(f"  Line {warning['line']}: {warning['message']}")
    print()

    # Demo 5: Metadata
    print("[1.5] Skill Metadata")
    print("-" * 40)
    metadata = skill.get_metadata()
    print(f"Name: {metadata.name}")
    print(f"Version: {metadata.version}")
    print(f"Category: {metadata.category.value}")
    print(f"Tags: {', '.join(metadata.tags)}")
    print(f"Description: {metadata.description_short}")
    print(f"External Dependencies: {', '.join(metadata.external_dependencies)}")
    print()


async def demo_graphviz():
    """Demonstrate Graphviz skill."""
    print("=" * 80)
    print("[2] DEMO: Graphviz Skill")
    print("=" * 80)
    print()

    skill = GraphvizSkill()

    # Demo 1: Render DOT
    print("[2.1] Render DOT Source")
    print("-" * 40)
    dot_source = """
    digraph G {
        node [shape=box];
        A -> B;
        B -> C;
        B -> D;
        C -> E;
        D -> E;
    }
    """
    result = await skill.execute(SkillInput(
        operation="render_dot",
        parameters={
            "dot_source": dot_source,
            "output": "graph.png",
            "format": "png"
        }
    ))
    print(f"Status: {result.status.value}")
    print(f"Message: {result.message}")
    if result.result:
        print(f"Nodes: {result.result['nodes']}")
        print(f"Edges: {result.result['edges']}")
        print(f"Output: {result.result['output']}")
        print(f"Size: {result.result['size_kb']} KB")
    print()

    # Demo 2: Layout Algorithms
    print("[2.2] Apply Layout Algorithm")
    print("-" * 40)
    for layout in ["dot", "neato", "circo"]:
        result = await skill.execute(SkillInput(
            operation="layout_graph",
            parameters={
                "dot_source": dot_source,
                "layout": layout,
                "output": f"graph_{layout}.png"
            }
        ))
        if result.result:
            print(f"  {layout:8s}: {result.result['nodes']} nodes, "
                  f"{result.result['edges']} edges, "
                  f"{result.result['iterations']} iterations")
    print()

    # Demo 3: Architecture Diagram
    print("[2.3] Generate Architecture Diagram")
    print("-" * 40)
    result = await skill.execute(SkillInput(
        operation="architecture_diagram",
        parameters={
            "components": [
                {"name": "Frontend", "type": "web"},
                {"name": "Backend", "type": "api"},
                {"name": "Database", "type": "storage"}
            ],
            "connections": [
                {"from": "Frontend", "to": "Backend"},
                {"from": "Backend", "to": "Database"}
            ],
            "style": "clean",
            "output": "architecture.png"
        }
    ))
    print(f"Status: {result.status.value}")
    if result.result:
        print(f"Components: {result.result['components']}")
        print(f"Connections: {result.result['connections']}")
        print(f"Layers: {result.result['layers']}")
        print(f"Style: {result.result['style']}")
        print(f"Output: {result.result['output']}")
    print()

    # Demo 4: Dependency Graph
    print("[2.4] Generate Dependency Graph")
    print("-" * 40)
    result = await skill.execute(SkillInput(
        operation="dependency_graph",
        parameters={
            "dependencies": {
                "main": ["module_a", "module_b"],
                "module_a": ["utils"],
                "module_b": ["utils", "config"]
            },
            "root": "main",
            "output": "dependencies.png"
        }
    ))
    print(f"Status: {result.status.value}")
    if result.result:
        print(f"Root: {result.result['root']}")
        print(f"Total Dependencies: {result.result['total_dependencies']}")
        print(f"Depth: {result.result['depth']}")
        print(f"Circular Dependencies: {result.result['circular_dependencies']}")
    print()

    # Demo 5: Multi-Format Export
    print("[2.5] Export to Multiple Formats")
    print("-" * 40)
    result = await skill.execute(SkillInput(
        operation="export",
        parameters={
            "source": "graph.dot",
            "formats": ["png", "svg", "pdf"]
        }
    ))
    print(f"Status: {result.status.value}")
    if result.result:
        print(f"Source: {result.result['source']}")
        print(f"Formats: {', '.join(result.result['formats'])}")
        print(f"Outputs:")
        for output in result.result['outputs']:
            print(f"  - {output}")
    print()

    # Demo 6: Metadata
    print("[2.6] Skill Metadata")
    print("-" * 40)
    metadata = skill.get_metadata()
    print(f"Name: {metadata.name}")
    print(f"Version: {metadata.version}")
    print(f"Category: {metadata.category.value}")
    print(f"Tags: {', '.join(metadata.tags)}")
    print(f"Description: {metadata.description_short}")
    print(f"Expected Latency: {metadata.expected_latency_ms}")
    print()


async def main():
    """Run all creative skills demos."""
    print()
    print("=" * 80)
    print("Creative Skills Demonstration")
    print("=" * 80)
    print()
    print("Demonstrating 2 Tier 3 creative skills:")
    print("  1. LaTeX Compiler - Professional document generation")
    print("  2. Graphviz - Architecture diagram generation")
    print()

    await demo_latex_compiler()
    await demo_graphviz()

    print("=" * 80)
    print("[DONE] All Creative Skills Demos Completed Successfully!")
    print("=" * 80)
    print()
    print("Both skills are working correctly.")
    print("All operations validated.")
    print("Metadata complete.")
    print()


if __name__ == "__main__":
    asyncio.run(main())
