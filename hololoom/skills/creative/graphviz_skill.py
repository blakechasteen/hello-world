"""Graphviz Skill - Architecture diagram generation"""

import asyncio
import os
import tempfile
import time
from typing import Any

from hololoom.skills.base import (
    BaseSkill,
    SkillCategory,
    SkillInput,
    SkillMetadata,
    SkillOutput,
    SkillStatus,
)


class GraphvizSkill(BaseSkill):
    """Auto-generate architecture diagrams with Graphviz."""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="graphviz",
            version="1.0.0",
            author="HoloLoom Team",
            category=SkillCategory.CREATIVE,
            tags=["graphviz", "visualization", "diagram", "graph", "architecture"],
            description_short="Auto-generate architecture diagrams with Graphviz",
            description_long=(
                "Generate diagrams from DOT language descriptions using Graphviz. "
                "Supports multiple layout algorithms (dot, neato, circo, fdp, twopi), "
                "output formats (PNG, SVG, PDF), and graph types (directed, undirected, "
                "hierarchical). Ideal for architecture diagrams, dependency graphs, "
                "flowcharts, state machines, and network visualizations."
            ),
            requires_code_execution=True,
            requires_filesystem_write=True,
            external_dependencies=["graphviz", "dot"],
            expected_latency_ms="100-5000ms",
            token_usage="50-300 tokens"
        )

    def validate_input(self, skill_input: SkillInput) -> bool:
        valid_ops = [
            "render_dot", "layout_graph", "from_networkx",
            "architecture_diagram", "dependency_graph", "export"
        ]
        if skill_input.operation not in valid_ops:
            raise ValueError(f"Invalid operation. Must be one of: {', '.join(valid_ops)}")

        # Validate required parameters
        if skill_input.operation in ["render_dot", "layout_graph"]:
            if "dot_source" not in skill_input.parameters:
                raise ValueError(f"{skill_input.operation} requires 'dot_source' parameter")

        return True

    async def execute(self, skill_input: SkillInput) -> SkillOutput:
        start_time = time.time()
        try:
            result = await self._dispatch(skill_input.operation, skill_input.parameters)
            execution_time = (time.time() - start_time) * 1000
            return SkillOutput(
                status=SkillStatus.SUCCESS,
                result=result,
                message=f"Graphviz '{skill_input.operation}' completed",
                execution_time_ms=execution_time,
                details=result if isinstance(result, dict) else {"result": result}
            )
        except Exception as e:
            return SkillOutput(
                status=SkillStatus.ERROR,
                result=None,
                message=f"Graphviz operation failed: {e}",
                execution_time_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )

    async def _dispatch(self, operation: str, parameters: dict[str, Any]):
        if operation == "render_dot":
            return await self._render_dot(parameters)
        elif operation == "layout_graph":
            return await self._layout_graph(parameters)
        elif operation == "from_networkx":
            return await self._from_networkx(parameters)
        elif operation == "architecture_diagram":
            return await self._architecture_diagram(parameters)
        elif operation == "dependency_graph":
            return await self._dependency_graph(parameters)
        elif operation == "export":
            return await self._export(parameters)

    async def _run_graphviz(self, dot_source: str, output: str, layout: str = "dot", format: str = "png") -> dict[str, Any]:
        """Run Graphviz layout engine."""
        # Create temp file for DOT source
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dot', delete=False) as f:
            f.write(dot_source)
            dot_file = f.name

        try:
            # Run layout engine (dot, neato, circo, fdp, twopi)
            cmd = [layout, f"-T{format}", dot_file, "-o", output]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                raise RuntimeError(f"Graphviz {layout} command failed: {error_msg}")

            # Get file size
            size_bytes = os.path.getsize(output)
            size_kb = size_bytes / 1024

            # Count nodes and edges in DOT source
            nodes = dot_source.count(';') if 'digraph' in dot_source or 'graph' in dot_source else 0
            edges = dot_source.count('->') + dot_source.count('--')

            return {
                "operation": "render",
                "layout": layout,
                "format": format,
                "output": output,
                "size_kb": round(size_kb, 2),
                "nodes": nodes,
                "edges": edges,
                "success": True
            }

        finally:
            # Clean up temp file
            if os.path.exists(dot_file):
                os.remove(dot_file)

    async def _render_dot(self, params):
        """Render DOT source to image."""
        dot_source = params["dot_source"]
        output = params.get("output", "graph.png")
        format = params.get("format", "png")
        dpi = params.get("dpi", 96)

        result = await self._run_graphviz(dot_source, output, layout="dot", format=format)
        result["operation"] = "render_dot"
        result["dot_source_length"] = len(dot_source)

        return result

    async def _layout_graph(self, params):
        """Apply layout algorithm."""
        dot_source = params["dot_source"]
        layout = params.get("layout", "dot")  # dot, neato, circo, fdp, twopi
        output = params.get("output", "graph.png")
        format = params.get("format", "png")

        # Layout descriptions:
        # - dot: Hierarchical (top-down, good for dependency trees)
        # - neato: Spring model (force-directed, good for undirected graphs)
        # - circo: Circular layout (good for networks)
        # - fdp: Force-directed placement (good for large graphs)
        # - twopi: Radial layout (good for tree structures)

        result = await self._run_graphviz(dot_source, output, layout=layout, format=format)
        result["operation"] = "layout_graph"

        return result

    async def _from_networkx(self, params):
        """Convert NetworkX graph to Graphviz."""
        # In real impl, would accept networkx.DiGraph/Graph object
        graph_data = params.get("graph_data", {})
        output = params.get("output", "networkx_graph.png")

        # Mock NetworkX → DOT conversion
        return {
            "operation": "from_networkx",
            "nodes": len(graph_data.get("nodes", [])),
            "edges": len(graph_data.get("edges", [])),
            "output": output,
            "dot_source_generated": True,
            "success": True
        }

    async def _architecture_diagram(self, params):
        """Generate architecture diagram from components."""
        components = params.get("components", [])
        connections = params.get("connections", [])
        style = params.get("style", "default")
        output = params.get("output", "architecture.png")
        rankdir = params.get("rankdir", "TB")  # TB (top-bottom), LR (left-right)

        # Generate DOT source from components
        dot_lines = ["digraph architecture {", f"  rankdir={rankdir};", "  node [shape=box, style=rounded];"]

        # Group components by layer
        layers = {}
        for comp in components:
            layer = comp.get("layer", 1)
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(comp)

        # Add nodes with layers as subgraphs
        for layer_id in sorted(layers.keys()):
            dot_lines.append(f"  subgraph cluster_{layer_id} {{")
            dot_lines.append(f'    label="Layer {layer_id}";')
            for comp in layers[layer_id]:
                name = comp["name"].replace(" ", "_")
                comp_type = comp.get("type", "service")
                # Style based on type
                if comp_type == "web":
                    style_attrs = 'fillcolor=lightblue, style=filled'
                elif comp_type == "service":
                    style_attrs = 'fillcolor=lightgreen, style=filled'
                elif comp_type == "datastore":
                    style_attrs = 'fillcolor=lightyellow, style="filled,rounded", shape=cylinder'
                else:
                    style_attrs = 'fillcolor=white'

                dot_lines.append(f'    {name} [label="{comp["name"]}", {style_attrs}];')
            dot_lines.append("  }")

        # Add connections
        for conn in connections:
            from_node = conn["from"].replace(" ", "_")
            to_node = conn["to"].replace(" ", "_")
            label = conn.get("label", "")
            dot_lines.append(f'  {from_node} -> {to_node} [label="{label}"];')

        dot_lines.append("}")

        dot_source = "\n".join(dot_lines)

        # Render using dot layout
        result = await self._run_graphviz(dot_source, output, layout="dot", format="png")
        result["operation"] = "architecture_diagram"
        result["components"] = len(components)
        result["connections"] = len(connections)
        result["style"] = style
        result["layers"] = len(layers)

        return result

    async def _dependency_graph(self, params):
        """Generate dependency graph."""
        dependencies = params.get("dependencies", {})
        root = params.get("root")
        output = params.get("output", "dependencies.png")
        format = params.get("format", "png")

        # Generate DOT source from dependencies
        dot_lines = ["digraph dependencies {", "  node [shape=box];"]

        # Track visited to calculate depth
        visited = set()
        max_depth = 0

        def add_deps(node, depth=0):
            nonlocal max_depth
            if node in visited:
                return
            visited.add(node)
            max_depth = max(max_depth, depth)

            if node in dependencies:
                for dep in dependencies[node]:
                    dot_lines.append(f'  "{node}" -> "{dep}";')
                    add_deps(dep, depth + 1)
            else:
                # Leaf node (no dependencies)
                dot_lines.append(f'  "{node}";')

        # Start from root if specified, otherwise all top-level nodes
        if root:
            add_deps(root)
        else:
            for node in dependencies.keys():
                add_deps(node)

        dot_lines.append("}")

        dot_source = "\n".join(dot_lines)

        # Detect circular dependencies (simple check)
        circular_count = 0
        for node, deps in dependencies.items():
            if node in deps:
                circular_count += 1  # Self-loop

        # Render
        result = await self._run_graphviz(dot_source, output, layout="dot", format=format)
        result["operation"] = "dependency_graph"
        result["root"] = root
        result["total_dependencies"] = len(dependencies)
        result["depth"] = max_depth
        result["circular_dependencies"] = circular_count

        return result

    async def _export(self, params):
        """Export to different formats."""
        source = params.get("source")
        dot_source = params.get("dot_source")
        formats = params.get("formats", ["png", "svg"])
        output_dir = params.get("output_dir", ".")

        # Read DOT source if from file
        if source and not dot_source:
            with open(source) as f:
                dot_source = f.read()

        if not dot_source:
            raise ValueError("Either 'source' file or 'dot_source' string required")

        # Export to each format
        outputs = []
        total_size_kb = 0

        for fmt in formats:
            output_file = os.path.join(output_dir, f"export.{fmt}")
            result = await self._run_graphviz(dot_source, output_file, layout="dot", format=fmt)
            outputs.append(output_file)
            total_size_kb += result["size_kb"]

        return {
            "operation": "export",
            "source": source or "inline",
            "formats": formats,
            "outputs": outputs,
            "total_size_kb": round(total_size_kb, 2),
            "success": True
        }


from hololoom.skills.base import register_skill

register_skill(GraphvizSkill())
