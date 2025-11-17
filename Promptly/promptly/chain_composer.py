#!/usr/bin/env python3
"""
Chain Composer - Combine and compose multiple chains

Provides capabilities for:
- Combining multiple chains into mega-chains
- Sharing context between chains
- Conditional chain execution
- Chain versioning and rollback
- Template chains with parameters
- Chain marketplace format
"""

import json
import copy
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ChainMetadata:
    """Metadata for a chain"""
    name: str
    version: str
    description: str
    author: Optional[str] = None
    created_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


class ChainComposer:
    """
    Compose and combine multiple chains.

    Features:
    - Sequential composition (chain A → chain B)
    - Parallel composition (chain A || chain B)
    - Conditional composition (if X then chain A else chain B)
    - Context sharing between chains
    - Chain versioning
    """

    def __init__(self):
        self.chains: Dict[str, Dict[str, Any]] = {}
        self.composed_chains: Dict[str, Dict[str, Any]] = {}

    def register_chain(self, chain_def: Dict[str, Any]) -> None:
        """Register a chain for composition"""
        chain_name = chain_def.get("name", "unnamed")
        self.chains[chain_name] = chain_def

    def compose_sequential(
        self,
        chain_names: List[str],
        name: str,
        share_context: bool = True
    ) -> Dict[str, Any]:
        """
        Compose chains sequentially (A → B → C).

        Args:
            chain_names: List of chain names in order
            name: Name for composed chain
            share_context: Whether to share variables between chains

        Returns:
            Composed chain definition
        """
        if not chain_names:
            raise ValueError("Must provide at least one chain")

        # Validate all chains exist
        for chain_name in chain_names:
            if chain_name not in self.chains:
                raise ValueError(f"Chain '{chain_name}' not found")

        composed_steps = []
        previous_chain_outputs = []

        for i, chain_name in enumerate(chain_names):
            chain = self.chains[chain_name]
            chain_steps = chain.get("steps", [])

            # Prefix step names to avoid conflicts
            for step in chain_steps:
                step_copy = copy.deepcopy(step)

                # Rename step
                original_name = step_copy.get("name", f"step_{i}")
                new_name = f"{chain_name}_{original_name}"
                step_copy["name"] = new_name

                # Update dependencies
                if "depends_on" in step_copy:
                    updated_deps = []
                    for dep in step_copy["depends_on"]:
                        # Check if dependency is from previous chain
                        if share_context and i > 0 and dep in previous_chain_outputs:
                            updated_deps.append(f"{chain_names[i-1]}_{dep}")
                        else:
                            updated_deps.append(f"{chain_name}_{dep}")
                    step_copy["depends_on"] = updated_deps
                elif i > 0 and share_context:
                    # First step of chain depends on last step of previous chain
                    if previous_chain_outputs:
                        step_copy["depends_on"] = [previous_chain_outputs[-1]]

                composed_steps.append(step_copy)

                # Track outputs
                previous_chain_outputs.append(new_name)

        # Create composed chain
        composed = {
            "name": name,
            "description": f"Sequential composition of: {', '.join(chain_names)}",
            "version": "1.0",
            "composed_from": chain_names,
            "composition_type": "sequential",
            "variables": self._merge_variables([self.chains[c] for c in chain_names]),
            "steps": composed_steps
        }

        self.composed_chains[name] = composed
        return composed

    def compose_parallel(
        self,
        chain_names: List[str],
        name: str,
        aggregation: str = "all"
    ) -> Dict[str, Any]:
        """
        Compose chains in parallel (A || B || C).

        Args:
            chain_names: List of chain names
            name: Name for composed chain
            aggregation: How to aggregate results (all, any, first)

        Returns:
            Composed chain definition
        """
        if not chain_names:
            raise ValueError("Must provide at least one chain")

        # Create parallel processor wrapper
        parallel_step = {
            "name": "parallel_chains",
            "type": "parallel",
            "config": {
                "tasks": [
                    {
                        "name": chain_name,
                        "chain": chain_name
                    }
                    for chain_name in chain_names
                ],
                "aggregation": aggregation,
                "error_strategy": "best_effort"
            }
        }

        composed = {
            "name": name,
            "description": f"Parallel composition of: {', '.join(chain_names)}",
            "version": "1.0",
            "composed_from": chain_names,
            "composition_type": "parallel",
            "variables": self._merge_variables([self.chains[c] for c in chain_names]),
            "steps": [parallel_step]
        }

        self.composed_chains[name] = composed
        return composed

    def compose_conditional(
        self,
        condition_chain: str,
        true_chain: str,
        false_chain: str,
        name: str,
        condition_field: str = "result"
    ) -> Dict[str, Any]:
        """
        Compose chains conditionally (if X then A else B).

        Args:
            condition_chain: Chain to evaluate condition
            true_chain: Chain to run if condition is true
            false_chain: Chain to run if condition is false
            name: Name for composed chain
            condition_field: Field to check in condition chain output

        Returns:
            Composed chain definition
        """
        # Validate chains exist
        for chain_name in [condition_chain, true_chain, false_chain]:
            if chain_name not in self.chains:
                raise ValueError(f"Chain '{chain_name}' not found")

        composed = {
            "name": name,
            "description": f"Conditional: if {condition_chain} then {true_chain} else {false_chain}",
            "version": "1.0",
            "composed_from": [condition_chain, true_chain, false_chain],
            "composition_type": "conditional",
            "variables": {},
            "steps": [
                # Run condition chain
                {
                    "name": "evaluate_condition",
                    "type": "chain",
                    "config": {"chain": condition_chain}
                },
                # Conditional branching
                {
                    "name": "conditional_execution",
                    "type": "conditional",
                    "depends_on": ["evaluate_condition"],
                    "config": {
                        "conditions": [
                            {
                                "type": "simple",
                                "field": condition_field,
                                "operator": "eq",
                                "value": True,
                                "action": {
                                    "type": "chain",
                                    "chain": true_chain
                                }
                            }
                        ],
                        "default": {
                            "type": "chain",
                            "chain": false_chain
                        }
                    }
                }
            ]
        }

        self.composed_chains[name] = composed
        return composed

    def create_template_chain(
        self,
        base_chain: str,
        name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a parameterized template from a chain.

        Args:
            base_chain: Base chain name
            name: Template name
            parameters: Template parameters

        Returns:
            Template chain definition
        """
        if base_chain not in self.chains:
            raise ValueError(f"Chain '{base_chain}' not found")

        chain = copy.deepcopy(self.chains[base_chain])

        # Add template parameters
        chain["name"] = name
        chain["template_base"] = base_chain
        chain["parameters"] = parameters
        chain["is_template"] = True

        self.composed_chains[name] = chain
        return chain

    def instantiate_template(
        self,
        template_name: str,
        instance_name: str,
        parameter_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Instantiate a template chain with specific parameter values.

        Args:
            template_name: Template chain name
            instance_name: Instance name
            parameter_values: Parameter values

        Returns:
            Instantiated chain
        """
        if template_name not in self.composed_chains:
            raise ValueError(f"Template '{template_name}' not found")

        template = self.composed_chains[template_name]
        if not template.get("is_template", False):
            raise ValueError(f"'{template_name}' is not a template")

        # Deep copy and substitute parameters
        instance = copy.deepcopy(template)
        instance["name"] = instance_name
        instance["instantiated_from"] = template_name
        instance["parameter_values"] = parameter_values
        del instance["is_template"]

        # Substitute parameters in chain definition
        instance_str = json.dumps(instance)
        for param, value in parameter_values.items():
            placeholder = f"{{{param}}}"
            instance_str = instance_str.replace(placeholder, str(value))

        instance = json.loads(instance_str)

        return instance

    def _merge_variables(self, chains: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge variables from multiple chains"""
        merged = {}
        for chain in chains:
            variables = chain.get("variables", {})
            for key, value in variables.items():
                if key in merged and merged[key] != value:
                    # Conflict: prefix with chain name
                    chain_name = chain.get("name", "unnamed")
                    merged[f"{chain_name}_{key}"] = value
                else:
                    merged[key] = value
        return merged

    def export_marketplace_format(self, chain_name: str) -> str:
        """
        Export chain in marketplace format (shareable JSON).

        Args:
            chain_name: Name of chain to export

        Returns:
            JSON string in marketplace format
        """
        if chain_name not in self.composed_chains:
            if chain_name not in self.chains:
                raise ValueError(f"Chain '{chain_name}' not found")
            chain = self.chains[chain_name]
        else:
            chain = self.composed_chains[chain_name]

        marketplace_format = {
            "format_version": "1.0",
            "chain": chain,
            "metadata": {
                "name": chain.get("name"),
                "version": chain.get("version", "1.0"),
                "description": chain.get("description", ""),
                "author": chain.get("author"),
                "created_at": datetime.now().isoformat(),
                "tags": chain.get("tags", []),
                "dependencies": chain.get("dependencies", [])
            },
            "examples": chain.get("examples", []),
            "documentation": chain.get("documentation", "")
        }

        return json.dumps(marketplace_format, indent=2)

    def import_from_marketplace(self, marketplace_json: str) -> str:
        """
        Import chain from marketplace format.

        Args:
            marketplace_json: JSON string in marketplace format

        Returns:
            Imported chain name
        """
        data = json.loads(marketplace_json)

        if data.get("format_version") != "1.0":
            raise ValueError("Unsupported marketplace format version")

        chain = data["chain"]
        chain_name = chain.get("name", "imported_chain")

        # Add metadata
        metadata = data.get("metadata", {})
        chain.update({
            "author": metadata.get("author"),
            "tags": metadata.get("tags", []),
            "dependencies": metadata.get("dependencies", []),
            "documentation": data.get("documentation", "")
        })

        self.chains[chain_name] = chain
        return chain_name

    def list_compositions(self) -> List[Dict[str, Any]]:
        """List all composed chains"""
        return [
            {
                "name": name,
                "type": chain.get("composition_type", "unknown"),
                "composed_from": chain.get("composed_from", []),
                "is_template": chain.get("is_template", False)
            }
            for name, chain in self.composed_chains.items()
        ]


__all__ = ['ChainComposer', 'ChainMetadata']
