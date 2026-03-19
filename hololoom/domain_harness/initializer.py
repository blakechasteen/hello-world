"""
Domain Harness Initializer - Prompt to Domain Memory

Transforms a user's project description into structured domain memory:
- Features (backlog with dependencies)
- Domain state (constraints, environment)
- Test scaffolding

This is a one-time expansion agent - it creates the stage, 
then workers execute against it.

Weaving Metaphor:
The Initializer warps the loom - setting up all the threads
before any weaving (worker execution) begins.
"""

import re
from dataclasses import dataclass
from typing import Any

from hololoom.domain_harness.schema import DependsOn, DomainState, Feature, create_feature

# ============================================================================
# Initialization Result
# ============================================================================

@dataclass
class InitializationResult:
    """Result of domain initialization."""
    project_name: str
    features: list[Feature]
    dependencies: list[DependsOn]
    state: DomainState
    test_scaffolding: dict[str, str]  # filename -> content

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Project: {self.project_name}",
            f"Features: {len(self.features)}",
            f"Dependencies: {len(self.dependencies)}",
            f"Test files: {len(self.test_scaffolding)}",
            "",
            "Feature Backlog:"
        ]
        for f in self.features:
            deps = [d.to_feature_id for d in self.dependencies if d.from_feature_id == f.id]
            dep_str = f" (depends on: {', '.join(deps)})" if deps else ""
            lines.append(f"  [{f.id}] {f.name}{dep_str}")

        return "\n".join(lines)


# ============================================================================
# Feature Parser
# ============================================================================

class FeatureParser:
    """
    Parse user prompt into structured features.
    
    Supports multiple formats:
    - Bullet points
    - Numbered lists
    - Freeform descriptions
    - Dependency annotations
    """

    @staticmethod
    def parse(prompt: str) -> list[dict[str, Any]]:
        """
        Extract feature definitions from prompt.
        
        Returns list of dicts with:
        - name: Feature name
        - description: Detailed description
        - depends_on: List of feature names it depends on
        - priority: Inferred priority
        """
        features = []

        # Split into lines and identify feature markers
        lines = prompt.strip().split('\n')

        current_feature = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for bullet/numbered list item
            bullet_match = re.match(r'^[-*•]\s*(.+)$', line)
            numbered_match = re.match(r'^\d+[.)]\s*(.+)$', line)

            if bullet_match or numbered_match:
                # Save previous feature
                if current_feature:
                    features.append(current_feature)

                content = bullet_match.group(1) if bullet_match else numbered_match.group(1)

                # Check for dependency annotation: "X (depends on Y, Z)"
                dep_match = re.search(r'\(depends?\s+on:?\s*([^)]+)\)', content, re.I)
                depends_on = []
                if dep_match:
                    deps = dep_match.group(1)
                    depends_on = [d.strip() for d in deps.split(',')]
                    content = content[:dep_match.start()].strip()

                # Check for priority annotation: "[HIGH]", "[P1]", etc.
                priority = 0
                priority_match = re.search(r'\[(HIGH|CRITICAL|P[0-2])\]', content, re.I)
                if priority_match:
                    p = priority_match.group(1).upper()
                    priority = {'HIGH': 10, 'CRITICAL': 20, 'P0': 20, 'P1': 10, 'P2': 5}.get(p, 0)
                    content = content[:priority_match.start()].strip()

                current_feature = {
                    'name': content,
                    'description': content,
                    'depends_on': depends_on,
                    'priority': priority
                }

            elif current_feature:
                # Continuation of previous feature description
                current_feature['description'] += ' ' + line

        # Don't forget the last feature
        if current_feature:
            features.append(current_feature)

        # If no structured features found, treat whole prompt as one feature
        if not features:
            features.append({
                'name': prompt[:50] + ('...' if len(prompt) > 50 else ''),
                'description': prompt,
                'depends_on': [],
                'priority': 0
            })

        return features


# ============================================================================
# Dependency Resolver
# ============================================================================

class DependencyResolver:
    """Resolve named dependencies to feature IDs."""

    @staticmethod
    def resolve(
        features: list[Feature],
        parsed_deps: list[dict[str, Any]]
    ) -> list[DependsOn]:
        """
        Convert named dependencies to DependsOn relationships.
        
        Uses fuzzy matching on feature names.
        """
        dependencies = []

        # Build name -> ID lookup
        name_to_id = {}
        for f in features:
            name_to_id[f.name.lower()] = f.id
            # Also add partial matches for first few words
            words = f.name.lower().split()[:3]
            for i in range(1, len(words) + 1):
                partial = ' '.join(words[:i])
                if partial not in name_to_id:
                    name_to_id[partial] = f.id

        # Resolve dependencies
        for i, parsed in enumerate(parsed_deps):
            if not parsed.get('depends_on'):
                continue

            from_feature = features[i]

            for dep_name in parsed['depends_on']:
                dep_name_lower = dep_name.lower().strip()

                # Try exact match
                if dep_name_lower in name_to_id:
                    to_id = name_to_id[dep_name_lower]
                    if to_id != from_feature.id:
                        dependencies.append(DependsOn(
                            from_feature_id=from_feature.id,
                            to_feature_id=to_id
                        ))
                else:
                    # Try fuzzy match
                    for name, fid in name_to_id.items():
                        if dep_name_lower in name or name in dep_name_lower:
                            if fid != from_feature.id:
                                dependencies.append(DependsOn(
                                    from_feature_id=from_feature.id,
                                    to_feature_id=fid
                                ))
                                break

        return dependencies


# ============================================================================
# Test Scaffolder
# ============================================================================

class TestScaffolder:
    """Generate test scaffolding for features."""

    @staticmethod
    def generate(features: list[Feature], project_name: str) -> dict[str, str]:
        """
        Generate test file templates.
        
        Returns dict of filename -> content.
        """
        scaffolding = {}

        # Main test file
        test_content = f'''"""
Tests for {project_name}
Auto-generated by Domain Harness Initializer
"""

import pytest


'''

        for feature in features:
            safe_name = re.sub(r'[^a-z0-9_]', '_', feature.name.lower())[:40]
            test_content += f'''
class Test{safe_name.title().replace("_", "")}:
    """Tests for: {feature.name}"""
    
    @pytest.mark.skip(reason="Not implemented")
    def test_{safe_name}_basic(self):
        """Basic test for {feature.name}"""
        # TODO: Implement test
        # Acceptance criteria:
'''
            for criterion in feature.acceptance_criteria or [feature.description]:
                test_content += f'        #   - {criterion}\n'

            test_content += '''        assert False, "Not implemented"

'''

        scaffolding['test_domain.py'] = test_content

        # Conftest
        conftest = f'''"""
Pytest configuration for {project_name}
"""

import pytest


@pytest.fixture
def domain_context():
    """Provide domain context for tests."""
    return {{
        "project": "{project_name}",
        "features": {len(features)}
    }}
'''
        scaffolding['conftest.py'] = conftest

        return scaffolding


# ============================================================================
# Main Initializer
# ============================================================================

class DomainInitializer:
    """
    Transform user prompt into domain memory.
    
    Usage:
        initializer = DomainInitializer(yarn=Neo4jKG())
        result = await initializer.initialize('''
            Build a REST API with:
            - User authentication (JWT)
            - CRUD for blog posts (depends on authentication)
            - Rate limiting
        ''')
    """

    def __init__(
        self,
        yarn=None,  # Neo4jKG or compatible
        warp=None,  # WarpSpace for embeddings
        llm=None    # LLM for advanced parsing (optional)
    ):
        self.yarn = yarn
        self.warp = warp
        self.llm = llm

        self.parser = FeatureParser()
        self.resolver = DependencyResolver()
        self.scaffolder = TestScaffolder()

    async def initialize(
        self,
        prompt: str,
        project_name: str | None = None,
        constraints: list[str] | None = None,
        environment: dict[str, str] | None = None
    ) -> InitializationResult:
        """
        Initialize domain memory from prompt.
        
        Args:
            prompt: User's project description
            project_name: Optional name (inferred if not provided)
            constraints: Global constraints/requirements
            environment: Environment configuration
            
        Returns:
            InitializationResult with all artifacts
        """
        # Infer project name if not provided
        if not project_name:
            project_name = self._infer_project_name(prompt)

        # Parse features (use LLM if available, else regex)
        if self.llm:
            parsed_features = await self._llm_parse_features(prompt)
        else:
            parsed_features = self.parser.parse(prompt)

        # Create Feature objects
        features = []
        for i, pf in enumerate(parsed_features):
            feature = create_feature(
                name=pf['name'],
                description=pf['description'],
                priority=pf.get('priority', len(parsed_features) - i),  # Default: order in list
                acceptance_criteria=[pf['description']] if pf['description'] != pf['name'] else []
            )
            features.append(feature)

        # Resolve dependencies
        dependencies = self.resolver.resolve(features, parsed_features)

        # Create domain state
        state = DomainState(
            project_name=project_name,
            constraints=constraints or [],
            environment=environment or {},
            total_features=len(features)
        )

        # Generate test scaffolding
        test_scaffolding = self.scaffolder.generate(features, project_name)

        # Persist to Neo4j if available
        if self.yarn:
            await self._persist_to_neo4j(features, dependencies, state)

        # Embed features in Qdrant if available
        if self.warp:
            await self._embed_features(features)

        return InitializationResult(
            project_name=project_name,
            features=features,
            dependencies=dependencies,
            state=state,
            test_scaffolding=test_scaffolding
        )

    def _infer_project_name(self, prompt: str) -> str:
        """Infer project name from prompt."""
        # Look for explicit project name
        name_match = re.search(r'(?:project|app|system|service):\s*([^\n,]+)', prompt, re.I)
        if name_match:
            return name_match.group(1).strip()

        # Use first significant noun phrase
        words = prompt.split()[:5]
        name = ' '.join(w for w in words if len(w) > 2 and w[0].isalpha())
        return name or "unnamed_project"

    async def _llm_parse_features(self, prompt: str) -> list[dict[str, Any]]:
        """Use LLM for more intelligent feature extraction."""
        # Would call self.llm with structured output prompt
        # Fall back to regex parsing for now
        return self.parser.parse(prompt)

    async def _persist_to_neo4j(
        self,
        features: list[Feature],
        dependencies: list[DependsOn],
        state: DomainState
    ):
        """Save to Neo4j graph."""
        # Create state node
        # await self.yarn.execute(
        #     "MERGE (s:DomainState {id: 'singleton'}) SET s += $props",
        #     {"props": state.to_cypher_properties()}
        # )

        # Create feature nodes
        for f in features:
            pass
            # await self.yarn.execute(
            #     "CREATE (f:Feature $props)",
            #     {"props": f.to_cypher_properties()}
            # )

        # Create dependency relationships
        for dep in dependencies:
            pass
            # await self.yarn.execute(dep.to_cypher())

    async def _embed_features(self, features: list[Feature]):
        """Embed features in Qdrant for semantic search."""
        for f in features:
            pass
            # text = f"{f.name}: {f.description}"
            # await self.warp.add(text, metadata=f.to_cypher_properties())


# ============================================================================
# CLI Helper
# ============================================================================

def initialize_from_file(filepath: str, yarn=None) -> InitializationResult:
    """
    Initialize domain from a prompt file.
    
    Sync wrapper for CLI usage.
    """
    import asyncio

    with open(filepath) as f:
        prompt = f.read()

    initializer = DomainInitializer(yarn=yarn)
    return asyncio.run(initializer.initialize(prompt))


# ============================================================================
# Prompt Templates
# ============================================================================

EXAMPLE_PROMPT = '''
Build a REST API with the following features:

- User authentication with JWT tokens [HIGH]
- User registration and profile management (depends on authentication)
- CRUD operations for blog posts (depends on authentication)
- Comment system for posts (depends on blog posts)
- Rate limiting middleware [HIGH]
- API documentation with OpenAPI/Swagger
- Unit tests with >80% coverage

Constraints:
- Python 3.11+
- FastAPI framework
- PostgreSQL database
- Docker deployment
'''

INITIALIZATION_SYSTEM_PROMPT = '''
You are a Domain Initializer Agent.

Your job is to convert the user's project description into structured domain memory artifacts.

Output a JSON object with:
1. features: Array of feature objects with:
   - name: Short descriptive name
   - description: Detailed description
   - depends_on: Array of feature names this depends on
   - priority: 0-20 (higher = more important)
   - acceptance_criteria: Array of testable criteria

2. constraints: Array of global project constraints

3. environment: Object with environment configuration

Rules:
- All features start with status "pending"
- Dependencies must reference other feature names exactly
- Higher priority features should be foundational
- Each feature should be independently testable
- Break large features into smaller, atomic units
'''
