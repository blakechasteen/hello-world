"""
HoloLoom Skill Templates
========================
Pre-built, production-ready skill templates for agentic reasoning.

Integrated from Promptly (November 2025).
Contains 13 skill templates: 8 base + 5 extended.

Each template includes:
- Description and metadata
- Runtime requirements (claude, ollama, etc.)
- Tags for categorization
- Template files (markdown docs, code examples, configs)
"""

from typing import Dict, List, Any


# ============================================================================
# Base Templates (8 templates)
# ============================================================================

TEMPLATES = {
    "code_reviewer": {
        "description": "Review code for best practices, bugs, and security issues",
        "metadata": {
            "runtime": "claude",
            "tags": ["code", "review", "quality"],
            "model": "claude-3-5-sonnet-20241022"
        },
        "files": {
            "review_checklist.md": """# Code Review Checklist

## Security
- [ ] No hardcoded credentials or API keys
- [ ] Input validation on all user inputs
- [ ] SQL injection prevention
- [ ] XSS prevention in web contexts
- [ ] Proper authentication/authorization

## Code Quality
- [ ] Clear, descriptive variable names
- [ ] Functions are single-purpose and focused
- [ ] No code duplication
- [ ] Proper error handling
- [ ] Edge cases handled

## Performance
- [ ] No obvious performance bottlenecks
- [ ] Efficient algorithms and data structures
- [ ] Database queries optimized
- [ ] Proper caching where applicable

## Maintainability
- [ ] Code is well-documented
- [ ] Tests are included
- [ ] Dependencies are minimal and justified
- [ ] Follows project conventions
""",
            "reviewer.py": """def review_code(code: str, language: str, focus: str = "all") -> dict:
    \"\"\"
    Review code and return structured feedback.

    Args:
        code: The code to review
        language: Programming language
        focus: "security", "performance", "quality", or "all"

    Returns:
        dict with 'issues', 'suggestions', 'score'
    \"\"\"
    # This is a template - actual implementation would use LLM
    return {
        "issues": [],
        "suggestions": [],
        "score": 0
    }
"""
        }
    },

    "api_designer": {
        "description": "Design RESTful APIs with best practices and OpenAPI specs",
        "metadata": {
            "runtime": "claude",
            "tags": ["api", "design", "rest"],
            "model": "claude-3-5-sonnet-20241022"
        },
        "files": {
            "design_principles.md": """# API Design Principles

## RESTful Best Practices
1. Use nouns for resources, not verbs
2. Use HTTP methods correctly (GET, POST, PUT, DELETE)
3. Use plural nouns for collections
4. Version your API (v1, v2, etc.)
5. Return proper HTTP status codes

## Resource Naming
- Good: `/users`, `/users/123`, `/users/123/posts`
- Bad: `/getUser`, `/user`, `/createPost`

## Status Codes
- 200 OK - Success
- 201 Created - Resource created
- 400 Bad Request - Client error
- 401 Unauthorized - Authentication required
- 404 Not Found - Resource not found
- 500 Internal Server Error - Server error

## Security
- Always use HTTPS
- Implement authentication (JWT, OAuth)
- Rate limiting
- Input validation
- CORS configuration
""",
            "openapi_template.yaml": """openapi: 3.0.0
info:
  title: API Name
  version: 1.0.0
  description: API description

servers:
  - url: https://api.example.com/v1

paths:
  /resource:
    get:
      summary: List resources
      responses:
        '200':
          description: Success
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Resource'

    post:
      summary: Create resource
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ResourceInput'
      responses:
        '201':
          description: Created

components:
  schemas:
    Resource:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
"""
        }
    },

    "data_analyzer": {
        "description": "Analyze datasets and generate insights with visualizations",
        "metadata": {
            "runtime": "claude",
            "tags": ["data", "analytics", "visualization"],
            "requires": ["pandas", "matplotlib"]
        },
        "files": {
            "analysis_template.py": """import pandas as pd
import matplotlib.pyplot as plt

def analyze_dataset(data: pd.DataFrame) -> dict:
    \"\"\"
    Perform comprehensive data analysis.

    Returns:
        dict with 'summary', 'insights', 'visualizations'
    \"\"\"
    analysis = {
        "summary": {
            "rows": len(data),
            "columns": len(data.columns),
            "missing_values": data.isnull().sum().to_dict(),
            "data_types": data.dtypes.to_dict()
        },
        "insights": [],
        "visualizations": []
    }

    # Statistical summary
    analysis["statistics"] = data.describe().to_dict()

    # Detect patterns
    numeric_cols = data.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if data[col].std() > 0:
            analysis["insights"].append(f"{col} has variance")

    return analysis
""",
            "viz_examples.md": """# Visualization Examples

## Distribution Plot
```python
plt.figure(figsize=(10, 6))
data['column'].hist(bins=30)
plt.title('Distribution of Column')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

## Correlation Heatmap
```python
import seaborn as sns
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

## Time Series
```python
data.set_index('date')['value'].plot(figsize=(12, 6))
plt.title('Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()
```
"""
        }
    },

    "documentation_writer": {
        "description": "Generate comprehensive documentation from code",
        "metadata": {
            "runtime": "claude",
            "tags": ["documentation", "markdown", "readme"]
        },
        "files": {
            "doc_structure.md": """# Documentation Structure

## README.md
1. Project Title & Description
2. Installation Instructions
3. Quick Start / Usage Examples
4. Features
5. API Reference (if applicable)
6. Contributing Guidelines
7. License

## API Documentation
1. Function/Method signature
2. Description
3. Parameters (with types)
4. Return value (with type)
5. Examples
6. Notes/Warnings

## Architecture Documentation
1. System Overview
2. Component Diagram
3. Data Flow
4. Key Design Decisions
5. Dependencies
6. Deployment
""",
            "doc_generator.py": """def generate_readme(project_info: dict) -> str:
    \"\"\"Generate README.md content.\"\"\"
    template = f\"\"\"# {project_info['name']}

{project_info.get('description', 'Project description')}

## Installation

```bash
{project_info.get('install_cmd', 'pip install project-name')}
```

## Quick Start

```python
{project_info.get('quick_start', '# Example usage here')}
```

## Features

{chr(10).join(f"- {f}" for f in project_info.get('features', []))}

## License

{project_info.get('license', 'MIT')}
\"\"\"
    return template
"""
        }
    },

    "test_generator": {
        "description": "Generate comprehensive unit tests for code",
        "metadata": {
            "runtime": "claude",
            "tags": ["testing", "pytest", "unittest"]
        },
        "files": {
            "test_template.py": """import pytest

def test_function_name():
    \"\"\"Test basic functionality.\"\"\"
    # Arrange
    input_data = ...
    expected = ...

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result == expected


def test_function_name_edge_case():
    \"\"\"Test edge case handling.\"\"\"
    with pytest.raises(ValueError):
        function_under_test(invalid_input)


def test_function_name_empty_input():
    \"\"\"Test with empty input.\"\"\"
    result = function_under_test([])
    assert result == []


@pytest.fixture
def sample_data():
    \"\"\"Fixture for test data.\"\"\"
    return {"key": "value"}


def test_with_fixture(sample_data):
    \"\"\"Test using fixture.\"\"\"
    result = process(sample_data)
    assert result is not None
""",
            "test_guidelines.md": """# Testing Guidelines

## Test Structure (AAA Pattern)
1. **Arrange** - Set up test data and conditions
2. **Act** - Execute the function being tested
3. **Assert** - Verify the results

## What to Test
- Happy path (normal usage)
- Edge cases (boundary conditions)
- Error cases (invalid input)
- Empty/null inputs
- Large inputs
- Concurrent access (if applicable)

## Test Naming
- Use descriptive names: `test_function_name_expected_behavior`
- Example: `test_divide_by_zero_raises_error`

## Coverage Goals
- Aim for 80%+ code coverage
- 100% coverage of critical paths
- All error handling paths tested
"""
        }
    },

    "prompt_engineer": {
        "description": "Optimize and improve prompts for better AI responses",
        "metadata": {
            "runtime": "claude",
            "tags": ["prompts", "optimization", "ai"]
        },
        "files": {
            "prompt_patterns.md": """# Prompt Engineering Patterns

## Chain of Thought
```
Let's think through this step by step:
1. First, analyze the problem
2. Then, consider alternatives
3. Finally, choose the best approach
```

## Few-Shot Learning
```
Example 1:
Input: [example]
Output: [expected output]

Example 2:
Input: [example]
Output: [expected output]

Now process:
Input: [actual input]
Output:
```

## Role-Based Prompting
```
You are an expert [role] with deep knowledge of [domain].
Your task is to [specific task].
Consider [important factors].
```

## Structured Output
```
Please respond in the following JSON format:
{
  "analysis": "...",
  "recommendation": "...",
  "confidence": 0-100
}
```

## Constraint Setting
```
Requirements:
- Response must be under 200 words
- Use technical terminology
- Include code examples
- Focus on practical application
```
""",
            "optimizer.py": """def optimize_prompt(original: str, goal: str) -> str:
    \"\"\"
    Optimize a prompt for better results.

    Improvements:
    - Add clarity
    - Set constraints
    - Provide examples
    - Define output format
    \"\"\"
    # Template for improved prompts
    return f\"\"\"
You are an expert assistant.

Context: {goal}

Task: {original}

Please provide:
1. A detailed analysis
2. Clear recommendations
3. Supporting examples

Format your response as structured output.
\"\"\"
"""
        }
    },

    "sql_designer": {
        "description": "Design database schemas and write optimized SQL queries",
        "metadata": {
            "runtime": "claude",
            "tags": ["sql", "database", "schema"]
        },
        "files": {
            "schema_template.sql": """-- Database Schema Template

-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);

-- Example foreign key relationship
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    content TEXT,
    published BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_posts_user_id ON posts(user_id);
CREATE INDEX idx_posts_published ON posts(published);
""",
            "query_patterns.md": """# SQL Query Patterns

## Efficient Joins
```sql
-- Use INNER JOIN for required relationships
SELECT u.username, p.title
FROM users u
INNER JOIN posts p ON u.id = p.user_id
WHERE p.published = TRUE;
```

## Aggregations
```sql
-- Count with GROUP BY
SELECT user_id, COUNT(*) as post_count
FROM posts
GROUP BY user_id
HAVING COUNT(*) > 5;
```

## Pagination
```sql
-- Efficient pagination
SELECT * FROM posts
ORDER BY created_at DESC
LIMIT 10 OFFSET 20;
```

## Subqueries vs Joins
```sql
-- Subquery (for filtering)
SELECT * FROM users
WHERE id IN (SELECT DISTINCT user_id FROM posts);

-- Join (for data retrieval)
SELECT u.*, p.title FROM users u
JOIN posts p ON u.id = p.user_id;
```
"""
        }
    },

    "error_handler": {
        "description": "Design robust error handling and logging strategies",
        "metadata": {
            "runtime": "claude",
            "tags": ["errors", "logging", "debugging"]
        },
        "files": {
            "error_patterns.py": """import logging
from functools import wraps
from typing import Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CustomError(Exception):
    \"\"\"Base exception for custom errors.\"\"\"
    def __init__(self, message: str, code: str = None):
        self.message = message
        self.code = code
        super().__init__(self.message)


def retry_on_failure(max_attempts: int = 3):
    \"\"\"Decorator to retry function on failure.\"\"\"
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_attempts - 1:
                        raise
            return None
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, **kwargs):
    \"\"\"Execute function with comprehensive error handling.\"\"\"
    try:
        return func(*args, **kwargs)
    except ValueError as e:
        logger.error(f"Value error in {func.__name__}: {e}")
        return None
    except KeyError as e:
        logger.error(f"Key error in {func.__name__}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error in {func.__name__}: {e}")
        raise
""",
            "logging_config.yaml": """version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.FileHandler
    level: DEBUG
    formatter: detailed
    filename: app.log

  error_file:
    class: logging.FileHandler
    level: ERROR
    formatter: detailed
    filename: errors.log

loggers:
  app:
    level: DEBUG
    handlers: [console, file, error_file]
    propagate: false

root:
  level: INFO
  handlers: [console]
"""
        }
    }
}


# ============================================================================
# Extended Templates (5 templates)
# ============================================================================

EXTENDED_TEMPLATES = {
    "sql_optimizer": {
        "description": "Optimize SQL queries for performance and readability",
        "metadata": {
            "runtime": "claude",
            "tags": ["sql", "database", "performance", "optimization"],
            "model": "claude-3-5-sonnet-20241022"
        },
        "files": {
            "optimizer.md": """# SQL Optimizer Skill

## Purpose
Analyze and optimize SQL queries for better performance, readability, and maintainability.

## Capabilities
- Index recommendations
- Query plan analysis
- N+1 query detection
- Join optimization
- Subquery to CTE conversion
- Performance profiling

## Usage
Provide a SQL query and optional schema information.
""",
            "optimize.py": """def optimize_sql(query: str, schema: dict = None, database_type: str = "postgresql"):
    \"\"\"
    Optimize SQL query for performance.

    Args:
        query: SQL query to optimize
        schema: Optional database schema
        database_type: Target database (postgresql, mysql, sqlite)

    Returns:
        Optimized query with explanation
    \"\"\"

    # Analysis checklist:
    # 1. Identify missing indexes
    # 2. Check for N+1 queries
    # 3. Convert subqueries to CTEs
    # 4. Optimize JOIN order
    # 5. Add query hints if needed
    # 6. Check for SELECT *
    # 7. Verify WHERE clause pushdown

    pass  # Implement with LLM
"""
        }
    },

    "ui_designer": {
        "description": "Design user interfaces with accessibility and UX best practices",
        "metadata": {
            "runtime": "claude",
            "tags": ["ui", "ux", "design", "accessibility"],
            "model": "claude-3-5-sonnet-20241022"
        },
        "files": {
            "designer.md": """# UI Designer Skill

## Purpose
Create accessible, beautiful, user-friendly interface designs.

## Capabilities
- Component design (buttons, forms, navigation)
- Color palette generation (WCAG compliant)
- Layout recommendations (responsive, mobile-first)
- Accessibility audits (ARIA, keyboard nav)
- Design system creation
- Figma-ready specs

## Principles
1. Accessibility first (WCAG 2.1 AA minimum)
2. Mobile-first responsive design
3. Clear visual hierarchy
4. Consistent spacing (8pt grid)
5. High contrast ratios
6. Keyboard navigable
""",
            "design.py": """def design_component(
    component_type: str,
    requirements: dict,
    style_guide: dict = None
):
    \"\"\"
    Design UI component with accessibility.

    Args:
        component_type: button, form, card, nav, etc.
        requirements: Functional requirements
        style_guide: Optional brand guidelines

    Returns:
        Component spec with HTML/CSS/accessibility notes
    \"\"\"

    # Design checklist:
    # 1. WCAG contrast ratios
    # 2. Touch targets (44x44px minimum)
    # 3. Focus indicators
    # 4. ARIA labels
    # 5. Responsive breakpoints
    # 6. Loading states
    # 7. Error states

    pass  # Implement with LLM
"""
        }
    },

    "system_architect": {
        "description": "Design scalable, maintainable system architectures",
        "metadata": {
            "runtime": "claude",
            "tags": ["architecture", "system-design", "scalability"],
            "model": "claude-3-5-sonnet-20241022"
        },
        "files": {
            "architect.md": """# System Architect Skill

## Purpose
Design robust, scalable, maintainable system architectures.

## Capabilities
- High-level architecture diagrams (C4 model)
- Technology stack recommendations
- Scalability analysis (CAP theorem considerations)
- Security architecture
- Data flow diagrams
- API design
- Microservices vs monolith trade-offs

## Deliverables
- Architecture Decision Records (ADRs)
- Component diagrams
- Deployment diagrams
- Sequence diagrams
- Risk analysis
""",
            "design_system.py": """def design_architecture(
    requirements: dict,
    constraints: dict = None,
    scale: str = "medium"
):
    \"\"\"
    Design system architecture.

    Args:
        requirements: Functional & non-functional requirements
        constraints: Budget, timeline, team size
        scale: small, medium, large, hyperscale

    Returns:
        Architecture specification with diagrams
    \"\"\"

    # Architecture checklist:
    # 1. Identify components & boundaries
    # 2. Define APIs & contracts
    # 3. Choose data stores (RDBMS, NoSQL, cache)
    # 4. Plan for failure (circuit breakers, retries)
    # 5. Security (auth, encryption, OWASP)
    # 6. Monitoring & observability
    # 7. Deployment strategy
    # 8. Cost estimation

    pass  # Implement with LLM
"""
        }
    },

    "refactoring_expert": {
        "description": "Refactor code for better maintainability and performance",
        "metadata": {
            "runtime": "claude",
            "tags": ["refactoring", "code-quality", "clean-code"],
            "model": "claude-3-5-sonnet-20241022"
        },
        "files": {
            "refactor.md": """# Refactoring Expert Skill

## Purpose
Systematically improve code quality without changing behavior.

## Capabilities
- Code smell detection (long methods, god classes, etc.)
- SOLID principle application
- Design pattern recommendations
- Dependency injection
- Extract method/class refactorings
- Performance optimizations
- Dead code elimination

## Approach
1. Identify code smells
2. Write characterization tests
3. Apply small, safe refactorings
4. Run tests after each step
5. Commit frequently
""",
            "refactor_code.py": """def refactor(
    code: str,
    language: str,
    focus: str = "maintainability"
):
    \"\"\"
    Refactor code systematically.

    Args:
        code: Source code to refactor
        language: Programming language
        focus: maintainability, performance, testability

    Returns:
        Refactored code with explanation of changes
    \"\"\"

    # Refactoring checklist:
    # 1. Extract long methods (>20 lines)
    # 2. Eliminate code duplication (DRY)
    # 3. Apply single responsibility principle
    # 4. Reduce cyclomatic complexity
    # 5. Remove magic numbers
    # 6. Add descriptive names
    # 7. Simplify conditionals
    # 8. Extract constants

    pass  # Implement with LLM
"""
        }
    },

    "security_auditor": {
        "description": "Audit code and systems for security vulnerabilities",
        "metadata": {
            "runtime": "claude",
            "tags": ["security", "vulnerabilities", "owasp"],
            "model": "claude-3-5-sonnet-20241022"
        },
        "files": {
            "audit.md": """# Security Auditor Skill

## Purpose
Identify security vulnerabilities and recommend fixes.

## Capabilities
- OWASP Top 10 coverage
- SQL injection detection
- XSS vulnerability scanning
- CSRF protection validation
- Authentication/authorization review
- Secrets detection (hardcoded keys, tokens)
- Dependency vulnerability checking
- Secure coding best practices

## OWASP Top 10 (2021)
1. Broken Access Control
2. Cryptographic Failures
3. Injection
4. Insecure Design
5. Security Misconfiguration
6. Vulnerable Components
7. Authentication Failures
8. Software & Data Integrity Failures
9. Logging & Monitoring Failures
10. Server-Side Request Forgery
""",
            "audit_security.py": """def security_audit(
    code: str = None,
    config: dict = None,
    architecture: dict = None
):
    \"\"\"
    Perform security audit.

    Args:
        code: Source code to audit
        config: System configuration
        architecture: Architecture diagrams/specs

    Returns:
        Security report with vulnerabilities and remediation
    \"\"\"

    # Security checklist:
    # 1. Input validation (sanitize all inputs)
    # 2. Output encoding (prevent XSS)
    # 3. Parameterized queries (prevent SQL injection)
    # 4. Authentication (strong passwords, MFA)
    # 5. Authorization (least privilege)
    # 6. Encryption (at rest & in transit)
    # 7. Secrets management (no hardcoded keys)
    # 8. HTTPS only
    # 9. CSRF tokens
    # 10. Security headers (CSP, HSTS, etc.)

    pass  # Implement with LLM
"""
        }
    }
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_all_templates() -> Dict[str, Dict[str, Any]]:
    """
    Get all skill templates (base + extended).

    Returns:
        Dict of all 13 skill templates
    """
    return {**TEMPLATES, **EXTENDED_TEMPLATES}


def list_templates() -> List[Dict[str, Any]]:
    """
    List all available templates with metadata.

    Returns:
        List of dicts with name, description, tags
    """
    all_templates = get_all_templates()
    return [
        {
            "name": name,
            "description": template["description"],
            "tags": template["metadata"].get("tags", []),
            "runtime": template["metadata"].get("runtime", "unknown"),
            "model": template["metadata"].get("model", None),
        }
        for name, template in all_templates.items()
    ]


def get_template(name: str) -> Dict[str, Any]:
    """
    Get a specific skill template by name.

    Args:
        name: Template name (e.g., 'code_reviewer')

    Returns:
        Template dict or None if not found
    """
    all_templates = get_all_templates()
    return all_templates.get(name)


# Template counts
TEMPLATE_COUNT_BASE = len(TEMPLATES)
TEMPLATE_COUNT_EXTENDED = len(EXTENDED_TEMPLATES)
TEMPLATE_COUNT_TOTAL = TEMPLATE_COUNT_BASE + TEMPLATE_COUNT_EXTENDED


if __name__ == "__main__":
    print(f"HoloLoom Skill Templates: {TEMPLATE_COUNT_TOTAL} total")
    print(f"  Base Templates: {TEMPLATE_COUNT_BASE}")
    print(f"  Extended Templates: {TEMPLATE_COUNT_EXTENDED}\n")

    print("Available Templates:")
    for template_info in list_templates():
        print(f"\n  {template_info['name']}")
        print(f"    {template_info['description']}")
        print(f"    Tags: {', '.join(template_info['tags'])}")
        print(f"    Runtime: {template_info['runtime']}")