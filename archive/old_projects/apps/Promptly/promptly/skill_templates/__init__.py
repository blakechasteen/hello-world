"""
Skill Templates for Promptly
=============================
Pre-built, production-ready skills for common AI tasks.
"""

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


def get_template(name: str) -> dict:
    """Get a skill template by name."""
    return TEMPLATES.get(name)


def list_templates() -> list:
    """List all available templates."""
    return [
        {
            "name": name,
            "description": template["description"],
            "tags": template["metadata"].get("tags", [])
        }
        for name, template in TEMPLATES.items()
    ]