#!/usr/bin/env python3
"""
Extended Skill Templates for Promptly
======================================
5 additional production-ready skill templates
"""

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


# Merge with existing templates
def get_all_templates():
    """Get all skill templates (original + extended)"""
    try:
        from skill_templates import TEMPLATES
        return {**TEMPLATES, **EXTENDED_TEMPLATES}
    except ImportError:
        return EXTENDED_TEMPLATES


if __name__ == "__main__":
    print("Extended Skill Templates for Promptly")
    print("\n5 New Templates:")
    for name, template in EXTENDED_TEMPLATES.items():
        print(f"\n  {name}")
        print(f"    {template['description']}")
        print(f"    Tags: {', '.join(template['metadata']['tags'])}")
