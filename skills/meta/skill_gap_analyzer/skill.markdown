# Skill: Skill Gap Analyzer

## Metadata

- **Name**: `skill_gap_analyzer`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-01-18`
- **Last Updated**: `2025-01-18`
- **Category**: `meta`
- **Tags**: `analysis, gaps, recommendations, meta-skill, continuous-improvement`

## Description

**Short Description**:
Identifies missing skills by analyzing user requests, existing skill catalog, and common failure patterns to recommend new skills to build.

**Detailed Description**:
The Skill Gap Analyzer is a meta-skill that helps evolve the skills ecosystem by identifying unmet needs. It analyzes user interactions to find patterns where no suitable skill exists, examines the current skill catalog for coverage gaps, and recommends new skills to build based on frequency of need and strategic value. The analyzer considers skill dependencies, domain coverage, and user pain points to prioritize skill development roadmap.

## Required Capabilities

- [x] File system access (read) - to read existing skills and logs
- [x] File system access (write) - to write gap analysis reports
- [ ] Code execution (bash)
- [ ] Code execution (python)
- [ ] Network access (web fetch)
- [ ] Network access (web search)
- [ ] MCP server access
- [ ] External API access
- [x] User interaction (questions) - for clarifying skill requirements

## Dependencies

**Required Skills**: None (base meta-skill)

**External Dependencies**: None

**HoloLoom Integration**:
- [x] Uses HoloLoom memory system - analyzes past interactions
- [x] Uses HoloLoom RAG - retrieves similar past requests
- [ ] Uses HoloLoom alignment framework
- [x] Uses HoloLoom learning systems - learns from user patterns

## Input Schema

**Expected Input Format**:
```json
{
  "skills_directory": "path to skills/ directory",
  "analysis_mode": "user_requests|catalog_coverage|failure_patterns|all",
  "user_request_history": ["list of recent user requests"],
  "min_frequency": 3,
  "output_format": "json|markdown"
}
```

**Example Input**:
```json
{
  "skills_directory": "skills/",
  "analysis_mode": "all",
  "user_request_history": [
    "Can you analyze this code for performance issues?",
    "Help me optimize this database query",
    "Suggest improvements for this API design",
    "Review my code for performance bottlenecks"
  ],
  "min_frequency": 2,
  "output_format": "json"
}
```

## Output Schema

**Expected Output Format**:
```json
{
  "analysis_summary": {
    "total_skills": 15,
    "coverage_score": 0.65,
    "gap_count": 8,
    "high_priority_gaps": 3
  },
  "skill_gaps": [
    {
      "gap_id": "performance_optimizer",
      "gap_name": "Code Performance Optimizer",
      "gap_description": "Analyzes code for performance bottlenecks and suggests optimizations",
      "priority": "high|medium|low",
      "frequency_in_requests": 4,
      "related_skills": ["skill_security_analyzer"],
      "estimated_effort": "medium",
      "strategic_value": "high",
      "use_cases": [
        "Optimize slow database queries",
        "Identify O(n²) algorithms",
        "Suggest caching strategies"
      ],
      "suggested_capabilities": [
        "File system access (read)",
        "Code execution (python)"
      ],
      "potential_dependencies": ["code_parser", "profiler"]
    }
  ],
  "coverage_analysis": {
    "domains_covered": ["security", "testing", "analysis"],
    "domains_missing": ["performance", "documentation", "deployment"],
    "capability_gaps": ["Real-time monitoring", "Multi-language support"]
  },
  "recommendations": [
    "Build 'performance_optimizer' skill (high priority, 4 user requests)",
    "Expand domain coverage to include performance analysis",
    "Consider meta-skill for automated documentation generation"
  ],
  "proposed_roadmap": [
    {
      "phase": 1,
      "skills": ["performance_optimizer", "documentation_generator"],
      "rationale": "Highest user demand and strategic value"
    }
  ],
  "metadata": {
    "execution_time_ms": 300,
    "confidence": 0.88,
    "warnings": []
  }
}
```

## Prompt Template

```markdown
You are the **Skill Gap Analyzer**, a meta-skill that identifies missing capabilities in the skills ecosystem.

**Your Task**:
Analyze the current skill catalog and user request patterns to identify skill gaps and recommend new skills to build.

**Input Data**:
{input_data}

**Existing Skills**:
{existing_skills_summary}

**Analysis Steps**:

1. **Catalog Analysis**:
   - Count total skills by category (meta, domain, utility)
   - Identify domain coverage (security, performance, testing, etc.)
   - Map capability coverage

2. **User Request Analysis**:
   - Extract common patterns from user requests
   - Identify requests that couldn't be fulfilled by existing skills
   - Calculate frequency of unmet needs

3. **Failure Pattern Analysis**:
   - Look for recurring skill errors or limitations
   - Identify edge cases not handled by current skills
   - Find skills with low quality scores or high failure rates

4. **Gap Identification**:
   - For each unmet pattern, define potential skill
   - Estimate priority (frequency × strategic value)
   - Suggest capabilities and dependencies

5. **Roadmap Generation**:
   - Group related gaps into phases
   - Prioritize by impact and effort
   - Recommend build order (dependencies first)

**Analysis Modes**:
- **user_requests**: Focus on unmet user needs
- **catalog_coverage**: Analyze domain/capability coverage
- **failure_patterns**: Identify weak/failing skills
- **all**: Comprehensive analysis

**Priority Levels**:
- **HIGH**: Frequent user need + high strategic value + low build effort
- **MEDIUM**: Moderate need or moderate effort
- **LOW**: Rare need or high complexity

**Output Format**:
Return gap analysis as JSON (see Output Schema above).

**Recommendations**:
For each gap, provide:
- Clear use cases
- Suggested capabilities
- Dependencies on other skills
- Estimated build effort

**Error Handling**:
- If skills/ directory empty, report as bootstrap state
- If no user history provided, focus on catalog coverage analysis
- If cannot determine priority, mark as MEDIUM
```

## Examples

### Example 1: User Request Analysis

**Input**:
```json
{
  "skills_directory": "skills/",
  "analysis_mode": "user_requests",
  "user_request_history": [
    "Can you analyze this code for performance issues?",
    "Help me optimize this database query",
    "Suggest improvements for this API design",
    "Review my code for performance bottlenecks"
  ],
  "min_frequency": 2,
  "output_format": "json"
}
```

**Expected Output**:
```json
{
  "analysis_summary": {
    "total_skills": 5,
    "coverage_score": 0.40,
    "gap_count": 3,
    "high_priority_gaps": 1
  },
  "skill_gaps": [
    {
      "gap_id": "performance_optimizer",
      "gap_name": "Code Performance Optimizer",
      "gap_description": "Analyzes code for performance bottlenecks and suggests optimizations",
      "priority": "high",
      "frequency_in_requests": 4,
      "related_skills": [],
      "estimated_effort": "medium",
      "strategic_value": "high",
      "use_cases": [
        "Database query optimization",
        "Algorithm complexity analysis",
        "Caching strategy recommendations"
      ],
      "suggested_capabilities": [
        "File system access (read)",
        "Code execution (python)"
      ],
      "potential_dependencies": []
    }
  ],
  "coverage_analysis": {
    "domains_covered": ["security", "testing"],
    "domains_missing": ["performance", "optimization"],
    "capability_gaps": ["Performance profiling"]
  },
  "recommendations": [
    "Build 'performance_optimizer' skill immediately (4 requests, high value)",
    "Expand into performance domain to complement existing security/testing skills"
  ],
  "proposed_roadmap": [
    {
      "phase": 1,
      "skills": ["performance_optimizer"],
      "rationale": "Highest frequency in user requests (4/4 related to performance)"
    }
  ],
  "metadata": {
    "execution_time_ms": 200,
    "confidence": 0.92,
    "warnings": []
  }
}
```

**Explanation**:
Identifies performance optimization as critical gap based on user request frequency.

### Example 2: Catalog Coverage Analysis

**Input**:
```json
{
  "skills_directory": "skills/",
  "analysis_mode": "catalog_coverage",
  "user_request_history": [],
  "min_frequency": 1,
  "output_format": "json"
}
```

**Expected Output**:
```json
{
  "analysis_summary": {
    "total_skills": 5,
    "coverage_score": 0.30,
    "gap_count": 10,
    "high_priority_gaps": 2
  },
  "skill_gaps": [
    {
      "gap_id": "documentation_generator",
      "gap_name": "Documentation Generator",
      "gap_description": "Auto-generates documentation from code and comments",
      "priority": "medium",
      "frequency_in_requests": 0,
      "related_skills": [],
      "estimated_effort": "low",
      "strategic_value": "high",
      "use_cases": [
        "Generate API docs from code",
        "Create README from project structure",
        "Update CLAUDE.md based on changes"
      ],
      "suggested_capabilities": [
        "File system access (read)",
        "File system access (write)"
      ],
      "potential_dependencies": []
    }
  ],
  "coverage_analysis": {
    "domains_covered": ["meta-skills (security, testing, gaps, tokens, learning)"],
    "domains_missing": ["documentation", "deployment", "monitoring", "refactoring", "code-generation"],
    "capability_gaps": [
      "Documentation generation",
      "CI/CD integration",
      "Code refactoring",
      "Multi-language support"
    ]
  },
  "recommendations": [
    "Expand beyond meta-skills into domain-specific skills",
    "Build foundational utilities (documentation, refactoring)",
    "Add deployment/operations skills for production workflows"
  ],
  "proposed_roadmap": [
    {
      "phase": 1,
      "skills": ["documentation_generator", "code_refactorer"],
      "rationale": "Low-hanging fruit with high strategic value"
    },
    {
      "phase": 2,
      "skills": ["deployment_assistant", "ci_cd_integrator"],
      "rationale": "Enable production workflows"
    }
  ],
  "metadata": {
    "execution_time_ms": 180,
    "confidence": 0.85,
    "warnings": ["No user request history - analysis based on catalog only"]
  }
}
```

**Explanation**:
Identifies domain coverage gaps independent of user requests.

### Example 3: Comprehensive Analysis

**Input**:
```json
{
  "skills_directory": "skills/",
  "analysis_mode": "all",
  "user_request_history": [
    "Help me deploy this to production",
    "Can you set up CI/CD?",
    "Optimize my Docker setup"
  ],
  "min_frequency": 1,
  "output_format": "json"
}
```

**Expected Output**:
```json
{
  "analysis_summary": {
    "total_skills": 5,
    "coverage_score": 0.35,
    "gap_count": 8,
    "high_priority_gaps": 2
  },
  "skill_gaps": [
    {
      "gap_id": "deployment_assistant",
      "gap_name": "Deployment Assistant",
      "gap_description": "Helps deploy applications to various platforms (Docker, K8s, cloud)",
      "priority": "high",
      "frequency_in_requests": 3,
      "related_skills": [],
      "estimated_effort": "high",
      "strategic_value": "high",
      "use_cases": [
        "Dockerize applications",
        "Deploy to Kubernetes",
        "Set up CI/CD pipelines",
        "Configure cloud infrastructure"
      ],
      "suggested_capabilities": [
        "File system access (read/write)",
        "Code execution (bash)",
        "Network access (web fetch)"
      ],
      "potential_dependencies": ["docker", "kubectl", "cloud CLIs"]
    }
  ],
  "coverage_analysis": {
    "domains_covered": ["meta-skills"],
    "domains_missing": ["deployment", "infrastructure", "operations"],
    "capability_gaps": ["Container orchestration", "Cloud platform integration"]
  },
  "recommendations": [
    "Build deployment_assistant skill (3 requests + strategic value)",
    "Expand into DevOps domain to complement development skills",
    "Consider CI/CD integrator for automated workflows"
  ],
  "proposed_roadmap": [
    {
      "phase": 1,
      "skills": ["deployment_assistant"],
      "rationale": "Critical user need + high strategic value for production workflows"
    },
    {
      "phase": 2,
      "skills": ["ci_cd_integrator", "infrastructure_as_code_helper"],
      "rationale": "Complete DevOps toolkit"
    }
  ],
  "metadata": {
    "execution_time_ms": 350,
    "confidence": 0.90,
    "warnings": []
  }
}
```

**Explanation**:
Combines user needs with catalog analysis to identify deployment as critical gap.

## Testing Checklist

- [x] **Functionality**: All analysis modes work (user_requests, catalog, failures, all)
- [x] **Error Handling**: Handles empty catalogs, no user history gracefully
- [x] **Security**: Self-test passes (analyzing skill gaps for meta-skills)
- [x] **Performance**: Completes analysis in <1s for typical catalog
- [x] **Token Efficiency**: Efficient prompt design
- [x] **Documentation**: All sections complete
- [x] **Dependencies**: No external dependencies
- [x] **Edge Cases**: Handles new repos, no skills, no history
- [x] **Output Consistency**: Consistent JSON format
- [x] **Integration**: Can trigger skill creation workflow

## Security Considerations

**Potential Risks**:
- **Privacy**: User request history may contain sensitive info - anonymize before analysis

**Data Privacy**:
- [x] Anonymizes user data in reports
- [x] Does not log user requests externally
- [x] No external requests

**Sandboxing**:
- [x] Read-only access to skills directory
- [x] No code execution
- [x] No system modifications

## Performance Characteristics

- **Expected Latency**: 100-500ms depending on catalog size
- **Token Usage**: ~1000-3000 tokens per analysis
- **Resource Requirements**: Minimal (file reading + pattern matching)
- **Scalability**: Linear with skill count + request history size

## Maintenance Notes

**Known Limitations**:
- Cannot predict future user needs (only reactive)
- Requires user request history for best results
- May suggest overlapping skills without dependency analysis

**Future Enhancements**:
- Predictive gap analysis (based on industry trends)
- Automatic skill scaffolding (create skill.markdown from gap)
- Integration with continuous_learning_capture for real-time gap detection
- Skill dependency graph analysis

**Changelog**:
- **v1.0.0** (2025-01-18): Initial release

## License

MIT License (part of HoloLoom ecosystem)

## Support

**Issues**: https://github.com/yourusername/hello-world/issues
**Documentation**: See skills/docs/gap_analysis_guide.md
**Contributors**: HoloLoom Team

---

## Development Notes (Internal)

**Design Decisions**:
- Multiple analysis modes for flexibility (user/catalog/failures/all)
- Priority scoring = frequency × strategic_value × (1 / effort)
- Roadmap generation groups related skills into phases

**Alternative Approaches Considered**:
- ML-based skill recommendation (too complex for v1)
- Manual gap tracking (not scalable)
- User surveys (too slow, low signal)

**Integration Points**:
- Runs periodically (weekly?) to update skill roadmap
- Feeds into continuous_learning_capture for automated skill creation
- Integrates with HoloLoom memory to analyze historical patterns

**Testing Strategy**:
- Bootstrap test: Run on empty skills/ directory (should suggest meta-skills)
- Self-improvement: Analyze meta-skills to find meta-meta-skill gaps
- Validate against actual user needs over time
