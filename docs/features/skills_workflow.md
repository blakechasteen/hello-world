# Claude Skills Workflow Guide

**Last Updated**: 2025-01-18
**Version**: 1.0

This guide provides a complete workflow for creating, testing, packaging, and deploying Claude skills to both Claude Code (local) and Claude Web/Desktop.

---

## Table of Contents

1. [Overview](#overview)
2. [Creating a New Skill](#creating-a-new-skill)
3. [Testing Your Skill](#testing-your-skill)
4. [Security Review](#security-review)
5. [Packaging the Skill](#packaging-the-skill)
6. [Deployment](#deployment)
   - [Claude Code (Local)](#claude-code-local)
   - [Claude Web/Desktop](#claude-webdesktop)
7. [Updating Skills](#updating-skills)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Overview

The Claude Skills ecosystem provides a structured approach to building reusable skills. The workflow follows these stages:

```
Create → Test → Secure → Package → Deploy
```

**Key Principles**:
- **Single Source of Truth**: All skills stored in git (`skills/` directory)
- **Validation First**: Skills must pass security and testing before deployment
- **Repeatable Process**: Standardized workflow for all skills
- **Human Review**: Meta-skills assist but don't replace human judgment

---

## Creating a New Skill

### 1. Choose Skill Type

Determine which category your skill belongs to:

- **meta**: Skills that analyze/manage other skills (e.g., `skill_security_analyzer`)
- **domain**: Domain-specific skills (e.g., `typescript_error_explainer`, HoloLoom operations)
- **utility**: General-purpose utilities

### 2. Create Skill Directory

```bash
# For domain-specific skill
mkdir -p skills/domain/my_new_skill

# For meta-skill
mkdir -p skills/meta/my_meta_skill
```

### 3. Copy Template

```bash
cp skills/templates/skill.markdown.template skills/domain/my_new_skill/skill.markdown
```

### 4. Fill in Template

Open `skills/domain/my_new_skill/skill.markdown` and fill in all sections:

**Required Sections**:
- [ ] Metadata (name, version, author, tags)
- [ ] Description (short + detailed)
- [ ] Required Capabilities
- [ ] Dependencies
- [ ] Input Schema
- [ ] Output Schema
- [ ] Prompt Template
- [ ] Examples (minimum 2-3)
- [ ] Testing Checklist
- [ ] Security Considerations

**Tips**:
- Be specific in Prompt Template instructions
- Provide diverse examples (basic, edge case, error)
- Include realistic input/output examples
- Document all capabilities needed

### 5. Add Examples from Real Usage (Recommended)

If this skill emerged from user interactions (via `continuous_learning_capture`), include real examples:

```json
{
  "input": {"field": "actual user input"},
  "output": {"result": "actual successful output"},
  "source": "User interaction 2025-01-18T10:30:00Z"
}
```

---

## Testing Your Skill

### Manual Testing

1. **Test Prompt Clarity**: Can you execute the prompt and get expected results?
2. **Test Examples**: Do all examples work correctly?
3. **Test Edge Cases**: What happens with invalid/unexpected input?
4. **Test Performance**: Does it complete in reasonable time?

### Automated Testing (Using `skill_tester`)

```bash
# Run skill_tester meta-skill (once implemented)
python scripts/run_skill_tester.py skills/domain/my_new_skill

# Or use Claude Code directly
# Invoke skill_tester skill with your skill as input
```

**Expected Output**:
```json
{
  "test_summary": {
    "total_tests": 3,
    "passed": 3,
    "failed": 0
  },
  "quality_score": 0.85,
  "ready_for_deployment": true
}
```

### Quality Checklist

Before proceeding, verify:
- [ ] All examples execute correctly
- [ ] Error handling is graceful
- [ ] Output matches declared schema
- [ ] Performance is acceptable (<5s for simple tasks)
- [ ] Documentation is clear and complete

---

## Security Review

### Automated Security Analysis

Run the `skill_security_analyzer` meta-skill:

```bash
# Via build script (automatically runs security check)
python scripts/build_skill.py skills/domain/my_new_skill --validate-only

# Or manually invoke skill_security_analyzer
# Use Claude Code to run the security analyzer skill
```

**Security Categories Checked**:
1. **Prompt Injection**: Can user input override skill instructions?
2. **Data Leaks**: Could the skill expose sensitive data?
3. **Privilege Escalation**: Excessive capability requests?
4. **Unsafe Operations**: Risky file/network operations?

### Review Security Report

```json
{
  "overall_risk": "low|medium|high|critical",
  "vulnerabilities": [
    {
      "category": "prompt_injection",
      "severity": "medium",
      "description": "...",
      "recommendation": "Wrap user input in XML tags"
    }
  ],
  "safe_for_deployment": true
}
```

### Fix Security Issues

If vulnerabilities found:
1. Review recommendations
2. Apply suggested fixes
3. Re-run security analyzer
4. Repeat until `safe_for_deployment: true`

**Critical**: Never deploy skills with HIGH or CRITICAL severity vulnerabilities.

---

## Packaging the Skill

### Validate Skill

```bash
# Validate without building
python scripts/build_skill.py skills/domain/my_new_skill --validate-only
```

**Output**:
```
✓ Validation passed for my_new_skill
  Warnings: 0
```

### Build Packages

```bash
# Build .zip and .skill packages
python scripts/build_skill.py skills/domain/my_new_skill
```

**Output**:
```
✓ my_new_skill built successfully
Output files:
  → skills/dist/my_new_skill-1.0.0.zip
  → skills/dist/my_new_skill-1.0.0.skill
```

### Build All Skills

```bash
# Build all skills in skills/ directory
python scripts/build_skill.py --all

# Build summary shows success/failures
```

### Package Contents

Each package contains:
- `skill.markdown` - Main skill definition
- `manifest.json` - Metadata (auto-generated)
- Any additional files in skill directory

---

## Deployment

### Claude Code (Local)

Claude Code loads skills from `.claude/skills/` in your project or global config.

#### Option 1: Copy Skill Package

```bash
# Copy to project .claude/skills/ directory
mkdir -p .claude/skills
cp skills/dist/my_new_skill-1.0.0.skill .claude/skills/

# Or unzip and copy directory
unzip skills/dist/my_new_skill-1.0.0.zip -d .claude/skills/my_new_skill
```

#### Option 2: Symlink (Development)

For active development, symlink the source directory:

```bash
ln -s $(pwd)/skills/domain/my_new_skill .claude/skills/my_new_skill
```

**Benefit**: Changes immediately reflected without rebuilding.

#### Verify Installation

```bash
# Check if skill is recognized
ls .claude/skills/

# Test skill via Claude Code
# Invoke the skill in a Claude Code session
```

### Claude Web/Desktop

Claude Web and Desktop use the web interface for skill management.

#### Upload Skill Package

1. **Navigate to Capabilities**:
   - Open Claude Web/Desktop
   - Go to Settings → Capabilities → Skills

2. **Upload Skill**:
   - Click "Upload Skill" or "Add Skill"
   - Select `skills/dist/my_new_skill-1.0.0.skill` or `.zip`
   - Confirm upload

3. **Verify Installation**:
   - Skill should appear in skills list
   - Status should show "Active" or "Enabled"

#### Enable/Disable Skills

- Toggle skills on/off in Capabilities → Skills UI
- Disable unused skills to reduce context/token usage

#### Update Skills

To update an existing skill:
1. Build new version (increment version in skill.markdown)
2. Upload new package (may overwrite or create new version)
3. Old version may be archived or replaced (depends on Claude implementation)

---

## Updating Skills

### Version Management

Follow semantic versioning:
- **Major** (1.0.0 → 2.0.0): Breaking changes (schema, capabilities)
- **Minor** (1.0.0 → 1.1.0): New features, backward compatible
- **Patch** (1.0.0 → 1.0.1): Bug fixes, documentation

### Update Workflow

1. **Update skill.markdown**:
   - Increment version
   - Update "Last Updated" date
   - Add entry to Changelog section

2. **Test Changes**:
   ```bash
   python scripts/build_skill.py skills/domain/my_new_skill --validate-only
   ```

3. **Rebuild Package**:
   ```bash
   python scripts/build_skill.py skills/domain/my_new_skill
   ```

4. **Redeploy**:
   - Claude Code: Copy new package or refresh symlink
   - Claude Web: Upload new version

5. **Git Commit**:
   ```bash
   git add skills/domain/my_new_skill/
   git commit -m "feat(skills): Update my_new_skill to v1.1.0"
   git push
   ```

---

## Troubleshooting

### Common Issues

#### Validation Fails

**Problem**: `build_skill.py` reports validation errors

**Solutions**:
1. Check error messages for missing sections
2. Ensure all required metadata fields are filled
3. Verify JSON schemas are valid
4. Run `--validate-only` for detailed errors

#### Skill Not Recognized

**Problem**: Claude doesn't recognize installed skill

**Solutions**:
- **Claude Code**:
  - Check `.claude/skills/` path is correct
  - Verify `manifest.json` is present
  - Restart Claude Code session
- **Claude Web**:
  - Verify upload succeeded
  - Check skill is enabled in Capabilities
  - Refresh browser

#### Skill Execution Errors

**Problem**: Skill fails during execution

**Solutions**:
1. Review error logs
2. Test prompt template manually
3. Check capability requirements are met
4. Simplify prompt and add back complexity gradually
5. Run `skill_tester` for automated diagnostics

#### Performance Issues

**Problem**: Skill is slow or uses too many tokens

**Solutions**:
1. Run `token_budget_adviser` for optimization suggestions
2. Compress verbose instructions
3. Reduce number of examples in prompt
4. Simplify output schema

### Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: Report bugs at GitHub issues
- **Examples**: Study existing meta-skills for patterns
- **Community**: (Add community forum link if applicable)

---

## Best Practices

### Skill Design

1. **Single Responsibility**: One skill does one thing well
2. **Clear Scope**: Well-defined boundaries (not too broad, not too narrow)
3. **Reusable**: Generalizable beyond specific use case
4. **Composable**: Can work with other skills via dependencies

### Prompt Engineering

1. **Be Specific**: Clear, unambiguous instructions
2. **Use Structure**: XML tags, JSON schemas for input boundaries
3. **Provide Context**: Explain what the skill does and why
4. **Show Examples**: Include 2-3 diverse examples in prompt
5. **Error Handling**: Specify how to handle edge cases

### Security

1. **Principle of Least Privilege**: Request only needed capabilities
2. **Input Sanitization**: Wrap user input in delimiters (XML tags, JSON)
3. **No Hardcoded Secrets**: Use environment variables or secure config
4. **Output Filtering**: Don't leak internal system details
5. **Regular Audits**: Re-run security analyzer periodically

### Testing

1. **Test Early**: Test as you develop, not at the end
2. **Test Realistically**: Use actual user queries as test cases
3. **Test Edge Cases**: Invalid input, empty input, extremely large input
4. **Test Performance**: Ensure acceptable latency/token usage
5. **Automate**: Use `skill_tester` for regression testing

### Documentation

1. **Keep Updated**: Update docs when changing skill
2. **Be Concise**: Clear and brief > verbose
3. **Provide Examples**: Real examples > hypothetical
4. **Timestamp**: Always datestamp updates
5. **Link Context**: Reference related skills/docs

### Version Control

1. **Atomic Commits**: One logical change per commit
2. **Descriptive Messages**: Explain what and why
3. **Tag Releases**: Git tag for major versions
4. **Changelog**: Maintain comprehensive changelog in skill.markdown
5. **Branch Strategy**: Feature branches for new skills, main for stable

### Maintenance

1. **Monitor Usage**: Track skill invocations and success rates
2. **Gather Feedback**: Listen to user complaints/suggestions
3. **Iterate**: Continuous improvement based on usage data
4. **Deprecate Gracefully**: Warn users before removing skills
5. **Archive**: Keep old versions for reference/rollback

---

## Workflow Example (End-to-End)

Here's a complete example of creating a new skill:

```bash
# 1. Create skill directory
mkdir -p skills/domain/typescript_error_explainer

# 2. Copy template
cp skills/templates/skill.markdown.template \
   skills/domain/typescript_error_explainer/skill.markdown

# 3. Edit skill.markdown (use your editor)
# Fill in metadata, prompt, examples, etc.

# 4. Validate
python scripts/build_skill.py \
  skills/domain/typescript_error_explainer --validate-only

# 5. Fix any validation errors, then test manually
# Test examples, check output quality

# 6. Build package
python scripts/build_skill.py \
  skills/domain/typescript_error_explainer

# 7. Deploy to Claude Code
cp skills/dist/typescript_error_explainer-1.0.0.skill \
   .claude/skills/

# 8. Test in Claude Code session
# Invoke skill and verify it works

# 9. Commit to git
git add skills/domain/typescript_error_explainer/
git commit -m "feat(skills): Add TypeScript error explainer skill"
git push

# 10. Upload to Claude Web (optional)
# Use web UI to upload .skill package
```

---

## Advanced Topics

### Skill Dependencies

If your skill depends on another skill:

1. **Declare in skill.markdown**:
   ```markdown
   **Required Skills**:
   - `skill_security_analyzer` - Used for security validation
   ```

2. **Ensure dependency is installed** before deploying your skill

3. **Call dependency** in your prompt template:
   ```markdown
   Before proceeding, run the `skill_security_analyzer` skill to validate...
   ```

### HoloLoom Integration

For skills that use HoloLoom's memory/RAG/alignment systems:

1. **Mark in skill.markdown**:
   ```markdown
   **HoloLoom Integration**:
   - [x] Uses HoloLoom memory system
   - [x] Uses HoloLoom RAG
   ```

2. **Document integration points**:
   ```markdown
   This skill uses HoloLoom's memory system to retrieve similar past
   interactions and provide context-aware recommendations.
   ```

3. **Handle graceful degradation** if HoloLoom unavailable

### Continuous Learning Integration

For skills created via `continuous_learning_capture`:

1. **Preserve real examples** from user interactions
2. **Mark auto-generated sections**:
   ```markdown
   # AUTO-GENERATED from user sessions 2025-01-10 to 2025-01-18
   # Reviewed and approved by: [Your Name]
   ```
3. **Review carefully** before deploying auto-generated skills

### Multi-File Skills

For complex skills with supporting files:

```
skills/domain/my_complex_skill/
  skill.markdown          # Main skill definition
  helpers/
    utils.py             # Helper utilities
  data/
    config.json          # Configuration
  tests/
    test_skill.py        # Unit tests
```

All files will be packaged in .skill/.zip automatically.

---

## Next Steps

- **Create your first skill**: Follow this workflow to create a domain-specific skill
- **Explore meta-skills**: Study existing meta-skills for patterns
- **Integrate with HoloLoom**: Leverage memory/RAG/alignment for advanced skills
- **Contribute**: Share useful skills with the community

---

## Appendix

### File Structure Reference

```
skills/
  meta/                          # Meta-skills (manage other skills)
    skill_security_analyzer/
      skill.markdown
      manifest.json             # Auto-generated
    skill_tester/
    skill_gap_analyzer/
    token_budget_adviser/
    continuous_learning_capture/

  domain/                        # Domain-specific skills
    typescript_error_explainer/
      skill.markdown
    [your_skill]/

  templates/
    skill.markdown.template      # Template for new skills

  dist/                          # Built packages (auto-generated)
    skill_security_analyzer-1.0.0.zip
    skill_security_analyzer-1.0.0.skill

  proposals/                     # Auto-generated skill proposals
    [proposed_skill]/            # From continuous_learning_capture
```

### Build Script Reference

```bash
# Build single skill
python scripts/build_skill.py <path>

# Validate only
python scripts/build_skill.py <path> --validate-only

# Build all skills
python scripts/build_skill.py --all

# Custom dist directory
python scripts/build_skill.py <path> --dist-dir custom/output

# Help
python scripts/build_skill.py --help
```

### Metadata Field Reference

All required fields in skill.markdown metadata:

- `name`: Skill identifier (lowercase, underscores)
- `version`: Semantic version (e.g., 1.0.0)
- `author`: Creator name or team
- `created`: YYYY-MM-DD
- `last_updated`: YYYY-MM-DD
- `category`: meta|domain|utility
- `tags`: Comma-separated keywords

---

**Document Version**: 1.0
**Last Updated**: 2025-01-18
**Maintained By**: HoloLoom Team
