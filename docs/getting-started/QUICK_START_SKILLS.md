# Quick Start: Claude Skills System

**Get started with Wave 1 skills in 5 minutes!**

---

## 🚀 Deploy Skills (2 commands)

```bash
# 1. Verify skills are packaged
ls skills/dist/*.skill

# 2. Deploy to Claude Code (local)
cp skills/dist/*.skill ~/.claude/skills/
```

**Done!** Skills are now available in Claude Code.

---

## 💡 Use Skills

### Example 1: HoloLoom RAG Helper

**Query**: "Use the hololoom_rag_helper skill to answer: What is Thompson Sampling?"

**Skill Input**:
```json
{
  "question": "What is Thompson Sampling?",
  "mode": "auto",
  "max_sources": 5
}
```

**Skill Output**:
```json
{
  "answer": "Thompson Sampling is a Bayesian approach...",
  "sources": [...],
  "confidence": 0.92,
  "reasoning_mode": "direct"
}
```

### Example 2: TypeScript Error Explainer

**Query**: "Use typescript_error_explainer to explain error TS2322"

**Skill Input**:
```json
{
  "error_code": "TS2322",
  "error_message": "Type 'string' is not assignable to type 'number'",
  "code_snippet": "let x: number = 'hello';"
}
```

**Skill Output**:
```json
{
  "explanation": "You're trying to assign a string to a number variable...",
  "fix_suggestions": [
    {
      "approach": "Change variable type to string",
      "code_example": "let x: string = 'hello';",
      "explanation": "..."
    }
  ]
}
```

### Example 3: Dockerfile Generator

**Query**: "Use dockerfile_generator to create a Dockerfile for my Node.js app"

**Skill Input**:
```json
{
  "project_structure": {
    "language": "nodejs",
    "files": ["package.json", "server.js"],
    "framework": "express"
  },
  "requirements": ["optimize for size"]
}
```

**Skill Output**:
```json
{
  "dockerfile": "# Multi-stage build...",
  "docker_compose": "version: '3.8'...",
  "best_practices": ["Multi-stage build", "Alpine Linux", ...]
}
```

---

## 🛠️ Create Your Own Skill (5 steps)

### Step 1: Create from Template

```bash
python scripts/create_skill.py my_awesome_skill --category domain
```

### Step 2: Edit skill.markdown

```bash
code skills/domain/my_awesome_skill/skill.markdown
```

Fill in all sections:
- Metadata (name, version, tags)
- Description (short + detailed)
- Input/Output schemas
- Prompt template
- Examples (3+ required)
- Testing checklist

### Step 3: Validate

```bash
python scripts/validate_all_skills.py --category domain
```

Check that all 4 gates pass:
- [1/4] Schema validation
- [2/4] Security analysis
- [3/4] Testing (3+ examples)
- [4/4] Token budget (<1000 tokens)

### Step 4: Package

```bash
python scripts/build_skill.py skills/domain/my_awesome_skill
```

Output: `skills/dist/my_awesome_skill-1.0.0.skill`

### Step 5: Deploy

```bash
cp skills/dist/my_awesome_skill-1.0.0.skill ~/.claude/skills/
```

**Done!** Your skill is ready to use.

---

## 📚 Available Skills

### Wave 1 Domain Skills (6)

| Skill | Purpose | Token Count | Use Case |
|-------|---------|-------------|----------|
| **hololoom_rag_helper** | HoloLoom RAG operations | 447 | Question answering with sources |
| **typescript_error_explainer** | Decode TS errors | 196 | TypeScript debugging |
| **python_debug_assistant** | Analyze tracebacks | 153 | Python debugging |
| **react_performance_optimizer** | React anti-patterns | 206 | React optimization |
| **sql_query_optimizer** | Database performance | 223 | SQL optimization |
| **dockerfile_generator** | Production Dockerfiles | 215 | Docker containerization |

### Meta Skills (5)

| Skill | Purpose |
|-------|---------|
| **continuous_learning_capture** | Pattern mining from sessions |
| **skill_gap_analyzer** | Capability gap detection |
| **skill_security_analyzer** | Security validation |
| **skill_tester** | Automated testing |
| **token_budget_adviser** | Token optimization |

---

## 🔧 Troubleshooting

### Skills not showing up?

```bash
# Check Claude Code skills directory
ls ~/.claude/skills/

# If empty, copy skills
cp skills/dist/*.skill ~/.claude/skills/
```

### Validation failing?

```bash
# Run validation with verbose output
python scripts/validate_all_skills.py --category domain

# Check specific skill
python scripts/build_skill.py skills/domain/my_skill --validate-only
```

### Need to update a skill?

```bash
# 1. Edit skill.markdown
code skills/domain/my_skill/skill.markdown

# 2. Increment version in metadata (1.0.0 → 1.1.0)

# 3. Revalidate
python scripts/validate_all_skills.py

# 4. Rebuild
python scripts/build_skill.py skills/domain/my_skill

# 5. Redeploy
cp skills/dist/my_skill-1.1.0.skill ~/.claude/skills/
```

---

## 📖 Documentation

- **[SKILLS_EXPANSION_ROADMAP.md](SKILLS_EXPANSION_ROADMAP.md)** - Complete 3-wave roadmap
- **[SKILLS_ARCHITECTURE_PATTERNS.md](SKILLS_ARCHITECTURE_PATTERNS.md)** - Architecture patterns
- **[docs/skills_workflow.md](docs/skills_workflow.md)** - Complete workflow guide
- **[WAVE_1_COMPLETE.md](WAVE_1_COMPLETE.md)** - Wave 1 summary

---

## 🎯 Best Practices

### When Creating Skills

1. **Keep skills focused** - One clear purpose per skill
2. **Provide 3+ examples** - Cover happy path, edge case, error
3. **Stay under 700 tokens** - Concise prompts perform better
4. **Include fix suggestions** - Don't just identify problems
5. **Test edge cases** - Empty inputs, malformed data, etc.

### When Using Skills

1. **Match input schema exactly** - Skills expect structured JSON
2. **Check confidence scores** - Low confidence = uncertain answer
3. **Review sources/suggestions** - Skills provide reasoning
4. **Iterate if needed** - Refine inputs based on output

### Token Budget Guidelines

- **Simple skills**: 200-400 tokens (e.g., typescript_error_explainer: 196)
- **Standard skills**: 500-700 tokens (e.g., hololoom_rag_helper: 447)
- **Complex skills**: 800-1000 tokens (hard limit)

**Wave 1 Average**: 240 tokens (excellent!)

---

## 🚀 Next Steps

1. **Try all 6 Wave 1 skills** - Get familiar with capabilities
2. **Create a custom skill** - Use quick-start script
3. **Integrate with projects** - Use skills in real workflows
4. **Prepare for Wave 2** - 8 HoloLoom integration skills coming!

---

## 💬 Support

- **Issues**: https://github.com/blakechasteen/hello-world/issues
- **Documentation**: See [docs/skills_workflow.md](docs/skills_workflow.md)
- **Examples**: All skills have 3+ examples in skill.markdown

---

**Ready to build amazing skills? Let's go! 🚀**

**zero-G ready ✈️**
