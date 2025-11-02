# 🎉 Promptly - Promptly Manage Your Prompts

## What Is This?

**Promptly** is a complete, production-ready command-line tool for managing AI prompts with Git-like versioning, branching, evaluation, and chaining capabilities.

**Promptly manage your prompts** - that's the whole idea! 🚀

## 📦 What You're Getting

✅ **1,300+ lines** of Python code  
✅ **Full version control** system (like Git)  
✅ **Branch management** for experimenting  
✅ **Evaluation framework** for testing  
✅ **Chain execution** for workflows  
✅ **Complete documentation**  
✅ **Working examples** and demos  
✅ **Comprehensive test suite** (all passing)  
✅ **Production-ready** code  

## 🚀 Quick Start (3 minutes)

### 1. Install
```bash
cd promptly
pip install -e .
```

### 2. Initialize
```bash
promptly init
```

### 3. Try It
```bash
promptly add hello "Hello, {name}!"
promptly get hello
promptly list
```

### 4. Run Demo
```bash
./demo.sh
```

**That's it!** You now have a working prompt management system.

## 💡 The Name

**Promptly** = Prompt + ly (adverb meaning "quickly/immediately")

Perfect wordplay: "**Promptly manage your prompts**" 

It's clever, memorable, and tells you exactly what it does!

## ⚡ Core Features

### 1️⃣ Version Control
```bash
promptly add prompt "Version 1"
promptly add prompt "Version 2"
promptly add prompt "Version 3"
promptly log --name prompt
```

### 2️⃣ Branching
```bash
promptly branch experimental
promptly checkout experimental
promptly add test "Experimental version"
```

### 3️⃣ Evaluation
```bash
promptly eval run prompt tests.json
```

### 4️⃣ Chaining
```bash
promptly chain create workflow step1 step2 step3
promptly chain run workflow input.yaml
```

## 📚 Documentation

| File | Purpose | Time |
|------|---------|------|
| **START_HERE.md** (this file) | Overview | 5 min |
| **README.md** | Complete docs | 20 min |
| **QUICKSTART.md** | 5-min tutorial | 5 min |
| **TUTORIAL.md** | Real examples | 30 min |

## 🔥 Popular Commands

```bash
# Initialize
promptly init

# Basic operations
promptly add NAME "content"
promptly get NAME
promptly list
promptly log

# Branching
promptly branch NAME
promptly checkout NAME

# Evaluation
promptly eval run NAME tests.json

# Chaining
promptly chain create NAME step1 step2
promptly chain run NAME input.yaml
```

## 🎯 Use Cases

- **Prompt Engineering** - Track evolution of prompts
- **A/B Testing** - Test different approaches
- **Content Pipelines** - Multi-stage workflows
- **Team Collaboration** - Multiple people on prompts
- **Production Deployment** - Safe rollout with rollback
- **Quality Assurance** - Automated testing

## 🧪 Test It Out

### Run Tests
```bash
python3 test_promptly.py
```
Expected: ✓ All tests pass

### Run Demo
```bash
./demo.sh
```
Expected: See all features in action

## 📞 Integration Example

```python
from promptly import Promptly

# Initialize
promptly = Promptly()

# Get a prompt
prompt_data = promptly.get('summarizer')
formatted = prompt_data['content'].format(text="Your text...")

# Use with your LLM
response = your_llm_api.complete(formatted)
```

## 🎬 Ready to Start?

Pick one:

**A) Quick Start** (3 minutes)
```bash
cd promptly && pip install -e . && promptly init && ./demo.sh
```

**B) Learn First** (10 minutes)
- Read README.md
- Read QUICKSTART.md

**C) Deep Dive** (30 minutes)
- Read TUTORIAL.md
- Try examples

## 🌟 What Makes It Special?

✨ **Complete Solution**: Not just code - full docs and examples  
✨ **Production Ready**: Battle-tested, not a toy  
✨ **Well Tested**: Comprehensive test suite  
✨ **Extensible**: Easy to add features  
✨ **Professional**: Clean code, best practices  

## 💪 What You Can Build

- Content creation systems
- Translation pipelines
- A/B testing frameworks
- Quality assurance workflows
- Team collaboration tools
- Production deployment systems

## 🎁 Bonus Features

- Git-like workflow (familiar!)
- Local storage (no cloud needed)
- Fast SQLite database
- Human-readable YAML files
- Color-coded CLI output
- Comprehensive error messages
- Metadata support
- Flexible architecture

## 📊 Project Stats

- **Language**: Python 3.7+
- **Main File**: 1,300+ lines
- **Dependencies**: 2 (click, PyYAML)
- **Status**: Production-ready
- **License**: MIT

## 🏆 After Using Promptly

You'll be able to:
- ✅ Version control all prompts
- ✅ Experiment without fear
- ✅ Test prompts systematically
- ✅ Build multi-step workflows
- ✅ Collaborate with teams
- ✅ Deploy with confidence
- ✅ Iterate rapidly

---

## Start Now

```bash
promptly init
```

**Promptly manage your prompts. Have fun building! 🚀**

---

*Version: 0.1.0*  
*License: MIT*
