# 📚 EdWIN Markdown Documentation

**Quick reference guides and FAQs for HoloLoom**

Static markdown documentation for fast lookup and offline reading.

---

## 📖 Available Documents

### 1. Quick Reference (`01_Quick_Reference.md`)

**What it covers:**
- Core concepts at a glance
- The 5-step basic pattern
- Configuration modes comparison
- Memory shard examples
- Query patterns
- Common operations
- Troubleshooting tips
- Code cheat sheet

**Best for:**
- Quick lookups while coding
- Remembering syntax
- Checking configuration options
- Finding code snippets

**Read time:** 5-10 minutes (reference)

---

### 2. FAQ (`02_FAQ.md`)

**What it covers:**
- General questions about EdWIN & HoloLoom
- Getting started guide
- Progress tracking explained
- Badge system
- Technical questions (async, imports, etc.)
- Troubleshooting common issues
- Best practices
- Community resources

**Best for:**
- Answering "How do I...?" questions
- Troubleshooting errors
- Understanding concepts
- Learning best practices

**Read time:** 10-15 minutes (scan for your question)

---

## 🎯 How to Use These Docs

### As Quick Reference

Keep these open while coding:

```bash
# In one terminal: Run EdWIN
python edwin.py

# In another terminal/window: Read docs
cat ui/markdown/01_Quick_Reference.md | less
```

Or open in your editor:
```bash
code ui/markdown/01_Quick_Reference.md
```

### As Learning Material

Read sequentially:
1. Start with `01_Quick_Reference.md` to get overview
2. Complete Lesson 1 and 2 in any interface
3. Return to quick reference as needed
4. Use `02_FAQ.md` when stuck

### As Offline Resource

These are plain markdown files - they work without internet:
- Download repository
- Read locally
- Copy/paste code examples
- No dependencies needed

---

## 📁 File Structure

```
ui/markdown/
├── README.md              # This file
├── 01_Quick_Reference.md  # Fast lookup guide
└── 02_FAQ.md             # Common questions
```

---

## ✨ Features

### Advantages of Markdown Docs

✅ **Fast** - No loading time, instant search
✅ **Offline** - Works without internet
✅ **Portable** - Read anywhere (terminal, editor, GitHub)
✅ **Searchable** - Use `grep`, find in page, etc.
✅ **Copy-friendly** - Easy to copy code examples
✅ **Version controlled** - Track changes with git

### Comparison to Other Interfaces

| Feature | Markdown | Terminal UI | Web UI | Notebooks |
|---------|----------|-------------|---------|-----------|
| **Speed** | Instant | Fast | Medium | Medium |
| **Interactive** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Offline** | ✅ Yes | ✅ Yes | ❌ Server needed | ❌ Jupyter needed |
| **Progress tracking** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Code execution** | ❌ No | ❌ No | Limited | ✅ Yes |
| **Quick lookup** | ✅ Best | Good | Good | Good |

**Use markdown docs when:**
- You need to look something up fast
- You're offline
- You want to read without running anything
- You need to copy code examples

---

## 🔍 Searching the Docs

### Using grep (terminal)

```bash
# Find all mentions of "Thompson Sampling"
grep -r "Thompson Sampling" ui/markdown/

# Find configuration examples
grep -A 5 "Config\." ui/markdown/01_Quick_Reference.md

# Find all code blocks
grep -B 1 "```python" ui/markdown/*.md
```

### Using your editor

**VS Code:**
- Press `Ctrl+F` (or `Cmd+F` on Mac)
- Search across files with `Ctrl+Shift+F`

**vim:**
```
/search term
n (next match)
N (previous match)
```

**GitHub:**
- Use built-in search in repository
- Click "Go to file" → type filename
- Use browser find-in-page

---

## 🎨 Reading Tips

### In Terminal

**With `less`:**
```bash
less ui/markdown/01_Quick_Reference.md
# Press / to search
# Press q to quit
```

**With `cat`:**
```bash
cat ui/markdown/01_Quick_Reference.md
```

**With syntax highlighting (bat):**
```bash
# Install bat: brew install bat (Mac) or apt install bat (Linux)
bat ui/markdown/01_Quick_Reference.md
```

### In Browser

**Using GitHub:**
1. Navigate to `edwin_tutor/ui/markdown/`
2. Click on any `.md` file
3. GitHub renders it beautifully!

**Locally:**
```bash
# Install markdown renderer (optional)
npm install -g markdown-it
markdown-it 01_Quick_Reference.md > reference.html
open reference.html
```

### In Editor

**VS Code:**
- Open file
- Press `Ctrl+Shift+V` to preview
- Or click preview icon in top right

**Most editors:**
- Install markdown preview extension
- Open file
- Use preview command

---

## 📝 Document Conventions

### Code Examples

**Runnable code:**
```python
# This is actual working code
from HoloLoom.config import Config
config = Config.fast()
```

**Conceptual code:**
```python
# This shows the pattern (may need adaptation)
shard = MemoryShard(text="...", source="...")
```

**Terminal commands:**
```bash
python edwin.py
```

### Symbols

| Symbol | Meaning |
|--------|---------|
| ✅ | Do this / Correct / Completed |
| ❌ | Don't do this / Wrong |
| 💡 | Tip or insight |
| ⚠️ | Warning / Important |
| 🎯 | Recommended approach |
| 🔧 | Configuration option |

### Tables

All tables use markdown format:
```markdown
| Column 1 | Column 2 |
|----------|----------|
| Value A  | Value B  |
```

---

## 🚧 Coming Soon

### Future Documentation

- **03_Advanced_Topics.md** - Deep dives into complex concepts
- **04_API_Reference.md** - Complete API documentation
- **05_Cookbook.md** - Common patterns and recipes
- **06_Architecture.md** - System design and internals
- **07_Contributing.md** - How to add lessons and features

### Enhancements

- ✨ **Diagrams** - Visual explanations (Mermaid or ASCII art)
- ✨ **Examples repository** - Downloadable code examples
- ✨ **Video links** - Embedded tutorial videos
- ✨ **Interactive playgrounds** - Try code in browser

---

## 🤝 Contributing

Want to improve the docs?

### Fix a Typo
1. Edit the `.md` file
2. Submit a PR

### Add an Example
1. Add to relevant section
2. Test the code
3. Submit a PR

### Add a Document
1. Follow existing format
2. Update this README
3. Submit a PR

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - Same as HoloLoom

---

## 🙏 Acknowledgments

Inspired by:
- [Rust Book](https://doc.rust-lang.org/book/) - Great learning resource
- [Django Docs](https://docs.djangoproject.com/) - Comprehensive reference
- [FastAPI Docs](https://fastapi.tiangolo.com/) - Clear examples

**Happy learning! 📚**

---

**Status:** ✅ 2 documents available (Quick Reference + FAQ)
**Next:** Advanced Topics, API Reference, Cookbook

*Good documentation is like a good friend - always there when you need it.*
