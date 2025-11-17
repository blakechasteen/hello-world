# 📓 EdWIN Jupyter Notebooks

**Interactive code-along lessons for HoloLoom**

Learn HoloLoom hands-on with Jupyter notebooks that integrate with your progress tracking!

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install jupyter notebook
```

Or install JupyterLab for a more modern experience:

```bash
pip install jupyterlab
```

### Launch Jupyter

```bash
cd edwin_tutor/ui/notebooks
jupyter notebook
```

Or with JupyterLab:

```bash
jupyter lab
```

Your browser will open automatically showing the notebook list.

---

## 📚 Available Lessons

### Lesson 1: What is HoloLoom?
**File:** `Lesson_01_What_is_HoloLoom.ipynb`
**Time:** 5 minutes
**XP:** 50

**What you'll learn:**
- What makes HoloLoom different
- Key concepts (memory shards, knowledge graph, Thompson Sampling)
- Configuration modes (BARE, FAST, FUSED)
- Interactive quiz to test understanding

**Features:**
- ✅ Markdown explanations
- 📝 Interactive quiz cells
- 🎯 Automatic progress tracking
- ⭐ XP rewards on completion

### Lesson 2: Your First Query
**File:** `Lesson_02_Your_First_Query.ipynb`
**Time:** 10 minutes
**XP:** 100

**What you'll learn:**
- The 5-step HoloLoom pattern
- Creating memory shards
- Configuring HoloLoom
- Writing queries
- Understanding responses

**Features:**
- ✅ Step-by-step code examples
- 🧩 Hands-on coding challenge
- 📝 Interactive quiz
- 💡 Progressive hints
- 🎯 Challenge points (20 XP)

---

## 🎓 How to Use

### 1. Start with Lesson 1

Open `Lesson_01_What_is_HoloLoom.ipynb` and click through the cells in order.

### 2. Run Cells Sequentially

- **Read** the markdown cells (explanations)
- **Run** the code cells by pressing `Shift+Enter`
- **Answer** quiz questions by editing the variables
- **Complete** challenges by writing code

### 3. Track Your Progress

The notebooks automatically integrate with EdWIN's progress tracker! Your XP, level, and badges sync across:
- ✅ Jupyter Notebooks (this interface)
- ✅ Terminal UI (`edwin.py`)
- ✅ Web UI (`ui/web/server.py`)
- ✅ Markdown docs

All progress is saved to `.edwin_progress.json` in the `edwin_tutor` directory.

### 4. Run the "Complete Lesson" Cell

At the end of each notebook, run the completion cell to:
- Mark the lesson as complete
- Earn XP rewards
- Check for level-ups
- Unlock badges

---

## ✨ Features

### Interactive Learning
- **Code cells** - Run actual Python code
- **Markdown cells** - Rich explanations with formatting
- **Quizzes** - Instant feedback on answers
- **Challenges** - Write code and validate solutions

### Progress Integration
- **Automatic tracking** - Progress saves as you complete lessons
- **XP system** - Earn experience points
- **Leveling** - Progress from Level 1 to Level 10
- **Badges** - Unlock achievements
- **Cross-platform** - Progress syncs across all EdWIN interfaces

### Visual Learning
- **Syntax highlighting** - Code is beautifully formatted
- **Rich output** - See results immediately
- **Emoji indicators** - Visual cues (✅ ❌ 💡 🎉)
- **Structured content** - Easy to follow, step-by-step

---

## 🎯 Example Workflow

```bash
# 1. Launch Jupyter
cd edwin_tutor/ui/notebooks
jupyter notebook

# 2. Open Lesson_01_What_is_HoloLoom.ipynb

# 3. Run each cell with Shift+Enter

# 4. Complete quiz and challenges

# 5. Run the "Complete Lesson" cell

# 6. Move to Lesson_02_Your_First_Query.ipynb

# 7. Repeat!
```

---

## 🛠️ Troubleshooting

### Jupyter won't launch

**Problem:** `command not found: jupyter`

**Solution:**
```bash
pip install jupyter notebook
# Or
pip install jupyterlab
```

### Imports failing

**Problem:** `ModuleNotFoundError: No module named 'core'`

**Solution:** The notebooks add the parent directories to the Python path automatically. If you still see errors:

1. Check you're running from `edwin_tutor/ui/notebooks/`
2. Verify the first code cell ran successfully (shows "✅ Setup complete!")
3. Try restarting the kernel (Kernel → Restart)

### Progress not saving

**Problem:** Completed lessons don't stay completed

**Solution:**
- Ensure you ran the "Complete Lesson" cell at the end
- Check that `.edwin_progress.json` exists in `edwin_tutor/` directory
- Verify write permissions

### Quiz answers not checking

**Problem:** Changing answers doesn't show correct/incorrect

**Solution:**
- Make sure you're changing the variable (e.g., `answer_1 = "B"`)
- Re-run the cell after changing your answer (Shift+Enter)

---

## 📖 Learning Tips

### Best Practices

- **Run cells in order** - Don't skip ahead
- **Read everything** - Don't just execute code blindly
- **Experiment** - Try changing values and re-running
- **Take notes** - Add your own markdown cells with `Esc` → `M`
- **Ask questions** - Use `# comments` to mark confusion points

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Shift+Enter` | Run cell and move to next |
| `Ctrl+Enter` | Run cell in place |
| `Esc` → `A` | Insert cell above |
| `Esc` → `B` | Insert cell below |
| `Esc` → `M` | Convert to markdown |
| `Esc` → `Y` | Convert to code |
| `Esc` → `D D` | Delete cell |

### Getting Stuck?

1. **Check the hints** - Challenges have progressive hints
2. **Re-read explanations** - Go back to the markdown cells
3. **Try the terminal UI** - Different perspective on same content
4. **Check the web UI** - Visual interface might help
5. **Review the lesson JSON** - Raw content in `content/beginner/*.json`

---

## 🚧 Coming Soon

### Future Lessons (in notebook format)

- **Lesson 3:** Understanding Memory Shards
- **Lesson 4:** Configuration Modes Deep Dive
- **Lesson 5:** Knowledge Graph Basics
- **Lesson 6:** Thompson Sampling Explained
- **Lesson 7:** Advanced Querying
- **Lesson 8:** Recursive Learning
- **Lesson 9:** Custom Adapters
- **Lesson 10:** Building Your First Project

### Enhanced Features

- ✨ **Video embeds** - Watch tutorials in-notebook
- ✨ **Live code execution** - Run HoloLoom code directly (when installed)
- ✨ **Visualizations** - Graph your knowledge graph
- ✨ **Export options** - Save notebooks as PDFs
- ✨ **Auto-save progress** - Save on every cell run

---

## 🎨 Customization

### Add Your Own Notes

Create markdown cells to take notes:

1. Press `Esc` to enter command mode
2. Press `B` to create a new cell below
3. Press `M` to convert it to markdown
4. Type your notes
5. Press `Shift+Enter` to render

### Create Practice Notebooks

You can duplicate notebooks and experiment:

1. Right-click notebook in file browser
2. Select "Duplicate"
3. Rename to `Practice_Lesson_XX.ipynb`
4. Experiment freely!

---

## 🤝 Contributing

Want to create more notebook lessons?

1. Fork the repository
2. Create a new `.ipynb` file
3. Follow the format of existing lessons:
   - Setup cell (imports, progress tracking)
   - Markdown explanations
   - Code examples
   - Quiz cells
   - Challenge cells
   - Completion cell
4. Test it thoroughly
5. Submit a PR!

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - Same as HoloLoom

---

## 🙏 Acknowledgments

Built with love for interactive, hands-on learning!

**Happy coding! 🚀**

---

**Status:** ✅ 2 lessons available, more coming soon!

*Notebooks are the best way to learn by doing. Every line of code you run makes you better!*
