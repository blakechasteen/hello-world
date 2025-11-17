# 🌐 EdWIN Web UI

**Browser-based interactive learning platform for HoloLoom**

A modern, accessible web interface for EdWIN Tutor with drag-and-drop navigation, interactive quizzes, code challenges, and real-time progress tracking.

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install flask flask-cors
```

### Run the Server

```bash
cd edwin_tutor/ui/web
python server.py
```

The server will start on `http://localhost:5000`

Open your browser and navigate to: **http://localhost:5000**

---

## ✨ Features

### 🎓 Interactive Lessons
- **Rich content rendering** - Markdown with syntax highlighting
- **Code examples** - Copy-paste ready examples
- **Visual navigation** - Easy-to-browse lesson library
- **Prerequisites tracking** - Locked lessons unlock as you progress

### 📝 Smart Quizzes
- **Instant feedback** - See correct answers immediately
- **Detailed explanations** - Understand why an answer is correct
- **Score tracking** - Monitor your quiz performance

### 🧩 Code Challenges
- **In-browser coding** - Write and submit code directly
- **Progressive hints** - 3-level hint system
- **Validation** - Instant feedback on your solutions

### 📊 Progress Dashboard
- **Real-time stats** - Level, XP, lessons completed
- **Visual progress bar** - See XP progress to next level
- **Badge showcase** - Display your achievements
- **Completion tracking** - Track overall progress

### 🏆 Gamification
- **XP System** - Earn experience points for completing lessons
- **Leveling** - Progress from Level 1 to Level 10
- **Badges** - Unlock achievements as you learn
- **Level-up celebrations** - Modal notifications when you level up

---

## 🎨 User Interface

### Layout

```
┌─────────────────────────────────────────────────┐
│  🎓 EdWIN Tutor                                 │
│  Your Interactive Guide to HoloLoom             │
├─────────────────────────────────────────────────┤
│  Progress: Level 2 | 150 XP | 3 lessons done   │
│  ████████░░░░░░ 60%                             │
├────────────┬────────────────────────────────────┤
│ Navigation │  Main Content                      │
│            │                                     │
│ 📚 Lessons │  ┌──────────────────────────────┐  │
│ 📊Progress │  │  Lesson Title                │  │
│ 🏆 Badges  │  │  📚 beginner | ⏱️ 10 min     │  │
│            │  ├──────────────────────────────┤  │
│ Lessons:   │  │  Lesson content here...      │  │
│            │  │                              │  │
│ 🌱Beginner │  │  [Quiz Section]              │  │
│  ✓ 01: ... │  │  [Challenge Section]         │  │
│  → 02: ... │  │                              │  │
│  🔒 03: ..│  │  [Complete Lesson Button]    │  │
│            │  └──────────────────────────────┘  │
└────────────┴────────────────────────────────────┘
```

### Color Scheme

- **Primary** - Purple gradient (`#667eea` → `#764ba2`)
- **Success** - Green (`#10b981`)
- **Warning** - Amber (`#f59e0b`)
- **Danger** - Red (`#ef4444`)
- **Gray Scale** - Modern gray palette

### Responsive Design

- **Desktop** - Full sidebar + content layout
- **Tablet/Mobile** - Stacked layout, collapsible sidebar
- **Accessibility** - WCAG 2.1 AA compliant colors

---

## 🔌 API Endpoints

The Flask server provides these REST endpoints:

### Lessons

- `GET /api/lessons` - Get all lessons grouped by difficulty
- `GET /api/lessons/<id>` - Get detailed lesson content

### Progress

- `GET /api/progress` - Get learner's current progress
- `POST /api/progress/start-lesson` - Mark lesson as started
- `POST /api/progress/complete-lesson` - Mark lesson as complete
- `POST /api/progress/complete-challenge` - Mark challenge as complete
- `POST /api/progress/use-hint` - Record hint usage

### Badges

- `GET /api/badges` - Get all badges with earned status

---

## 📁 File Structure

```
ui/web/
├── server.py              # Flask backend (REST API)
├── static/                # Frontend assets
│   ├── index.html        # Main HTML page
│   ├── style.css         # Modern, accessible styling
│   └── app.js            # JavaScript interactivity
└── README.md             # This file
```

---

## 🛠️ How It Works

### Architecture

```
Browser (HTML/CSS/JS)
    ↓ HTTP GET/POST
Flask Server (server.py)
    ↓ Python imports
EdWIN Core (lesson.py, progress.py)
    ↓ Read/Write
.edwin_progress.json + content/*.json
```

### Data Flow

1. **Page Load**:
   - Browser requests `/`
   - Server serves `static/index.html`
   - `app.js` loads and calls API endpoints

2. **Load Lessons**:
   - `app.js` → `GET /api/lessons`
   - Server → `LessonManager.lessons`
   - Response → Render lesson list in sidebar

3. **Select Lesson**:
   - User clicks lesson
   - `app.js` → `GET /api/lessons/<id>`
   - Server → `LessonManager.get_lesson(id)`
   - Response → Render lesson content

4. **Complete Lesson**:
   - User clicks "Complete Lesson"
   - `app.js` → `POST /api/progress/complete-lesson`
   - Server → `ProgressTracker.mark_lesson_complete()`
   - Server saves to `.edwin_progress.json`
   - Response → Update UI, show level-up modal if applicable

### Progress Sync

All progress is saved to `.edwin_progress.json` and shared across:
- ✅ Terminal UI (`edwin.py`)
- ✅ Web UI (this interface)
- 🔜 Jupyter notebooks
- 🔜 Markdown docs

---

## 🧪 Testing

### Manual Testing Checklist

- [ ] Server starts without errors
- [ ] Homepage loads successfully
- [ ] Lesson list renders with correct icons
- [ ] Locked lessons show lock icon
- [ ] Completed lessons show checkmark
- [ ] Clicking lesson loads content
- [ ] Quiz questions render correctly
- [ ] Quiz submission shows correct/incorrect feedback
- [ ] Challenge code input works
- [ ] Hint buttons reveal hints progressively
- [ ] "Complete Lesson" awards XP
- [ ] Progress bar updates after completing lesson
- [ ] Level-up modal shows when leveling up
- [ ] Progress view shows correct stats
- [ ] Badges view shows earned/unearned badges
- [ ] Navigation between views works

### Test in Different Browsers

- Chrome/Edge (Chromium)
- Firefox
- Safari (if on Mac)

---

## 🎯 Usage Examples

### Starting the Server

```bash
# From edwin_tutor/ui/web directory
python server.py
```

Output:
```
🎓 Starting EdWIN Web Server...
📚 Open http://localhost:5000 in your browser
 * Running on http://0.0.0.0:5000
```

### Using the Interface

1. **Browse Lessons** - Click lessons in the sidebar
2. **Read Content** - Scroll through lesson content
3. **Take Quiz** - Select answers and click "Submit Answers"
4. **Solve Challenges** - Write code and click "Submit Solution"
5. **Get Hints** - Click "Show Hint" for progressive help
6. **Complete Lesson** - Click "Complete Lesson" to earn XP
7. **Track Progress** - Click "My Progress" to see stats
8. **View Badges** - Click "Badges" to see achievements

---

## 🎨 Customization

### Change Colors

Edit `static/style.css` root variables:

```css
:root {
    --primary: #3b82f6;      /* Your primary color */
    --success: #10b981;      /* Success color */
    --warning: #f59e0b;      /* Warning color */
}
```

### Change Port

Edit `server.py`:

```python
app.run(debug=True, host='0.0.0.0', port=8080)  # Your port
```

### Add Custom Endpoints

Add new routes to `server.py`:

```python
@app.route('/api/custom', methods=['GET'])
def custom_endpoint():
    return jsonify({'message': 'Custom data'})
```

---

## 🐛 Troubleshooting

### Server won't start

**Problem**: `ImportError: No module named flask`

**Solution**:
```bash
pip install flask flask-cors
```

### Lessons not loading

**Problem**: Empty lesson list

**Solution**:
- Check that `content/beginner/*.json` files exist
- Verify JSON is valid (use a JSON validator)
- Check server console for errors

### Progress not saving

**Problem**: Completed lessons don't stay completed

**Solution**:
- Check write permissions for `.edwin_progress.json`
- Verify `ProgressTracker` is saving correctly
- Check server console for save errors

### Quiz/Challenge not working

**Problem**: Clicking buttons does nothing

**Solution**:
- Check browser console for JavaScript errors (F12)
- Verify `app.js` is loaded correctly
- Try hard refresh (Ctrl+Shift+R)

---

## 🚀 Future Enhancements

### Planned Features

- ✨ **Code editor** - Syntax highlighting with CodeMirror/Monaco
- ✨ **Live code execution** - Run Python in browser (Pyodide)
- ✨ **Video tutorials** - Embed video content in lessons
- ✨ **Discussion forum** - Community Q&A integration
- ✨ **Dark mode** - Theme toggle
- ✨ **Offline support** - Service worker caching
- ✨ **Mobile app** - Progressive Web App (PWA)

### Contributing

Want to improve the Web UI? Check out `../../../CONTRIBUTING.md` for guidelines!

---

## 📄 License

MIT License - Same as HoloLoom

---

**Happy learning! 🚀**

*Built with care by developers who believe learning should be visual, interactive, and fun.*
