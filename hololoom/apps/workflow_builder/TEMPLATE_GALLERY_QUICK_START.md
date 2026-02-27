# Template Gallery - Quick Start Guide

## What Is It?

A modern, beautiful UI for discovering, previewing, and loading pre-built HoloLoom workflows into the workflow builder.

## How to Open

### Option 1: Direct File (Easiest)
```bash
# Windows
start hololoom/web_dashboard/template_gallery.html

# Mac/Linux
open hololoom/web_dashboard/template_gallery.html
```

### Option 2: Local Server (Recommended for Development)
```bash
cd hololoom/web_dashboard
python -m http.server 8000
# Visit: http://localhost:8000/template_gallery.html
```

### Option 3: In VS Code
Right-click `template_gallery.html` → "Open with Live Server"

## What You'll See

### Top Section (Header)
- **Search Bar** (🔍): Find templates by name or description
- **Category Tabs**: Filter by All, Research, CRM, Support, Content, Safety
- **Template Count**: Shows how many templates match your filters

### Main Grid
8 pre-built templates displayed as cards:

```
┌─────────────────────────────────┐
│  🔍  Research Pipeline          │
│  Multi-query research with      │
│  synthesis and refinement       │
│                                  │
│  ★★★ 6 agents ⏱️ 2-5 min       │
│                           [USE] │
└─────────────────────────────────┘
```

Each card shows:
- **Icon**: Category emoji (🔍 = Research, 👥 = CRM, etc.)
- **Name**: Template title
- **Description**: What the template does
- **Complexity**: 1-3 dots (simple to complex)
- **Metadata**: Number of agents, estimated time
- **Tags**: Keywords like "Research", "Safety", etc.
- **Use Button**: Click to preview

## How to Use a Template

### Step 1: Browse or Search
- Click category tabs to filter
- Type in search bar to find specific templates
- Look for "NEW" or "POPULAR" badges

### Step 2: Preview
Click the **"Use"** button on any template card:
- Modal pops up with full details
- Shows workflow diagram (coming soon)
- Displays estimated time, complexity, etc.

### Step 3: Load
Click **"Use Template"** in the modal:
- Template loads into workflow builder
- You can customize it there
- Or run it as-is

## Templates Available

| Template | Category | Complexity | Use For |
|----------|----------|-----------|---------|
| Research Pipeline | Research | ⭐⭐⭐ | Multi-query research |
| Safety-Gated Query | Safety | ⭐⭐ | Safe execution |
| Lead Scoring | CRM | ⭐ | Qualify leads |
| Multi-Factor Scoring | CRM | ⭐⭐⭐ | Advanced scoring |
| Daily Action List | CRM | ⭐⭐ | Task prioritization |
| BDR Outbound | CRM | ⭐⭐ | Sales outreach |
| Support Triage | Support | ⭐⭐ | Ticket routing |
| Content Creation | Content | ⭐⭐⭐ | Generate content |

## Pro Tips

### 1. Use Search
- Type "safety" to find safe workflows
- Type "crmm" (even with typo) and it finds "CRM" templates
- Search works on tags too

### 2. Filter by Category
- Click "Research" to see research workflows
- Click "All Templates" to reset filter
- Tabs are always visible at top

### 3. Check Complexity
- **⭐ (Simple)**: 1-3 agents, <1 min
- **⭐⭐ (Medium)**: 4-6 agents, 1-3 min
- **⭐⭐⭐ (Complex)**: 7+ agents, 2-5 min

### 4. Look for Status
- **NEW**: Latest templates (try these!)
- **POPULAR**: Most-used templates (proven)
- **BETA**: Experimental (feedback appreciated)

### 5. Keyboard Navigation
- Press **Escape** to close preview modal
- Press **Tab** to navigate between templates
- Press **Enter** to open/use templates

## File Locations

```
hololoom/web_dashboard/
├── template_gallery.html         ← Main UI file
├── template_gallery.js           ← Advanced features (optional)
├── TEMPLATE_GALLERY_README.md    ← Full documentation
├── example_workflows/            ← Template JSON files
│   ├── research_pipeline.json
│   ├── safety_gated_query.json
│   ├── crm/
│   │   ├── lead_scoring_simple.json
│   │   ├── multi_factor_scoring.json
│   │   └── daily_action_list.json
│   └── llm/
│       ├── customer_support_triage.json
│       └── content_creation.json
└── workflow_builder.html         ← Where templates load
```

## What's New (Wave 1)

✅ **Complete**:
- Dark theme UI with smooth animations
- 8 pre-built templates with metadata
- Search and category filtering
- Beautiful preview modal
- Responsive design (mobile-friendly)
- Zero external dependencies
- Usage analytics tracking

🚀 **Coming Next (Wave 2)**:
- Workflow diagram visualization
- Template customization wizard
- Import/export functionality
- Template variants (easy/hard modes)
- Recommendations based on use case

## Common Questions

**Q: Can I create my own templates?**
A: Yes! Copy a JSON file from `example_workflows/`, customize it, and it'll load automatically.

**Q: Will my templates sync?**
A: Currently stored locally. Cloud sync coming in Wave 2.

**Q: How long do templates take to run?**
A: Check the "⏱️" time estimate on each card. Most are 1-5 minutes.

**Q: Can I modify templates after loading?**
A: Yes! Once in the workflow builder, you can edit every node.

**Q: Why no external libraries?**
A: Pure HTML/CSS/JS = instant load, no dependencies, works offline.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Escape | Close preview modal |
| Tab | Navigate between templates |
| Enter | Use selected template |
| / | Focus search (when implemented) |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Templates won't load | Ensure `example_workflows/` directory exists |
| Modal stuck | Press Escape or refresh page |
| Search not working | Check spelling, search is case-insensitive |
| Styling looks weird | Zoom to 100%, refresh browser cache |

## Next Steps

1. **Try the Gallery**: Open `template_gallery.html`
2. **Browse Templates**: Explore different categories
3. **Load a Template**: Click "Use" on any template
4. **Customize**: Edit it in the workflow builder
5. **Learn More**: See `TEMPLATE_GALLERY_README.md` for advanced features

## More Information

- **Full Docs**: [TEMPLATE_GALLERY_README.md](TEMPLATE_GALLERY_README.md)
- **Workflow Builder**: [workflow_builder.html](workflow_builder.html)
- **Example Workflows**: [example_workflows/](example_workflows/)

---

**Happy building! 🚀**

Questions? Check the full README or explore the HTML/JS source code.
