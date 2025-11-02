# Promptly Web Analytics Dashboard

Beautiful web interface for visualizing Promptly prompt analytics in real-time.

## Features

### Live Statistics Dashboard
- **Total Executions** - Count of all prompt runs
- **Unique Prompts** - Number of different prompts executed
- **Success Rate** - Percentage of successful executions
- **Average Time** - Mean execution time across all prompts
- **Average Quality** - Mean quality score (when available)

### Prompt List
- View all executed prompts with stats
- Click for detailed history (coming soon)
- See execution count, success rate, avg time, quality score
- Auto-refresh every 30 seconds

### Recommendations Panel
- AI-powered improvement suggestions
- Identifies high-error prompts
- Suggests optimizations for slow prompts
- Highlights degrading quality trends
- Recommends archiving low-usage prompts

### Top Performers
Switch between metrics:
- **Quality** - Highest quality scores
- **Speed** - Fastest execution times
- **Cost Efficiency** - Best cost per execution

Top 5 prompts ranked with key metrics

## Installation

```bash
# Install Flask
pip install flask

# Run the dashboard
python web_dashboard.py
```

## Usage

1. Start the server:
```bash
python web_dashboard.py
```

2. Open browser to:
```
http://localhost:5000
```

3. Dashboard shows real-time analytics from SQLite database

## API Endpoints

### GET /api/summary
Overall analytics summary
```json
{
  "total_executions": 42,
  "unique_prompts": 12,
  "success_rate": 95.2,
  "avg_execution_time": 14.3,
  "avg_quality_score": 0.87,
  "total_cost": 0.15
}
```

### GET /api/prompts
List all prompts with stats
```json
[
  {
    "name": "sql_optimizer",
    "executions": 12,
    "success_rate": 100.0,
    "avg_time": 18.5,
    "avg_quality": 0.88
  }
]
```

### GET /api/prompt/<name>
Detailed stats for specific prompt
```json
{
  "prompt_name": "sql_optimizer",
  "total_executions": 12,
  "success_rate": 100.0,
  "avg_execution_time": 18.5,
  "avg_quality_score": 0.88,
  "quality_trend": "improving",
  "total_cost": 0.00,
  "last_executed": "2025-10-26T15:30:00"
}
```

### GET /api/prompt/<name>/history
Execution history for a prompt (last 50)
```json
[
  {
    "timestamp": "2025-10-26T15:30:00",
    "execution_time": 17.2,
    "quality_score": 0.92,
    "success": true
  }
]
```

### GET /api/recommendations
AI-powered recommendations
```json
[
  {
    "prompt_name": "slow_query",
    "issue": "Slow execution",
    "recommendation": "Simplify instructions or reduce context",
    "priority": "medium"
  }
]
```

### GET /api/top/<metric>?limit=5
Top prompts by metric (quality, speed, cost_efficiency)
```json
[
  {
    "prompt_name": "fast_parser",
    "total_executions": 45,
    "success_rate": 100.0,
    "avg_execution_time": 2.3,
    "avg_quality_score": 0.91,
    "total_cost": 0.00
  }
]
```

### GET /api/timeline
Execution timeline grouped by day (last 30 days)
```json
[
  {
    "date": "2025-10-26",
    "count": 15,
    "avg_time": 14.2,
    "avg_quality": 0.87
  }
]
```

## Architecture

### Backend (Flask)
- `web_dashboard.py` - Flask application
- RESTful API endpoints
- Direct SQLite access via PromptAnalytics

### Frontend (HTML/CSS/JS)
- `templates/dashboard.html` - Single-page dashboard
- Vanilla JavaScript (no frameworks)
- Auto-refresh every 30 seconds
- Responsive grid layout

### Styling
- Modern gradient background (purple/blue)
- Card-based UI with shadows
- Hover animations
- Color-coded badges and metrics
- Mobile-responsive design

## Color Scheme

- Primary: #667eea (purple-blue)
- Secondary: #764ba2 (deep purple)
- Success: #d4edda (light green)
- Warning: #fff3cd (light yellow)
- Info: #d1ecf1 (light blue)

## Future Enhancements

### Coming Soon
- [ ] Detailed prompt view with charts
- [ ] Execution timeline graph
- [ ] Quality trend visualization
- [ ] Export reports to PDF
- [ ] Filter by date range
- [ ] Search prompts
- [ ] Compare multiple prompts
- [ ] Real-time WebSocket updates
- [ ] Custom dashboard widgets
- [ ] Dark mode toggle

### Chart Libraries (Planned)
- Chart.js for time series
- D3.js for complex visualizations
- Plotly for interactive graphs

## Security Notes

**Development Mode:**
- Dashboard runs in Flask debug mode
- No authentication required
- Accessible on local network (0.0.0.0)

**Production Deployment:**
- Add authentication (Flask-Login)
- Use production WSGI server (Gunicorn)
- Enable HTTPS
- Restrict to localhost or VPN
- Add rate limiting
- Implement CSRF protection

## Performance

- SQLite queries optimized with indexes
- Auto-refresh minimizes load (30s interval)
- Pagination for large datasets (future)
- Caching layer (future)

## Dependencies

```
flask>=2.0.0
```

Optional for production:
```
gunicorn>=20.1.0
flask-cors>=3.0.10
```

## Troubleshooting

### Port already in use
```bash
# Change port in web_dashboard.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Database not found
- Dashboard uses same analytics.db as Promptly
- Default location: `~/.promptly/analytics.db`
- Run some prompts first to populate database

### Empty dashboard
- Execute prompts using Promptly MCP tools
- Or run demo scripts:
  ```bash
  python demo_analytics_live.py
  ```

## Integration

Works seamlessly with:
- Promptly MCP server (tools auto-record analytics)
- Loop composition (metrics tracked automatically)
- HoloLoom bridge (persistent memory + analytics)
- Ollama execution (local, $0 cost tracking)
- Claude API execution (cost tracking)

## Screenshots

```
┌─────────────────────────────────────────┐
│ Promptly Analytics Dashboard            │
│ Real-time insights into prompt perf.    │
└─────────────────────────────────────────┘

┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│  42  │ │  12  │ │ 95.2 │ │ 14.3 │
│Exec. │ │Prompt│ │Succ. │ │ Time │
└──────┘ └──────┘ └──────┘ └──────┘

┌─────────────────┐ ┌─────────────┐
│ All Prompts     │ │ Top Quality │
│                 │ │ 1. fast_... │
│ • sql_opt...    │ │ 2. code_... │
│ • code_rev...   │ │ 3. sql_...  │
│ • ui_design...  │ │             │
└─────────────────┘ └─────────────┘
```

## License

Part of Promptly - MIT License
