# Promptly - Quick Start Guide

## 🚀 Fastest Way to Get Started

### Option 1: Run Locally (Fastest - 30 seconds)

```bash
# Install dependencies
pip install flask flask-socketio eventlet

# Run the real-time dashboard
cd promptly
python web_dashboard_realtime.py

# Open browser
http://localhost:5000
```

### Option 2: Docker (Recommended for Production - 5 minutes)

```bash
# Build and run
docker-compose -f docker-compose.simple.yml up -d

# Check logs
docker-compose logs -f

# Access
http://localhost:5000
```

### Option 3: Full Docker Stack (with Nginx - 10 minutes)

```bash
# Build and run with nginx proxy
docker-compose up -d

# Access
http://localhost        # Through nginx
http://localhost:5000   # Direct access
```

## 📊 What You Get

### Real-time Dashboard
- Live WebSocket updates
- No refresh needed
- Push notifications
- Activity feed
- 10 interactive charts

### Features
- Prompt analytics
- Team collaboration
- User authentication
- Shared prompts/skills
- Activity tracking

## 🔧 Configuration

### Environment Variables
```bash
export PROMPTLY_DATA_DIR=/path/to/data  # Default: ~/.promptly
export FLASK_ENV=production             # or development
```

### Database Location
- Analytics: `~/.promptly/analytics.db`
- Team data: `~/.promptly/team.db`

## 📖 Next Steps

1. **Populate demo data:**
   ```bash
   python demo_dashboard_charts.py
   ```

2. **Create a team:**
   ```bash
   python team_collaboration.py
   ```

3. **Use with Claude Desktop:**
   - Install MCP server
   - Configure in Claude Desktop
   - Use 27 MCP tools

## 🐳 Docker Commands

```bash
# Build
docker build -t promptly:latest .

# Run
docker run -p 5000:5000 -v promptly-data:/data promptly:latest

# Stop
docker-compose down

# View logs
docker-compose logs -f promptly-web

# Restart
docker-compose restart

# Remove everything
docker-compose down -v
```

## 🌐 Access Points

- **Real-time Dashboard:** http://localhost:5000
- **Enhanced Dashboard:** http://localhost:5000/enhanced
- **Simple Dashboard:** http://localhost:5000/simple

## ⚡ Quick Test

```bash
# Terminal 1: Run server
python promptly/web_dashboard_realtime.py

# Terminal 2: Generate test data
python promptly/demo_dashboard_charts.py

# Browser: Watch live updates at http://localhost:5000
```

## 🔐 Team Features

```python
from promptly.team_collaboration import TeamCollaboration

# Create team
collab = TeamCollaboration()
user = collab.create_user("you", "you@example.com", "password")
team = collab.create_team("My Team", "Description", user.user_id)

# Share prompt
prompt = collab.share_prompt("SQL Optimizer", "...", user.user_id, team.team_id)
```

## 🛠️ Troubleshooting

### Port already in use
```bash
# Change port in web_dashboard_realtime.py
socketio.run(app, port=5001)
```

### Docker build fails
```bash
# Clean and rebuild
docker system prune -a
docker-compose build --no-cache
```

### Database permissions
```bash
# Fix permissions
chmod 755 ~/.promptly
chmod 644 ~/.promptly/*.db
```

## 📚 Documentation

- [FINAL_COMPLETE.md](FINAL_COMPLETE.md) - Complete feature list
- [COMPLETE_SUMMARY.md](COMPLETE_SUMMARY.md) - Platform overview
- [CHARTS_ADDED.md](CHARTS_ADDED.md) - Chart documentation

## 💡 Tips

1. **Best performance:** Use Docker with eventlet
2. **Development:** Use local Flask server
3. **Production:** Use docker-compose with nginx
4. **Teams:** Enable team collaboration features

## ✅ Success Criteria

You'll know it's working when you see:
- "Promptly Real-time Analytics Dashboard" in terminal
- Green "Connected" status in browser
- Live execution count updates
- Charts rendering smoothly

---

**That's it! You're ready to go!** 🎉

For detailed docs, see [FINAL_COMPLETE.md](FINAL_COMPLETE.md)
