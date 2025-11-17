# Quick Start Guide

Deploy your first HoloLoom workflow in 5 minutes.

**[← Back to Docs](../README_WORKFLOWS_FIRST.md)**

---

## 🎯 Goal

By the end of this guide, you will have:
1. ✅ Picked a workflow that saves you time
2. ✅ Deployed it (one click)
3. ✅ Seen your first results
4. ✅ Measured the impact

**Total Time**: <5 minutes

---

## Step 1: Choose Your First Workflow (1 minute)

What do you want to automate?

### High-Impact Options

**📧 Drowning in email?**
→ [Inbox Triage](../workflow-gallery.md#inbox-triage) - Save 2 hours/day
- [See Sarah's story](../success-stories/sarah-inbox.md)

**🐛 Spending too much time on bug triage?**
→ [Bug Triage](../workflow-gallery.md#bug-triage-automation) - Save 30 min/bug
- [See TechCorp's story](../success-stories/techcorp-bugs.md)

**📊 Weekly reports taking forever?**
→ [Report Generation](../workflow-gallery.md#report-generation) - Save 4 hours/week

**🎬 Meeting notes piling up?**
→ [Meeting Summarization](../workflow-gallery.md#meeting-summarization) - 30 min/meeting

**📝 Code reviews are tedious?**
→ [Code Review Automation](../workflow-gallery.md#code-review-automation) - Save 20 min/PR

**👉 Or** [browse all 100+ workflows →](../workflow-gallery.md)

---

## Step 2: Deploy Your Workflow (2 minutes)

### Option A: Cloud Version (Fastest)

```bash
# 1. Click "Deploy" on workflow card
# 2. Select "Cloud Deployment"
# 3. Click "Create Account" or "Login"
# 4. Done!
```

Your workflow is now running in the cloud.

**Result**: Workflow is live and processing in ~30 seconds

### Option B: Local Deployment

```bash
# Install HoloLoom
pip install hololoom

# Deploy workflow
hololoom deploy inbox-triage

# Done!
```

Your workflow is running locally on your machine.

### Option C: Enterprise (Self-Hosted)

```bash
# Deploy to your infrastructure
hololoom deploy inbox-triage --platform kubernetes
# OR
hololoom deploy inbox-triage --platform docker
```

---

## Step 3: Configure Your Workflow (1-2 minutes)

Each workflow needs credentials to access external services:

### For Inbox Triage:
1. **Connect Gmail**
   - Click "Connect Gmail Account"
   - Authorize HoloLoom to access email
   - Done (OAuth secure)

2. **Optional: Connect Slack**
   - Click "Add Slack Notifications"
   - Authorize HoloLoom
   - Choose notification channel
   - Done

### For Bug Triage:
1. **Connect Jira**
   - Enter Jira URL
   - Create API token in Jira
   - Paste token
   - Done

2. **Optional: Connect Slack**
   - Same as above

### For Report Generation:
1. **Connect Data Source**
   - Google Sheets, SQL database, etc.
   - Choose database credentials
   - Done

### For Other Workflows:
See [Integrations Guide →](integrations.md) for specific setup steps

---

## Step 4: Run Your First Workflow

### Option 1: Automatic (Recommended)

Most workflows run automatically:
- **Inbox Triage**: Processes emails automatically every hour (configurable)
- **Bug Triage**: Processes bugs as they arrive
- **Report Generation**: Runs on schedule (daily/weekly/monthly)

Your first results appear within 1-2 hours.

### Option 2: Manual Test Run

For instant feedback, run a test:

```bash
hololoom run inbox-triage --test
```

Result: See sample output immediately

### Option 3: Demo Mode

See a complete walkthrough:

```bash
hololoom demo inbox-triage
```

Result: 5-minute demo showing how the workflow works

---

## Step 5: Check Your Analytics Dashboard (1 minute)

Every workflow has an analytics dashboard showing:

```
┌─────────────────────────────────────┐
│ Your Impact                         │
├─────────────────────────────────────┤
│ This Week:                          │
│ ⏱️  8.5 hours saved                 │
│ 📧 156 emails processed             │
│ ✅ 95% accuracy                     │
│ 💰 $1.25 saved in email processing  │
│                                     │
│ ROI: 150x 🚀                        │
│                                     │
│ [View Detailed Dashboard]           │
└─────────────────────────────────────┘
```

**View your dashboard**:
```bash
hololoom dashboard
```

Or visit: `http://localhost:8000/dashboard`

---

## Step 6: (Optional) Customize Your Workflow

If you want to tweak settings:

```bash
# Open visual configuration
hololoom edit inbox-triage
```

You can customize:
- Classification rules (what counts as "urgent")
- Routing rules (where to send notifications)
- Response templates
- Frequency (hourly, daily, real-time)

---

## 🎉 You're Done!

Congratulations! Your workflow is:
- ✅ Running
- ✅ Saving you time
- ✅ Tracking impact
- ✅ Getting smarter with each run

---

## Next Steps

### Learn More About Your Workflow

- [Inbox Triage Details →](../workflow-gallery.md#inbox-triage)
- [Bug Triage Details →](../workflow-gallery.md#bug-triage-automation)
- [Complete Workflow Gallery →](../workflow-gallery.md)

### Try Another Workflow

Popular combinations:
- **Productivity Boost**: Inbox Triage + Meeting Summarization + Calendar Optimization
- **Development Speed**: Bug Triage + Code Review + Test Generator
- **Insights Pipeline**: Report Generation + Data Cleaning + Competitive Intelligence

### Customize Your Workflow

- [Visual Workflow Builder →](../docs/creating-workflows.md)
- [Advanced Customization →](../docs/customization.md)
- [Custom Agents →](../docs/internals/custom-agents.md)

### Share Your Success

Got value from a workflow? [Share your story →](../success-stories/)

---

## Troubleshooting

### Workflow isn't processing emails

**Check**:
1. Is Gmail connected? (`hololoom status inbox-triage`)
2. Are there any errors? (`hololoom logs inbox-triage`)
3. Try manually running once: (`hololoom run inbox-triage --now`)

[More troubleshooting →](troubleshooting.md)

### Not seeing expected accuracy

**Note**: Workflows improve over time. Give it 50+ runs to settle in.

### Want to modify the workflow

Use the visual builder:
```bash
hololoom edit inbox-triage
```

Or customize via code:
```python
from HoloLoom.workflows import load_workflow

workflow = load_workflow("inbox-triage")
workflow.config.classification.urgency_threshold = 0.8
workflow.save()
```

---

## Key Concepts

### Confidence Scores

Each workflow decision includes a confidence score (0-100%):
```
✅ Classified as "URGENT" (87% confidence)
```

High confidence = trust the workflow
Low confidence = might need human review

### Analytics

Every workflow tracks:
- **Latency**: How long it takes to process
- **Success Rate**: Percentage of successful runs
- **Accuracy**: Matches against manual review
- **Impact**: Hours saved, value created

View via dashboard:
```bash
hololoom dashboard
```

### Integrations

Workflows connect to external services:
- **Email**: Gmail, Outlook, etc.
- **Chat**: Slack, Microsoft Teams, Discord
- **Tracking**: Jira, GitHub, Linear
- **Databases**: SQL, Postgres, MongoDB
- **APIs**: Any HTTP endpoint

[See all integrations →](integrations.md)

---

## Common Questions

**Q: How much does it cost?**
A: Free to try. $10-50/month for production use. Per-execution costs vary ($0.01-0.10 per run).

**Q: Is my data secure?**
A: Yes. Workflows run on your infrastructure (or our secure cloud). You control the data.

**Q: Can I deploy on my own servers?**
A: Yes. We support Docker, Kubernetes, and more. [See deployment options →](deploying-workflows.md)

**Q: How accurate is it?**
A: 90-98% depending on workflow. You can manually correct it to improve accuracy over time.

**Q: What if I need help?**
A: [Support resources →](../docs/troubleshooting.md)

---

## Success Tips

### 1. Start Simple
Use pre-built templates first. Customize later.

### 2. Monitor Your Workflow
Check the analytics dashboard weekly. Look for:
- Trends (accuracy improving?)
- Anomalies (sudden drops in accuracy?)
- Patterns (when does it work best?)

### 3. Give Feedback
If the workflow makes a mistake, correct it. It learns from feedback:
```bash
hololoom feedback inbox-triage --corrected "This is not urgent"
```

### 4. Share With Your Team
If a workflow saves you time, your team probably also needs it.

### 5. Track the ROI
See how many hours you're saving:
```bash
hololoom impact inbox-triage
```

---

## What To Do Next

**Already deployed your first workflow?**

1. ✅ Check your analytics dashboard
2. ✅ Give it feedback on 5-10 results (helps it learn)
3. ✅ Deploy a second workflow (most users combine 2-3)
4. ✅ Share your success story

**Need help?**
- [FAQ →](faq.md)
- [Troubleshooting →](troubleshooting.md)
- [Contact Support →](https://discord.gg/hololoom)

---

## Quick Reference

| Task | Command |
|------|---------|
| **Browse workflows** | `hololoom gallery` |
| **Deploy workflow** | `hololoom deploy NAME` |
| **Run immediately** | `hololoom run NAME --now` |
| **View analytics** | `hololoom dashboard` |
| **Edit workflow** | `hololoom edit NAME` |
| **View logs** | `hololoom logs NAME` |
| **Share feedback** | `hololoom feedback NAME --message "..."` |

---

**Congratulations! You're now part of the HoloLoom community.** 🎉

[Back to Home →](../README_WORKFLOWS_FIRST.md) | [Browse More Workflows →](../workflow-gallery.md) | [See Success Stories →](../success-stories/)
