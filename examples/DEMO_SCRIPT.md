# Promptly Platform Video Demo Script

**Total Duration:** 12-15 minutes
**Target Audience:** Developers, ML Engineers, Product Managers
**Format:** Screencast with narration

---

## Setup Before Recording

### Environment
- [ ] Clean terminal with custom prompt
- [ ] IDE with syntax highlighting (VS Code recommended)
- [ ] Browser with clean profile (no extensions visible)
- [ ] Demo repository cloned
- [ ] All dependencies installed
- [ ] Screen resolution: 1920x1080
- [ ] Font size: 14-16pt for visibility

### Assets Needed
- [ ] Sample prompts prepared
- [ ] Test data files
- [ ] Example code snippets
- [ ] Demo outputs ready

---

## Act 1: Introduction (60 seconds)

### Scene 1.1: Title Screen
**Duration:** 10 seconds
**Visuals:** Promptly logo with tagline

**Narration:**
> "Welcome to Promptly - the complete platform for prompt engineering, management, and optimization. In this demo, we'll show you how Promptly transforms prompt development from an ad-hoc process into a systematic, version-controlled workflow."

---

### Scene 1.2: Problem Statement
**Duration:** 20 seconds
**Visuals:** Split screen showing chaotic prompt management vs. Promptly organized view

**Narration:**
> "Managing prompts across teams is challenging. Prompts scattered in notebooks, inconsistent versions, no testing framework, and no visibility into what's working. Promptly solves all of these problems."

**Text Overlay:**
- ❌ Scattered prompts across codebases
- ❌ No version control
- ❌ Inconsistent quality
- ❌ Manual testing
- ✅ **Promptly provides the solution**

---

### Scene 1.3: Platform Overview
**Duration:** 30 seconds
**Visuals:** Dashboard/architecture diagram

**Narration:**
> "Promptly gives you Git-like version control for prompts, comprehensive evaluation frameworks, powerful chain composition, and production-ready APIs. Whether you're working solo or with a team, Promptly scales with your needs."

**Key Features Highlight:**
- Version Control & Branching
- Evaluation & Testing
- Chain Composition
- Analytics & Monitoring
- REST API & SDK

---

## Act 2: Getting Started (90 seconds)

### Scene 2.1: Installation
**Duration:** 20 seconds
**Screen:** Terminal

**Commands:**
```bash
pip install promptly
promptly --version
```

**Narration:**
> "Getting started is simple. Install Promptly with pip, and you're ready to go. Let's create our first repository."

---

### Scene 2.2: Repository Initialization
**Duration:** 30 seconds
**Screen:** Terminal

**Commands:**
```bash
mkdir my-prompts
cd my-prompts
promptly init
ls -la .promptly/
```

**Narration:**
> "Initialize a Promptly repository just like Git. This creates a .promptly directory with a SQLite database for storing prompts, evaluations, and metadata. You can also use PostgreSQL, MongoDB, or Redis for production deployments."

**Text Overlay:**
- 📁 Local repository created
- 🗄️ SQLite database initialized
- 🌿 Main branch ready
- ⚙️ Configuration set

---

### Scene 2.3: Create First Prompt
**Duration:** 40 seconds
**Screen:** Terminal and text editor side-by-side

**Commands:**
```bash
promptly create summarizer "Summarize the following text in {max_words} words:\n\n{text}"
promptly list
promptly show summarizer
```

**Narration:**
> "Creating prompts is straightforward. Use the CLI or Python SDK. Prompts support variable placeholders using curly braces. Let's see our prompt in action."

---

## Act 3: Version Control & Branching (120 seconds)

### Scene 3.1: Branching Workflow
**Duration:** 45 seconds
**Screen:** Terminal with visual branch diagram

**Commands:**
```bash
promptly branch development
promptly checkout development
promptly list-branches
```

**Narration:**
> "Just like Git, Promptly supports branching. Create a development branch to test changes before merging to production. This is perfect for A/B testing different prompt variations."

**Visual:** Show branch diagram with main → development

---

### Scene 3.2: Making Changes
**Duration:** 45 seconds
**Screen:** Terminal

**Commands:**
```bash
promptly create summarizer_v2 "Provide a concise summary of the following text (max {max_words} words):\n\nText: {text}\n\nSummary:"
promptly eval summarizer_v2 --test-cases test_cases.json
```

**Narration:**
> "We can create improved versions and evaluate them side-by-side. Promptly tracks all changes, scores, and metadata. Let's compare the original with our new version."

**Show:** Test results comparison table

---

### Scene 3.3: Merging Best Version
**Duration:** 30 seconds
**Screen:** Terminal

**Commands:**
```bash
promptly checkout main
promptly merge development
promptly history summarizer
```

**Narration:**
> "After testing confirms the new version performs better, merge it to main. Full version history is preserved, so you can always roll back if needed."

**Visual:** Version history graph

---

## Act 4: Evaluation Framework (120 seconds)

### Scene 4.1: Defining Test Cases
**Duration:** 40 seconds
**Screen:** Text editor showing test_cases.json

**File Content:**
```json
{
  "test_cases": [
    {
      "inputs": {"text": "Long article...", "max_words": 50},
      "expected": "Concise summary...",
      "evaluator": "semantic_similarity"
    }
  ]
}
```

**Narration:**
> "Create comprehensive test suites with inputs, expected outputs, and evaluation criteria. Promptly supports multiple evaluators: semantic similarity, keyword matching, LLM-based scoring, and custom functions."

---

### Scene 4.2: Running Evaluations
**Duration:** 40 seconds
**Screen:** Terminal with progress bar

**Commands:**
```bash
python -c "from examples.use_cases.customer_service import CustomerServiceDemo; demo = CustomerServiceDemo(); demo.run()"
```

**Narration:**
> "Run evaluations across all your test cases. Promptly tracks scores over time, helping you identify regressions and improvements. Let's look at a real-world example from our customer service use case."

**Show:** Evaluation results table with scores

---

### Scene 4.3: Quality Metrics
**Duration:** 40 seconds
**Screen:** Dashboard/visualization

**Visuals:** Charts showing:
- Score trends over time
- Quality distribution
- Pass/fail rates

**Narration:**
> "Visualize quality metrics over time. Track prompt performance across dimensions like accuracy, relevance, and engagement. Set quality thresholds to prevent regressions."

---

## Act 5: Chain Composition (120 seconds)

### Scene 5.1: Simple Chain
**Duration:** 40 seconds
**Screen:** Text editor showing chain YAML

**File Content:**
```yaml
name: content_pipeline
description: Multi-stage content generation
steps:
  - name: outline
    prompt: content_outline
    inputs: {topic: "{user_topic}"}
  - name: draft
    prompt: content_draft
    inputs: {outline: "{outline}"}
  - name: edit
    prompt: content_edit
    inputs: {draft: "{draft}"}
```

**Narration:**
> "Chains let you compose multiple prompts into workflows. This content pipeline generates an outline, writes a draft, then edits it - all automatically. Each step uses the output from the previous one."

---

### Scene 5.2: Executing Chains
**Duration:** 40 seconds
**Screen:** Terminal showing execution

**Commands:**
```bash
python -c "from examples.use_cases.content_generation import ContentGenerationDemo; demo = ContentGenerationDemo(); demo.run()"
```

**Narration:**
> "Execute chains with a single command. Promptly handles data flow between steps, error handling, and retry logic. Let's run our content generation pipeline."

**Show:** Live execution with progress

---

### Scene 5.3: Advanced Features
**Duration:** 40 seconds
**Screen:** Split view: YAML and visual diagram

**Features Shown:**
- Parallel execution
- Conditional branching
- Error handling
- Retry policies

**Narration:**
> "Advanced chains support parallel execution, conditional logic, loops, and sophisticated error handling. Build production-ready workflows for RAG, agents, content generation, and more."

**Visual:** Workflow diagram with parallel branches

---

## Act 6: Production Features (120 seconds)

### Scene 6.1: Analytics & Monitoring
**Duration:** 40 seconds
**Screen:** Analytics dashboard

**Visuals:**
- Usage metrics
- Performance graphs
- Quality trends
- Error rates

**Narration:**
> "Monitor prompt performance in production. Track usage patterns, latency, quality scores, and error rates. Promptly's analytics help you optimize prompts based on real-world data."

**Show:** Interactive dashboard

---

### Scene 6.2: REST API
**Duration:** 40 seconds
**Screen:** Terminal and browser with Swagger UI

**Commands:**
```bash
python -m Promptly.promptly.api.main
# Open browser to http://localhost:8000/docs
```

**Narration:**
> "Promptly provides a production-ready REST API with authentication, rate limiting, and comprehensive endpoints. Integrate Promptly into any application, regardless of language."

**Show:**
- Swagger documentation
- Example API calls
- Response formats

---

### Scene 6.3: SDK Usage
**Duration:** 40 seconds
**Screen:** Python code editor

**Code:**
```python
from Promptly.promptly.sdk.client import PromptlyClient

client = PromptlyClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Get prompt
prompt = client.get_prompt("summarizer")

# Execute chain
result = client.execute_chain(
    "content_pipeline",
    inputs={"user_topic": "AI trends"}
)

# Track evaluation
client.evaluate_prompt(
    "summarizer",
    test_cases=test_data
)
```

**Narration:**
> "The Python SDK provides an intuitive interface for all Promptly features. Use it in your applications, notebooks, or deployment pipelines."

---

## Act 7: Real-World Use Cases (90 seconds)

### Scene 7.1: Customer Service
**Duration:** 30 seconds
**Screen:** Running customer_service.py demo

**Narration:**
> "Let's see Promptly in action. This customer service demo shows automated response quality scoring across multiple templates, tracking satisfaction correlation, and A/B testing greetings. All powered by Promptly's evaluation framework."

**Show:** Demo output with quality scores

---

### Scene 7.2: Content Generation
**Duration:** 30 seconds
**Screen:** Running content_generation.py demo

**Narration:**
> "Content generation pipelines benefit from Promptly's chain composition. This example creates blog posts, social media content, and newsletters from a single brief - with quality checks at each stage."

**Show:** Generated content examples

---

### Scene 7.3: Code Review
**Duration:** 30 seconds
**Screen:** Running code_review.py demo

**Narration:**
> "Automated code reviews with Promptly ensure consistent quality. This demo analyzes security, performance, and best practices - all trackable and improvable over time."

**Show:** Review results with scores

---

## Act 8: Conclusion & Call to Action (60 seconds)

### Scene 8.1: Recap
**Duration:** 30 seconds
**Screen:** Feature overview with checkmarks

**Narration:**
> "We've seen how Promptly provides version control, comprehensive evaluation, powerful chains, production analytics, and REST APIs. Everything you need to manage prompts at scale."

**Checklist:**
- ✅ Version control & branching
- ✅ Evaluation framework
- ✅ Chain composition
- ✅ Analytics & monitoring
- ✅ Production API
- ✅ Multiple storage backends

---

### Scene 8.2: Getting Started
**Duration:** 20 seconds
**Screen:** Documentation landing page

**Resources:**
- 📚 Documentation: promptly.dev/docs
- 💻 GitHub: github.com/your-org/promptly
- 🎓 Interactive Tutorial: Run locally
- 📊 Benchmark Suite: Performance tests
- 🎯 Use Case Demos: 5 real-world examples

**Narration:**
> "Get started today with our interactive tutorial, explore the use case demos, and join our community. Links in the description."

---

### Scene 8.3: Call to Action
**Duration:** 10 seconds
**Screen:** Call to action overlay

**Text:**
- ⭐ Star us on GitHub
- 💬 Join our Discord
- 📖 Read the docs
- 🚀 Get started now

**Narration:**
> "Thanks for watching! Star us on GitHub, join our community, and let us know what you build with Promptly."

---

## Post-Production Checklist

### Editing
- [ ] Add transitions between scenes
- [ ] Include background music (subtle, non-distracting)
- [ ] Add text overlays for key points
- [ ] Include code syntax highlighting
- [ ] Ensure consistent audio levels
- [ ] Add chapter markers

### Graphics
- [ ] Animated logo intro
- [ ] Terminal prompts clearly visible
- [ ] Highlight important code sections
- [ ] Show/hide mouse cursor appropriately
- [ ] Add visual diagrams for complex concepts

### Final Review
- [ ] All narration clear and well-paced
- [ ] Visuals match narration timing
- [ ] No dead air or awkward pauses
- [ ] Call to action prominent
- [ ] Links in description
- [ ] Captions/subtitles available

---

## Video Metadata

### Title Options
1. "Promptly: Complete Prompt Management Platform Demo"
2. "Version Control for AI Prompts - Promptly Platform"
3. "Promptly Demo: Manage, Test & Deploy Prompts at Scale"

### Description Template
```
Promptly is a complete platform for prompt engineering and management.
This demo covers:
- Repository setup and version control
- Comprehensive evaluation framework
- Chain composition for complex workflows
- Production analytics and monitoring
- REST API and SDK integration

🔗 Links:
- Documentation: [URL]
- GitHub: [URL]
- Interactive Tutorial: [URL]
- Use Case Demos: [URL]

⏱️ Timestamps:
0:00 - Introduction
1:00 - Getting Started
2:30 - Version Control
4:30 - Evaluation Framework
6:30 - Chain Composition
8:30 - Production Features
10:30 - Real-World Use Cases
12:00 - Conclusion

#AI #PromptEngineering #MLOps #Developer Tools
```

### Tags
- prompt engineering
- AI development
- MLOps
- version control
- developer tools
- machine learning
- LLM
- prompt management
- testing framework
- API

---

## Recording Tips

### Audio
- Use a quality microphone
- Record in a quiet environment
- Speak clearly and at a moderate pace
- Leave pauses for editing
- Record voiceover separately if needed

### Screen Recording
- Use high-quality screen recorder (OBS, ScreenFlow, Camtasia)
- 1080p minimum resolution
- 30 FPS frame rate
- Hide desktop clutter
- Use zoom for important details

### Performance
- Close unnecessary applications
- Disable notifications
- Pre-load all demos
- Have backup recordings
- Test run before final recording

---

## Alternative Formats

### Short Version (3-5 minutes)
Focus on:
1. Quick intro (30s)
2. Basic usage (60s)
3. Key feature (90s)
4. Use case (60s)
5. Call to action (30s)

### Tutorial Series
Break into episodes:
1. Getting Started
2. Version Control & Branching
3. Evaluation Framework
4. Chains & Workflows
5. Production Deployment

### Live Demo Format
- Interactive Q&A
- Real-time problem solving
- Community engagement
- Extended deep dives

---

## Success Metrics

Track:
- Views and watch time
- Engagement rate (likes, comments)
- Click-through to docs/GitHub
- New users from video
- Community growth

Target:
- 70%+ average view duration
- 5%+ engagement rate
- 1000+ views in first month
- 100+ new GitHub stars
