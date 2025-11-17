# Promptly Demonstration Suite

**Complete collection of production-ready demonstrations for the Promptly platform.**

This repository contains comprehensive demos showcasing every aspect of Promptly, from basic usage to advanced production deployments. Whether you're new to Promptly or planning a production rollout, these demos will guide you through the entire platform.

---

## 📚 Table of Contents

- [Quick Start](#quick-start)
- [Demo Overview](#demo-overview)
- [Installation](#installation)
- [Demos](#demos)
  - [1. Interactive Tutorial](#1-interactive-tutorial)
  - [2. End-to-End Production Demo](#2-end-to-end-production-demo)
  - [3. Real-World Use Cases](#3-real-world-use-cases)
  - [4. Benchmark Suite](#4-benchmark-suite)
  - [5. Video Demo Script](#5-video-demo-script)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/hello-world.git
cd hello-world

# Install dependencies
pip install -r requirements.txt

# Run interactive tutorial (recommended for beginners)
python examples/interactive_tutorial.py

# Run full production demo
python examples/e2e_production_demo.py

# Run a specific use case
python examples/use_cases/customer_service.py
```

---

## 📋 Demo Overview

| Demo | Duration | Level | Focus Area |
|------|----------|-------|------------|
| **Interactive Tutorial** | 15-20 min | Beginner | Step-by-step learning |
| **E2E Production Demo** | 5-10 min | Intermediate | Full platform showcase |
| **Customer Service** | 3-5 min | Intermediate | Quality scoring & optimization |
| **Content Generation** | 5-7 min | Advanced | Multi-stage pipelines |
| **Code Review** | 3-5 min | Intermediate | Automated analysis |
| **Data Processing** | 5-7 min | Advanced | ETL workflows |
| **A/B Testing** | 4-6 min | Advanced | Systematic optimization |
| **Benchmark Suite** | 10-15 min | Advanced | Performance testing |

**Total Demonstration Time:** ~1-2 hours for complete walkthrough

---

## 💾 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 500MB free disk space (for demo outputs)

### Basic Installation

```bash
# Install Promptly
pip install promptly

# Verify installation
python -c "from Promptly.promptly.promptly import Promptly; print('✓ Promptly installed')"
```

### Optional Dependencies

For full feature support:

```bash
# Analytics and visualization
pip install plotext matplotlib pandas

# Template engine
pip install jinja2

# Database backends
pip install psycopg2-binary  # PostgreSQL
pip install pymongo          # MongoDB
pip install redis            # Redis

# API dependencies
pip install fastapi uvicorn pydantic

# Testing
pip install pytest pytest-asyncio
```

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/hello-world.git
cd hello-world

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Verify setup
python -m pytest Promptly/promptly/test_promptly.py
```

---

## 🎯 Demos

### 1. Interactive Tutorial

**File:** `examples/interactive_tutorial.py`

**Description:** Guided, step-by-step introduction to Promptly with interactive prompts and visual progress indicators.

**What You'll Learn:**
- Repository initialization
- Creating and managing prompts
- Branching and version control
- Running evaluations
- Building chains
- Analytics and monitoring
- Templates and composition
- API integration

**Usage:**
```bash
python examples/interactive_tutorial.py
```

**Features:**
- ✨ Interactive prompts for hands-on learning
- 📊 Visual progress indicators
- 💡 Tips and best practices
- 📜 Completion certificate generation
- 🎨 Color-coded terminal output

**Expected Output:**
```
tutorial_workspace/
├── my-prompts/
│   └── .promptly/
├── completion_certificate.json
└── tutorial_state.json
```

**Duration:** 15-20 minutes

---

### 2. End-to-End Production Demo

**File:** `examples/e2e_production_demo.py`

**Description:** Comprehensive demonstration of the entire Promptly platform in a production context.

**What's Demonstrated:**
1. **Initialization** with production storage (PostgreSQL/MongoDB/Redis)
2. **Prompt Creation** using templates and metadata
3. **Evaluation** with multiple evaluators
4. **Chain Building** (RAG pipeline example)
5. **Workflow Execution** with error handling
6. **Analytics Tracking** (performance, usage, quality)
7. **REST API** integration via SDK
8. **Visualization** and reporting

**Usage:**
```bash
# Basic (SQLite)
python examples/e2e_production_demo.py

# With PostgreSQL
python examples/e2e_production_demo.py --storage postgresql

# With MongoDB
python examples/e2e_production_demo.py --storage mongodb
```

**Arguments:**
- `--storage`: Storage backend (sqlite, postgresql, mongodb, redis)
- `--skip-api`: Skip API demonstration if server not running

**Expected Output:**
```
demo_output/
├── promptly_repo/
├── performance_data.csv
├── demo_report.html
└── demo_report.md
```

**Key Metrics:**
- Prompts created: 6-10
- Evaluations run: 3-5
- Chains executed: 1-2
- Total duration: 5-10 minutes

---

### 3. Real-World Use Cases

Five production-ready examples demonstrating Promptly in real business scenarios.

#### 3.1 Customer Service Automation

**File:** `examples/use_cases/customer_service.py`

**Scenario:** Automated response quality scoring for customer support

**Features:**
- Multiple response templates (formal, friendly, technical, billing, escalation)
- Multi-dimensional quality evaluation (empathy, clarity, accuracy, professionalism)
- A/B testing of greeting templates
- Response time tracking
- Satisfaction correlation analysis

**Usage:**
```bash
python examples/use_cases/customer_service.py
```

**Output:**
```
cs_output/
├── cs_repo/
└── customer_service_report.json
```

**Key Results:**
- 5 response templates tested
- Quality scores across 5 dimensions
- A/B test winner identification
- Correlation analysis between quality and satisfaction

---

#### 3.2 Content Generation Pipeline

**File:** `examples/use_cases/content_generation.py`

**Scenario:** Multi-stage content pipeline with quality validation

**Pipeline Stages:**
1. **Outline Generation** - Create structured outline
2. **Draft Writing** - Generate full draft
3. **Editing** - Refine and improve
4. **SEO Optimization** - Add meta tags and keywords
5. **Social Media Adaptation** - Create multi-platform content
6. **Newsletter Version** - Convert to email format
7. **Quality Check** - Comprehensive validation

**Usage:**
```bash
python examples/use_cases/content_generation.py
```

**Output:**
```
content_output/
├── content_repo/
├── blog_post.md
├── social_media_package.txt
├── newsletter.html
├── seo_metadata.json
└── content_generation_report.json
```

**Pipeline Performance:**
- 7 stages completed
- 8+ content pieces generated
- 88% time saved vs manual process
- Quality score: 8.5+/10

---

#### 3.3 Automated Code Review

**File:** `examples/use_cases/code_review.py`

**Scenario:** Automated code quality analysis across multiple languages

**Review Dimensions:**
- **Quality:** Readability, maintainability, code smells
- **Security:** Vulnerabilities, injection risks, data protection
- **Performance:** Time/space complexity, bottlenecks
- **Best Practices:** Framework patterns, testing, documentation
- **Complexity:** Cyclomatic complexity, cognitive load

**Usage:**
```bash
python examples/use_cases/code_review.py
```

**Output:**
```
code_review_output/
├── review_repo/
└── code_review_report.json
```

**Supported Languages:**
- Python
- JavaScript/TypeScript
- Java
- Go
- Rust

**Review Metrics:**
- Critical issues detected
- High-priority findings
- Average quality score
- Consistency analysis

---

#### 3.4 Data Processing ETL

**File:** `examples/use_cases/data_processing.py`

**Scenario:** ETL pipeline with validation and quality checks

**Pipeline Phases:**
1. **Extraction:** CSV, JSON, API sources
2. **Transformation:** Normalization, standardization
3. **Validation:** Completeness, accuracy, consistency
4. **Deduplication:** Exact and fuzzy matching
5. **Enrichment:** Geographic, demographic, derived fields
6. **Loading:** Output to various formats

**Usage:**
```bash
python examples/use_cases/data_processing.py
```

**Output:**
```
data_processing_output/
├── etl_repo/
└── etl_pipeline_report.json
```

**Pipeline Stats:**
- Records processed
- Validation pass/fail rates
- Duplicates found and resolved
- Data quality score

---

#### 3.5 A/B Testing Framework

**File:** `examples/use_cases/ab_testing.py`

**Scenario:** Systematic prompt optimization through controlled experimentation

**Testing Capabilities:**
- **Multi-variant Testing:** Test 3+ variations simultaneously
- **Statistical Significance:** Calculate p-values and confidence intervals
- **Multivariate Testing:** Test multiple factors at once
- **Sequential Testing:** Real-time monitoring with early stopping
- **Optimization Recommendations:** Data-driven insights

**Usage:**
```bash
python examples/use_cases/ab_testing.py
```

**Output:**
```
ab_testing_output/
├── ab_test_repo/
└── optimization_recommendations.json
```

**Test Results:**
- Variants tested: 4+
- Sample size per variant: 1000
- Statistical significance: 95% confidence
- Winner identification with lift calculation

---

### 4. Benchmark Suite

**File:** `examples/benchmark_suite.py`

**Description:** Comprehensive performance testing across all Promptly components.

**Benchmarked Operations:**
- Prompt CRUD (Create, Read, Update, List)
- Evaluation execution
- Chain operations
- Branching and merging
- Storage backend comparison
- Analytics queries
- Template rendering
- API throughput (simulated)

**Usage:**
```bash
# Basic run (100 iterations)
python examples/benchmark_suite.py

# High-precision run (1000 iterations)
python examples/benchmark_suite.py --iterations 1000

# Custom output path
python examples/benchmark_suite.py --output my_benchmark.html
```

**Arguments:**
- `--iterations`: Number of iterations per benchmark (default: 100)
- `--output`: Output HTML report filename (default: benchmark_report.html)

**Output:**
```
benchmark_output/
├── bench_repo/
├── benchmark_report.html
└── benchmark_results.json
```

**Report Contents:**
- Overall performance summary
- Detailed timing statistics (min, mean, median, p95, p99, max)
- Storage backend comparison
- Performance tier classification
- Optimization recommendations

**Performance Tiers:**
- **Excellent (<10ms):** Template rendering, prompt reads
- **Good (10-50ms):** List operations, evaluations
- **Acceptable (50-100ms):** Prompt creates, chain operations
- **Review (>100ms):** Complex analytics queries

**Expected Benchmarks:**
- Prompt read: <5ms
- Prompt create: 20-50ms
- Evaluation: 30-70ms
- Chain execution: 60-120ms
- Template render: <2ms

---

### 5. Video Demo Script

**File:** `examples/DEMO_SCRIPT.md`

**Description:** Complete script for creating a professional video demonstration of Promptly.

**Contents:**
- **Act-by-act breakdown** (8 acts, 12-15 minutes total)
- **Narration scripts** with timing
- **Screen recording instructions**
- **Visual guidelines** and best practices
- **Post-production checklist**
- **Alternative formats** (short version, tutorial series)

**Acts:**
1. **Introduction** (60s) - Problem statement and platform overview
2. **Getting Started** (90s) - Installation and first repository
3. **Version Control** (120s) - Branching and merging
4. **Evaluation Framework** (120s) - Testing and quality metrics
5. **Chain Composition** (120s) - Building workflows
6. **Production Features** (120s) - Analytics, API, SDK
7. **Real-World Use Cases** (90s) - Customer service, content, code review
8. **Conclusion** (60s) - Recap and call to action

**Usage Guide:**

1. **Pre-recording Setup:**
   ```bash
   # Prepare demo environment
   ./setup_video_demo.sh
   ```

2. **Recording:**
   - Follow script timing
   - Record in 1080p at 30 FPS
   - Use high-quality microphone
   - Screen resolution: 1920x1080

3. **Post-production:**
   - Add transitions
   - Include background music
   - Add text overlays
   - Insert chapter markers

**Target Metrics:**
- Watch time: 70%+ average
- Engagement: 5%+ (likes, comments)
- Views: 1000+ in first month
- Conversions: 100+ GitHub stars

---

## 📤 Output Files

All demos generate organized output in dedicated directories:

```
examples/
├── demo_output/                    # E2E production demo
│   ├── promptly_repo/
│   ├── performance_data.csv
│   ├── demo_report.html
│   └── demo_report.md
│
├── tutorial_workspace/             # Interactive tutorial
│   ├── my-prompts/
│   └── completion_certificate.json
│
├── benchmark_output/               # Benchmark suite
│   ├── bench_repo/
│   ├── benchmark_report.html
│   └── benchmark_results.json
│
└── use_cases/
    ├── cs_output/                  # Customer service
    │   ├── cs_repo/
    │   └── customer_service_report.json
    │
    ├── content_output/             # Content generation
    │   ├── content_repo/
    │   ├── blog_post.md
    │   ├── social_media_package.txt
    │   └── newsletter.html
    │
    ├── code_review_output/         # Code review
    │   ├── review_repo/
    │   └── code_review_report.json
    │
    ├── data_processing_output/     # Data processing
    │   ├── etl_repo/
    │   └── etl_pipeline_report.json
    │
    └── ab_testing_output/          # A/B testing
        ├── ab_test_repo/
        └── optimization_recommendations.json
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:**
```
ImportError: No module named 'Promptly'
```

**Solution:**
```bash
# Add repository root to Python path
export PYTHONPATH=/path/to/hello-world:$PYTHONPATH

# Or run from repository root
cd /path/to/hello-world
python examples/interactive_tutorial.py
```

#### 2. Database Locked

**Problem:**
```
sqlite3.OperationalError: database is locked
```

**Solution:**
```bash
# Close all Promptly instances
pkill -f promptly

# Remove lock file
rm examples/demo_output/promptly_repo/.promptly/promptly.db-journal

# Restart demo
python examples/e2e_production_demo.py
```

#### 3. Missing Dependencies

**Problem:**
```
ModuleNotFoundError: No module named 'jinja2'
```

**Solution:**
```bash
# Install optional dependencies
pip install jinja2 plotext pyyaml

# Or install all
pip install -r requirements.txt
```

#### 4. Permission Errors

**Problem:**
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
```bash
# Make output directories writable
chmod -R 755 examples/

# Or run from user directory
mkdir ~/promptly-demos
cd ~/promptly-demos
python /path/to/hello-world/examples/interactive_tutorial.py
```

#### 5. Performance Issues

**Problem:** Benchmarks running very slowly

**Solution:**
```bash
# Reduce iterations
python examples/benchmark_suite.py --iterations 10

# Close background applications
# Ensure sufficient RAM (4GB+)
# Use SSD for better I/O
```

---

## 🎓 Learning Path

### For Beginners

1. ✅ **Start Here:** Interactive Tutorial (15-20 min)
   - Hands-on introduction
   - Core concepts
   - Basic workflows

2. ✅ **Next:** E2E Production Demo (5-10 min)
   - Complete platform overview
   - Production features
   - Integration patterns

3. ✅ **Then:** Choose a Use Case (3-7 min)
   - Start with Customer Service
   - Real-world application
   - Best practices

### For Intermediate Users

1. ✅ Run all Use Case demos
2. ✅ Benchmark Suite for performance understanding
3. ✅ Customize demos for your use case
4. ✅ Explore API and SDK integration

### For Advanced Users

1. ✅ Review all demo source code
2. ✅ Modify for production scenarios
3. ✅ Create custom evaluators
4. ✅ Build complex chains
5. ✅ Implement custom storage backends

---

## 📊 Demo Comparison Matrix

| Feature | Tutorial | E2E Demo | Use Cases | Benchmark |
|---------|----------|----------|-----------|-----------|
| **Interactive** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Production Ready** | ⚠️ Partial | ✅ Yes | ✅ Yes | ⚠️ Testing |
| **Real Data** | ❌ No | ⚠️ Simulated | ⚠️ Simulated | ✅ Yes |
| **Code Examples** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Reports Generated** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Customizable** | ⚠️ Limited | ✅ Yes | ✅ Yes | ✅ Yes |
| **Time Required** | 15-20 min | 5-10 min | 3-7 min | 10-15 min |
| **Complexity** | Low | Medium | Medium-High | High |

---

## 🤝 Contributing

We welcome contributions to the demo suite!

### Adding a New Demo

1. **Create demo file:**
   ```bash
   touch examples/use_cases/my_use_case.py
   ```

2. **Follow template structure:**
   - Setup method
   - Step-by-step execution
   - Progress indicators
   - Error handling
   - Report generation
   - Main entry point

3. **Add documentation:**
   - Update this README
   - Include docstring
   - Add example output
   - Document requirements

4. **Test thoroughly:**
   ```bash
   python examples/use_cases/my_use_case.py
   ```

5. **Submit PR:**
   - Clear description
   - Screenshots/output examples
   - Updated documentation

### Demo Code Standards

- ✅ Clear, commented code
- ✅ Error handling
- ✅ Progress indicators
- ✅ JSON report generation
- ✅ Cleanup on exit
- ✅ Command-line arguments
- ✅ Helpful messages

---

## 📞 Support

### Resources

- **Documentation:** [promptly.dev/docs](https://promptly.dev/docs)
- **GitHub Issues:** [github.com/your-org/promptly/issues](https://github.com/your-org/promptly/issues)
- **Discord Community:** [discord.gg/promptly](https://discord.gg/promptly)
- **Email Support:** support@promptly.dev

### Getting Help

1. **Check this README** for common solutions
2. **Search existing issues** on GitHub
3. **Ask in Discord** for community help
4. **Open an issue** with:
   - Demo being run
   - Full error message
   - Python version
   - Operating system
   - Steps to reproduce

---

## 📝 License

All demo code is provided under the MIT License. See LICENSE file for details.

---

## 🙏 Acknowledgments

- Thanks to all contributors
- Inspired by best practices from Git, Docker, and modern DevOps tools
- Built with ❤️ by the Promptly team

---

## 🚀 What's Next?

After completing the demos:

1. **Explore the API:** `python -m Promptly.promptly.api.main`
2. **Build your first project:** Start with a real use case
3. **Join the community:** Share your prompts and workflows
4. **Read the docs:** Deep dive into advanced features
5. **Contribute back:** Help improve Promptly

---

**Happy Prompting! 🎉**

For questions, feedback, or just to say hi, reach out on Discord or GitHub.

---

*Last Updated: November 2025*
*Demo Suite Version: 1.0.0*
