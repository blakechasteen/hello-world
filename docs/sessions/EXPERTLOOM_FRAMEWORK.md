# ExpertLoom Framework - Community Extension System

**Built: October 24, 2024**

## Vision

ExpertLoom is a domain-agnostic expertise capture platform that helps workers preserve, share, and monetize their knowledge. It combines:

- **Technical Innovation:** Entity resolution + measurement extraction + semantic search
- **Social Mission:** "Expertise as Equity" - workers own and license their knowledge
- **Community Growth:** Framework extended by experts contributing new domains

## What We Built

### 1. Core Infrastructure (mythRL_core/)

**Entity Resolution System** ([resolver.py](mythRL_core/entity_resolution/resolver.py))
- Domain-agnostic entity registry with alias mapping
- Multiple resolution strategies (exact, longest, greedy)
- Confidence scoring
- Fast lookup via pre-built indices

**Measurement Extraction** ([extractor.py](mythRL_core/entity_resolution/extractor.py))
- Numeric pattern extraction (8 frames, 50g, 32 PSI)
- Categorical pattern extraction (calm, dirty, squealing)
- Temporal extraction (dates, timestamps)
- Domain-specific custom patterns

**Complete Pipeline:**
Text → Entity Resolution → Measurement Extraction → Structured Payload → Vector Storage (Qdrant)

### 2. Domain Template System

**DOMAIN_TEMPLATE/** ([mythRL_core/domains/DOMAIN_TEMPLATE/](mythRL_core/domains/DOMAIN_TEMPLATE/))
- [registry.json](mythRL_core/domains/DOMAIN_TEMPLATE/registry.json) - Complete annotated template with all fields explained
- [marketplace.json](mythRL_core/domains/DOMAIN_TEMPLATE/marketplace.json) - Marketplace metadata for Expert tier licensing

**Key Template Sections:**
- **Entities:** Canonical IDs + aliases + attributes
- **Measurement Patterns:** Regex patterns for numeric + categorical extraction
- **Relationships:** How entities connect (REQUIRES, USES, AFFECTS)
- **Micropolicies:** Expert rules ("if X then Y")
- **Reverse Queries:** Diagnostic questions system asks users
- **Validation:** Test phrases to ensure extraction works

### 3. Example Domains

**Beekeeping Domain** ([mythRL_core/domains/beekeeping/](mythRL_core/domains/beekeeping/))
- 10 entities (hives, treatments, beekeepers, forage)
- 34 aliases
- Real-world tested with actual beekeeping notes

**Automotive Domain** ([mythRL_core/domains/automotive/](mythRL_core/domains/automotive/))
- 10 entities (vehicle, fluids, tires, components)
- 44 aliases
- 11 measurement patterns (tire pressure, mileage, voltage, etc.)
- 5 micropolicies (safety alerts for low pressure, worn brakes, etc.)
- 5 reverse queries (diagnostic question flows)
- Full [README documentation](mythRL_core/domains/automotive/README.md)
- [Marketplace listing](mythRL_core/domains/automotive/marketplace.json) example

**Demonstrates:**
- Same code, completely different domains
- Domain-agnostic architecture working in practice
- Community contribution pattern

### 4. Contribution System

**Contribution Guide** ([CONTRIBUTING_DOMAINS.md](CONTRIBUTING_DOMAINS.md))
- Step-by-step instructions for creating domains
- Best practices for entity definition
- Regex pattern writing tips
- Micropolicy encoding guidelines
- Reverse query design
- Community vs Expert tier explanation
- Revenue sharing model details

**Domain Validator** ([validate_domain.py](validate_domain.py))
Automated validation tool that checks:
- Schema completeness (required fields)
- Regex pattern validity
- Test phrase coverage
- Entity definition quality
- Micropolicy correctness
- Reverse query structure

Run: `python validate_domain.py <path_to_registry.json>`

Results:
- ✓ Automotive domain: **PASSED** (28 checks, 3 warnings)
- ✗ Beekeeping domain: **FAILED** (4 checks, 9 warnings, 2 errors) - needs updates

### 5. Demonstration Scripts

**demo_entity_resolver.py** - Entity resolution basics
**demo_auto_tagger.py** - Measurement extraction + entity resolution
**demo_full_pipeline.py** - End-to-end: text → embed → store → query (Qdrant)
**demo_automotive_domain.py** - Automotive domain validation

All demos working and tested on Windows.

## Technical Stack

- **Python 3.12+**
- **sentence-transformers** - Free local embeddings (all-MiniLM-L6-v2, 384-dim)
- **Qdrant** - Vector database with hybrid search
- **NetworkX** - Knowledge graph (beekeeping demo)
- **Regex** - Measurement pattern extraction
- **JSON** - Domain schema format

## Architecture Principles

### 1. Domain-Agnostic Core
The entity resolution and measurement extraction engines know nothing about specific domains. All domain knowledge lives in JSON configs.

### 2. Graceful Degradation
- Missing optional dependencies → warnings, not crashes
- Failed extractions → empty results, system continues
- Unicode issues → ASCII fallbacks

### 3. Community Extensibility
Anyone can create a domain:
1. Copy DOMAIN_TEMPLATE
2. Fill in their expertise
3. Run validator
4. Submit PR (community) or apply for Expert verification

### 4. Expertise as Equity
Experts can monetize their domains:
- Free tier (limited queries)
- Pro tier ($9.99/month) → 70% to expert
- Enterprise tier (custom pricing)

**Example earnings:** 200 Pro subscribers × $7 = $1,400/month passive income

## Social Impact

### Problems We Address

**1. Aging Workforce Brain Drain**
- 10,000 baby boomers retire daily
- Decades of expertise lost when they leave
- Solution: Capture knowledge before retirement

**2. Wealth Inequality from Automation**
- AI replaces workers but owners keep all value
- Workers who trained the systems get nothing
- Solution: Workers own and license their expertise

**3. Exploitative Knowledge Extraction**
- Companies mine worker knowledge without compensation
- "Can you train the new person?" = unpaid knowledge transfer
- Solution: Fair compensation for captured expertise

### Target Users

**Experts (Knowledge Creators):**
- Master mechanics
- Experienced beekeepers
- Professional chefs
- Expert gardeners
- Skilled tradespeople
- Retiring teachers
- Rural workers with specialized knowledge

**Learners (Knowledge Consumers):**
- DIY enthusiasts
- New professionals
- Small businesses
- Students
- Hobbyists
- Anyone learning a new skill

### Economic Model

**For Experts:**
- 70% of revenue from your domains
- Passive income after initial creation
- Recognition and verified credentials
- Control over pricing and updates
- Retain IP ownership

**For Platform:**
- 30% of revenue for infrastructure
- Hosts marketplace and search
- Handles payments and licensing
- Provides validation and support
- Maintains core framework

**For Users:**
- Free tier for basic learning
- Pro tier for serious users ($9.99/month)
- Enterprise for businesses (custom)
- Always transparent pricing

## Current Status

### ✅ Completed

1. **Core entity resolution engine** - Working across domains
2. **Measurement extraction with custom patterns** - Domain-specific regex
3. **Full pipeline demos** - Text → Qdrant working end-to-end
4. **Domain template system** - Complete with annotations
5. **Automotive example domain** - 95% completeness score, validates
6. **Contribution documentation** - Step-by-step guide for creators
7. **Domain validator** - Automated quality checks
8. **Marketplace metadata schema** - Expert tier licensing structure

### 🚧 In Progress / Next Steps

**Immediate Technical:**
1. Temporal search with time-weighted scoring (exponential decay)
2. Reverse query engine implementation
3. Micropolicy evaluation engine
4. Prediction tracking system (ground truth eval)

**Community Framework:**
1. Domain discovery CLI (`expertloom search <query>`)
2. Domain installer (`expertloom install automotive`)
3. Expert verification process
4. Revenue dashboard
5. Usage analytics

**Platform Development:**
1. Web marketplace UI
2. Payment processing integration
3. Expert verification panel
4. Rating and review system
5. Documentation site

**Additional Domains (Community Contributions):**
- Cooking/recipes
- Woodworking
- Gardening
- Home repair
- Small business management
- Healthcare (nursing, eldercare)
- Education (teaching strategies)

## Key Files Reference

### Core Framework
- `mythRL_core/entity_resolution/resolver.py` - Entity registry and resolution
- `mythRL_core/entity_resolution/extractor.py` - Measurement extraction
- `mythRL_core/entity_resolution/__init__.py` - Public API

### Templates
- `mythRL_core/domains/DOMAIN_TEMPLATE/registry.json` - Domain schema template
- `mythRL_core/domains/DOMAIN_TEMPLATE/marketplace.json` - Marketplace metadata

### Example Domains
- `mythRL_core/domains/beekeeping/registry.json` - Original test domain
- `mythRL_core/domains/automotive/registry.json` - Complete example domain
- `mythRL_core/domains/automotive/README.md` - Domain documentation
- `mythRL_core/domains/automotive/marketplace.json` - Marketplace listing

### Documentation
- `CONTRIBUTING_DOMAINS.md` - How to create domains
- `EXPERTLOOM_FRAMEWORK.md` - This file
- Demo scripts: `demo_*.py` - Working examples

### Tools
- `validate_domain.py` - Domain quality validator

## Design Decisions

### Why JSON for Domain Schemas?
- Human-readable and editable
- No coding required for domain creation
- Easy version control (git)
- Lightweight and portable
- Standard tooling support

### Why Regex for Measurements?
- Fast and deterministic
- No API costs (vs LLM extraction)
- Fully controllable by domain creators
- Works offline
- Easy to test and debug

### Why Qdrant?
- Best-in-class vector search
- Hybrid search (semantic + filters)
- Open source with cloud option
- Rich metadata payloads
- Python-first API

### Why Sentence-Transformers?
- 100% free, no API keys
- Runs locally
- Good quality embeddings
- Matryoshka support (multiple scales)
- Active community

### Why 70/30 Revenue Split?
- Expert does the hard work (domain creation)
- Platform provides infrastructure and users
- 70% competitive with creator platforms (YouTube: 55%, Patreon: 90% minus fees)
- Sustainable for both parties

## Success Metrics

### For Domains
- Validation passing rate
- Entity coverage (% of domain concepts captured)
- Test phrase coverage (% of entities tested)
- Measurement extraction accuracy
- User ratings and reviews

### For Experts
- Monthly revenue from licenses
- Active users of domain
- Domain update frequency
- Community engagement (forum posts, support)

### For Platform
- Total domains available
- Active domain creators
- Paying subscribers
- Query volume
- User retention

## Vision Forward

### Short Term (3 months)
- 10 high-quality community domains
- 3 verified expert domains
- Basic marketplace UI
- Revenue processing for experts

### Medium Term (6 months)
- 50+ domains across 10 categories
- 20 verified experts earning passive income
- Domain recommendation engine
- Mobile app for field use

### Long Term (1 year+)
- 500+ domains
- 100+ verified experts
- Enterprise partnerships
- International expansion (multilingual)
- Domain collaboration tools (multiple experts per domain)

## Getting Started

### For Users (Learning)
```bash
# Install ExpertLoom
pip install expertloom

# Search for domains
expertloom search automotive

# Install a domain
expertloom install automotive

# Use in your code
from expertloom import load_domain
domain = load_domain("automotive")
```

### For Creators (Contributing)
```bash
# Clone repository
git clone https://github.com/yourusername/mythRL

# Copy template
cp -r mythRL_core/domains/DOMAIN_TEMPLATE mythRL_core/domains/my_domain

# Edit registry.json with your expertise

# Validate
python validate_domain.py mythRL_core/domains/my_domain/registry.json

# Submit
# Community tier: Pull request
# Expert tier: Apply for verification
```

## Community

### Getting Help
- Documentation: `/docs`
- Examples: `demo_*.py` scripts
- Issues: GitHub issues
- Discussions: GitHub discussions (coming soon)
- Discord: (coming soon)

### Contributing
- Create domains: See CONTRIBUTING_DOMAINS.md
- Improve core: Standard GitHub contribution flow
- Documentation: PRs welcome
- Bug reports: GitHub issues

## License

- **Core Framework:** MIT
- **Example Domains:** MIT
- **Your Domains:** You choose (MIT, Apache, CC-BY, Proprietary)

## Credits

Built by Blake (mythRL project) with guidance from Claude (Anthropic).

**Special thanks to:**
- The beekeeping community for inspiration
- Open source LLM and embedding communities
- Everyone who will contribute domains

---

## Call to Action

**For Experts:**
Your knowledge is valuable. Don't let it disappear when you retire or move on. Capture it, share it, and earn from it.

**For Learners:**
Support the experts who are preserving knowledge. Pay for the domains you use. Leave reviews. Spread the word.

**For Developers:**
Help us build the infrastructure. Contribute code, documentation, or domains. This is open source - we're building it together.

**For Everyone:**
This is an experiment in economic justice through knowledge ownership. If it works, experts can earn fair compensation for their life's work. If it fails, we learn and try again.

Let's build something that actually helps people.

---

**Status:** ✅ Foundation complete, ready for community contributions
**Next milestone:** First 10 community domains + 3 verified expert domains
**Timeline:** 3 months to public beta

**Let's fucking go.**
