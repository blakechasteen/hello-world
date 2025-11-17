#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Query Interface
=====================

Demonstrates natural language queries over org-mode knowledge graphs.

This demo:
1. Creates sample org files
2. Loads them into knowledge graph
3. Runs various natural language queries
4. Shows different output formats
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.memory.graph import KG
from HoloLoom.spinningWheel.orgmode import OrgModeSpinner
from HoloLoom.spinningWheel.base import SpinnerConfig
from HoloLoom.query.engine import QueryEngine
from HoloLoom.query.formatter import format_result


async def create_sample_knowledge_graph():
    """Create a sample knowledge graph from org content."""
    print("Creating sample knowledge graph from org-mode content...")

    # Sample org content (realistic work scenario)
    org_content = """
* Work Projects                                       :work:

** Project: Customer Dashboard                       :frontend:priority:
   :PROPERTIES:
   :EFFORT: 40:00
   :PROGRESS: 75%
   :END:

*** DONE Design mockups
    CLOSED: [2025-11-10 Sun 14:00]

*** IN-PROGRESS Implement components
    Started on Monday. Making good progress.

*** TODO Add authentication
    DEADLINE: <2025-11-20 Wed>
    Needs to integrate with OAuth2 system.

*** TODO Testing and QA
    DEADLINE: <2025-11-22 Fri>

** Project: API Refactor                             :backend:

*** TODO Refactor auth endpoints
    DEADLINE: <2025-11-25 Mon>
    High priority - blocking other work.

*** TODO Update documentation
    Related to: [[Customer Dashboard][Dashboard project]]

* Personal Projects                                  :personal:

** Learning: Rust Programming                        :learning:

*** DONE Read Rust Book chapters 1-5
    CLOSED: [2025-11-15 Wed 18:00]

*** IN-PROGRESS Build CLI tool project
    Practicing ownership concepts.

*** TODO Contribute to open source
    DEADLINE: <2025-12-01 Sun>

* Journal                                            :journal:

** <2025-11-17 Sun>
   Productive day! Finished the component implementation.
   Found a tricky bug in the auth flow - took 2 hours to debug.

   Tomorrow: focus on testing.
"""

    # Parse org content
    kg = KG()
    spinner = OrgModeSpinner(SpinnerConfig(enable_enrichment=False))

    shards = await spinner.spin({
        'content': org_content,
        'source': 'demo_tasks.org',
        'episode': 'demo:work'
    })

    print(f"  Parsed {len(shards)} headings from org content")

    # Build knowledge graph
    from HoloLoom.memory.graph import KGEdge

    for shard in shards:
        # Add hierarchical edges
        if shard.metadata.get('parent_id'):
            kg.add_edge(KGEdge(
                src=shard.id,
                dst=shard.metadata['parent_id'],
                type='CHILD_OF',
                weight=1.0,
                metadata={'level': shard.metadata.get('level', 1)}
            ))

        # Add deadline edges
        if shard.metadata.get('deadline'):
            deadline_str = shard.metadata['deadline']
            kg.add_edge(KGEdge(
                src=shard.id,
                dst=f'time::{deadline_str}',
                type='DEADLINE',
                weight=1.0
            ))

        # Add TODO state edges (simulate history)
        todo_motifs = [m for m in shard.motifs if m.startswith('TODO:')]
        if todo_motifs:
            state = todo_motifs[0].split(':')[1]

            if state == 'IN-PROGRESS':
                # Add temporal edge showing when it was started
                kg.add_edge(KGEdge(
                    src=shard.id,
                    dst=f'change::{datetime.now().isoformat()}',
                    type='CHANGE_TODO_STATE_CHANGE',
                    weight=1.0,
                    metadata={
                        'old_value': 'TODO',
                        'new_value': 'IN-PROGRESS',
                        'title': shard.entities[0] if shard.entities else shard.id
                    }
                ))

        # Add link edges
        if 'links' in shard.metadata:
            for link in shard.metadata['links']:
                kg.add_edge(KGEdge(
                    src=shard.id,
                    dst=link['target'],
                    type='LINKS_TO',
                    weight=1.0,
                    metadata={'description': link['description']}
                ))

    stats = kg.stats()
    print(f"  Built graph: {stats['num_nodes']} nodes, {stats['num_edges']} edges")

    return kg, shards


async def demo_queries(kg):
    """Run demonstration queries."""
    print("\n" + "="*80)
    print("DEMONSTRATION: Natural Language Queries")
    print("="*80)

    engine = QueryEngine(kg)

    # List of demo queries
    demo_queries = [
        ("What's due this week?", "text"),
        ("Show me statistics", "text"),
        ("Find notes about dashboard", "text"),
        ("What's due?", "org"),  # Org format output
    ]

    for i, (query_str, output_format) in enumerate(demo_queries, 1):
        print(f"\n{'─'*80}")
        print(f"Query #{i}: '{query_str}'")
        if output_format != 'text':
            print(f"Format: {output_format}")
        print('─'*80 + "\n")

        result = await engine.query(query_str)
        formatted = format_result(result, format=output_format)

        print(formatted)

        # Show intent info
        print(f"\n[Intent: {result['intent']['type']}, "
              f"Confidence: {result['intent']['confidence']:.2f}]")


async def demo_cli_usage():
    """Show CLI usage examples."""
    print("\n" + "="*80)
    print("CLI USAGE EXAMPLES")
    print("="*80)

    print("""
Once you have org files loaded into a knowledge graph, you can query from CLI:

1. Basic queries:
   $ python -m HoloLoom.query "What should I work on?"
   $ python -m HoloLoom.query "What's due this week?"
   $ python -m HoloLoom.query "Show me statistics"

2. With specific knowledge graph:
   $ python -m HoloLoom.query "What's due?" --kg ~/knowledge.jsonl

3. Different output formats:
   $ python -m HoloLoom.query "What's due?" --format json
   $ python -m HoloLoom.query "What's due?" --format org >> ~/org/query-results.org

4. With live monitoring:
   $ python -m HoloLoom.query "What am I working on?" --monitor ~/org

5. Verbose mode:
   $ python -m HoloLoom.query "Statistics" --verbose

Query types supported:
  • "What should I work on?"        → Next task suggestions
  • "What am I working on?"          → Current tasks (IN-PROGRESS)
  • "What's due [timeframe]?"        → Deadlines
  • "When did I finish X?"           → Temporal queries
  • "Show my notes about X"          → Search
  • "What's the status of X?"        → Status checks
  • "What did I accomplish today?"   → Completed tasks
  • "What's blocked?"                → Blocked tasks
  • "What's high priority?"          → Priority sorting
  • "Show me statistics"             → Graph stats
""")


async def demo_python_api():
    """Show Python API usage."""
    print("\n" + "="*80)
    print("PYTHON API USAGE")
    print("="*80)

    print("""
Use the query interface from Python code:

```python
from HoloLoom.memory.graph import KG
from HoloLoom.query import query

# Load knowledge graph
kg = KG.load('knowledge.jsonl')

# Simple query
result = await query("What should I work on?", kg=kg)

# With formatting
from HoloLoom.query.formatter import format_result
formatted = format_result(result, format='text')
print(formatted)

# Access structured results
for task in result['results']:
    print(f"Task: {task['title']}")
    print(f"Deadline: {task['deadline']}")
    print(f"Days until: {task['days_until']}")

# Or use QueryEngine directly
from HoloLoom.query.engine import QueryEngine

engine = QueryEngine(kg)
result = await engine.query("What's due this week?")
```

With Emacs integration (coming soon):

```elisp
(defun hololoom-query (query-string)
  "Query HoloLoom from Emacs."
  (interactive "sQuery: ")
  (let ((result (shell-command-to-string
                 (format "python -m HoloLoom.query '%s'" query-string))))
    (with-output-to-temp-buffer "*HoloLoom Query*"
      (princ result))))

;; Use it:
;; M-x hololoom-query RET What should I work on? RET
```
""")


async def demo_integration_with_orgmode():
    """Show integration with org-mode files."""
    print("\n" + "="*80)
    print("INTEGRATION WITH ORG-MODE")
    print("="*80)

    print("""
The query interface integrates seamlessly with your org files:

1. **Load org files into graph:**

   ```python
   from HoloLoom.spinningWheel.orgmode import OrgModeSpinner
   from HoloLoom.memory.graph import KG

   spinner = OrgModeSpinner(SpinnerConfig())
   kg = KG()

   # Parse org file
   with open('~/org/tasks.org') as f:
       content = f.read()

   shards = await spinner.spin({
       'content': content,
       'source': 'tasks.org',
       'episode': 'work'
   })

   # Add to graph
   for shard in shards:
       # Add edges based on org structure
       ...
   ```

2. **Live monitoring integration:**

   ```python
   from HoloLoom.spinningWheel.orgmode_live import OrgLiveMonitor

   monitor = OrgLiveMonitor(kg, watch_dir='~/org')
   await monitor.start()

   # Now queries automatically reflect latest org file state
   result = await query("What's due?", kg=kg, monitor=monitor)
   ```

3. **Save graph for fast queries:**

   ```bash
   # Load org files once
   python -c "
   from HoloLoom.spinningWheel.orgmode import OrgModeSpinner
   from HoloLoom.memory.graph import KG
   # ... load and build graph ...
   kg.save('~/knowledge.jsonl')
   "

   # Then query quickly
   python -m HoloLoom.query "What's due?" --kg ~/knowledge.jsonl
   ```

4. **Query results to org:**

   ```bash
   # Export query results as org file
   python -m HoloLoom.query "What's due this week?" --format org > ~/org/this-week.org

   # Now you can include it in your main org file:
   # #+INCLUDE: "this-week.org"
   ```
""")


async def main():
    """Run complete demonstration."""
    print("="*80)
    print("HoloLoom Query Interface - Complete Demo")
    print("="*80)

    # Create sample data
    kg, shards = await create_sample_knowledge_graph()

    # Run demo queries
    await demo_queries(kg)

    # Show usage examples
    await demo_cli_usage()
    await demo_python_api()
    await demo_integration_with_orgmode()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("""
The Query Interface provides:

✓ Natural language queries over org-mode knowledge graphs
✓ Multiple output formats (text, JSON, org)
✓ CLI and Python API
✓ Integration with live file monitoring
✓ Semantic search and temporal queries
✓ Extensible handler system

Next steps:

1. Try it with your org files:
   python -m HoloLoom.query "What should I work on?" --monitor ~/org

2. Improve query classification:
   - Add ML-based intent detection
   - Support more complex queries
   - Add semantic embeddings for search

3. Emacs integration:
   - Create hololoom.el package
   - Add keybindings
   - Real-time query interface

4. Advanced features:
   - Query suggestions (autocomplete)
   - Query history
   - Saved queries
   - Query composition

The foundation is built - now you can query your org knowledge with natural language! 🚀
""")


if __name__ == '__main__':
    asyncio.run(main())
