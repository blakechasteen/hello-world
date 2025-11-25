#!/usr/bin/env python3
"""
Custom Skill Creation Demo
===========================
Step-by-step tutorial for creating a custom HoloLoom skill from scratch.

This demo creates a "commit-message-generator" skill that:
1. Analyzes git diff output
2. Generates conventional commit messages
3. Follows best practices for commit message format

Steps demonstrated:
1. Create YAML skill template
2. Test the skill
3. Show results
4. Explain customization options

Run: PYTHONPATH=. python demos/demo_custom_skill.py
"""

import asyncio
from pathlib import Path
import yaml

from HoloLoom.agentic import execute_skill, get_registry
from HoloLoom.config import Config


# Step 1: Define the skill template as Python dict
# (In production, this would be a YAML file)
COMMIT_MESSAGE_SKILL = {
    "name": "commit-message-generator",
    "version": "1.0.0",
    "description": "Generate conventional commit messages from git diffs",

    "metadata": {
        "category": "development",
        "author": "Tutorial Demo",
        "tags": ["git", "commit", "conventional-commits"],
        "created": "2025-11-22"
    },

    "system_prompt": """You are a git commit message expert specializing in Conventional Commits specification.

Your expertise includes:
- Conventional Commits format (type(scope): subject)
- Commit types: feat, fix, docs, style, refactor, test, chore
- Writing clear, concise commit messages
- Following best practices (imperative mood, no period at end, etc.)

Your role is to:
1. Analyze code changes from git diff
2. Determine appropriate commit type and scope
3. Generate clear, descriptive commit message
4. Follow Conventional Commits specification exactly

Always be precise and follow git best practices.""",

    "user_prompt_template": """Git diff to analyze:
```diff
{diff}
```

{% if previous_commits %}
Recent commit messages for context:
{previous_commits}
{% endif %}

Please generate a conventional commit message following this format:

<type>(<scope>): <subject>

<body>

Where:
- type: feat|fix|docs|style|refactor|test|chore
- scope: area of codebase affected (optional but recommended)
- subject: short description (imperative mood, no period)
- body: detailed explanation (optional, for complex changes)

Guidelines:
- Use imperative mood ("add feature" not "added feature")
- Keep subject line ≤50 characters
- Wrap body at 72 characters
- Explain WHAT and WHY, not HOW""",

    "parameters": [
        {
            "name": "diff",
            "type": "string",
            "required": True,
            "description": "Git diff output showing code changes"
        },
        {
            "name": "previous_commits",
            "type": "string",
            "required": False,
            "description": "Recent commit messages for context and style matching"
        }
    ],

    "reasoning": {
        "default_strategy": "critique",
        "max_iterations": 2,
        "quality_threshold": 0.85
    }
}


def create_skill_yaml_file():
    """Create the skill YAML file."""
    print("=" * 80)
    print("STEP 1: Creating Skill YAML File")
    print("=" * 80)

    skills_dir = Path("HoloLoom/agentic/skills")
    yaml_path = skills_dir / "commit-message-generator.yaml"

    # Write YAML file
    with open(yaml_path, 'w') as f:
        yaml.dump(COMMIT_MESSAGE_SKILL, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Created: {yaml_path}")
    print("\nYAML Content:")
    print("-" * 80)
    with open(yaml_path) as f:
        print(f.read())

    return yaml_path


async def test_skill_basic():
    """Test the skill with a simple change."""
    print("\n\n" + "=" * 80)
    print("STEP 2: Testing Skill - Basic Feature Addition")
    print("=" * 80)

    sample_diff = """diff --git a/app.py b/app.py
index 1234567..abcdefg 100644
--- a/app.py
+++ b/app.py
@@ -10,6 +10,12 @@ class User:
         self.name = name
         self.email = email

+    def validate_email(self):
+        '''Validate email format.'''
+        import re
+        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
+        return re.match(pattern, self.email) is not None
+
     def save(self):
         db.save(self)"""

    print("\nSample git diff:")
    print(sample_diff)

    result = await execute_skill(
        skill_name="commit-message-generator",
        parameters={"diff": sample_diff},
        config=Config.fast()
    )

    print("\n" + "=" * 80)
    print("GENERATED COMMIT MESSAGE:")
    print("=" * 80)
    print(result.output)

    print("\n" + "=" * 80)
    print("METADATA:")
    print("=" * 80)
    print(f"✓ Success: {result.success}")
    print(f"✓ Confidence: {result.confidence:.2f}")
    print(f"✓ Iterations: {result.iterations}")


async def test_skill_with_context():
    """Test the skill with previous commit context."""
    print("\n\n" + "=" * 80)
    print("STEP 3: Testing Skill - With Commit History Context")
    print("=" * 80)

    sample_diff = """diff --git a/tests/test_user.py b/tests/test_user.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/tests/test_user.py
@@ -0,0 +1,15 @@
+import pytest
+from app import User
+
+def test_validate_email_valid():
+    user = User("Alice", "alice@example.com")
+    assert user.validate_email() is True
+
+def test_validate_email_invalid():
+    user = User("Bob", "invalid-email")
+    assert user.validate_email() is False"""

    previous_commits = """feat(auth): add email validation method
fix(user): correct email regex pattern
test(auth): add login endpoint tests"""

    print("\nSample git diff (new test file):")
    print(sample_diff)

    print("\nPrevious commits for style reference:")
    print(previous_commits)

    result = await execute_skill(
        skill_name="commit-message-generator",
        parameters={
            "diff": sample_diff,
            "previous_commits": previous_commits
        },
        config=Config.fast()
    )

    print("\n" + "=" * 80)
    print("GENERATED COMMIT MESSAGE:")
    print("=" * 80)
    print(result.output)

    print("\n" + "=" * 80)
    print("Note: Message style matches previous commits!")
    print("=" * 80)


async def test_skill_complex_change():
    """Test the skill with a complex refactoring."""
    print("\n\n" + "=" * 80)
    print("STEP 4: Testing Skill - Complex Refactoring")
    print("=" * 80)

    complex_diff = """diff --git a/database.py b/database.py
index 1234567..abcdefg 100644
--- a/database.py
+++ b/database.py
@@ -1,20 +1,35 @@
-class Database:
-    def __init__(self):
-        self.connection = None
+from typing import Optional
+import asyncio
+from contextlib import asynccontextmanager

-    def connect(self, host, port):
-        self.connection = create_connection(host, port)
+class Database:
+    '''Async database connection manager.'''

-    def query(self, sql):
-        return self.connection.execute(sql)
+    def __init__(self, host: str, port: int):
+        self.host = host
+        self.port = port
+        self._connection: Optional[Connection] = None

-    def close(self):
-        if self.connection:
-            self.connection.close()
+    @asynccontextmanager
+    async def connection(self):
+        '''Async context manager for database connections.'''
+        conn = await create_async_connection(self.host, self.port)
+        try:
+            yield conn
+        finally:
+            await conn.close()
+
+    async def query(self, sql: str):
+        '''Execute async query with automatic connection management.'''
+        async with self.connection() as conn:
+            return await conn.execute(sql)"""

    print("\nComplex refactoring diff:")
    print(complex_diff[:500] + "...")  # Show preview

    result = await execute_skill(
        skill_name="commit-message-generator",
        parameters={"diff": complex_diff},
        config=Config.fused()  # Use FUSED for complex changes
    )

    print("\n" + "=" * 80)
    print("GENERATED COMMIT MESSAGE:")
    print("=" * 80)
    print(result.output)


async def explain_customization():
    """Explain customization options."""
    print("\n\n" + "=" * 80)
    print("STEP 5: Customization Options")
    print("=" * 80)

    print("""
The skill template can be customized in several ways:

1. SYSTEM PROMPT
   - Define expertise and perspective
   - Set tone and style
   - Specify domain knowledge

   Example modifications:
   • Add your company's specific commit conventions
   • Include team-specific terminology
   • Define custom commit types beyond conventional commits

2. USER PROMPT TEMPLATE
   - Structure the request
   - Handle optional parameters with {% if %} conditionals
   - Specify output format

   Example modifications:
   • Add 'files_changed' parameter to show affected files
   • Add 'breaking_changes' boolean parameter
   • Request JIRA ticket number extraction

3. PARAMETERS
   - Add new parameters: branch_name, issue_number, etc.
   - Make parameters optional with defaults
   - Use different types: string, number, boolean, array, object

   Example additions:
   • ticket_number (string, optional) - Auto-link to issue tracker
   • is_breaking (boolean) - Flag breaking changes
   • files_limit (number) - Max files to analyze

4. REASONING CONFIGURATION
   - Change strategy: refine, critique, verify, elegance
   - Adjust iterations for quality vs speed
   - Set quality threshold based on criticality

   Example modifications:
   • Use 'verify' strategy for important releases
   • Increase max_iterations to 5 for detailed analysis
   • Lower threshold to 0.75 for faster responses

5. METADATA
   - Categorize for organization
   - Tag for discoverability
   - Track versioning

To modify, just edit the YAML file and the skill reloads automatically!
""")


async def show_skill_integration():
    """Show how to integrate skill into workflow."""
    print("\n" + "=" * 80)
    print("STEP 6: Integration Into Workflow")
    print("=" * 80)

    print("""
Integration Examples:

1. Git Hook Integration (pre-commit):
   ```bash
   #!/bin/bash
   # .git/hooks/prepare-commit-msg

   DIFF=$(git diff --cached)

   python3 << EOF
   import asyncio
   from HoloLoom.agentic import execute_skill

   async def main():
       result = await execute_skill(
           "commit-message-generator",
           {"diff": '''$DIFF'''}
       )
       print(result.output)

   asyncio.run(main())
   EOF
   ```

2. CI/CD Pipeline:
   ```python
   # In your CI script
   result = await execute_skill(
       "commit-message-generator",
       {
           "diff": pr.diff,
           "previous_commits": pr.commits[-5:]
       }
   )

   # Validate PR title matches commit message convention
   if not result.output.startswith(("feat:", "fix:", "docs:")):
       raise ValueError("PR title must follow conventional commits")
   ```

3. IDE Integration (VS Code):
   ```typescript
   // Extension that generates commit messages
   const diff = await gitAPI.getDiff();
   const message = await hololoomClient.executeSkill(
       "commit-message-generator",
       { diff }
   );
   await gitAPI.setCommitMessage(message);
   ```

4. Command-Line Tool:
   ```python
   # commit-gen.py
   import subprocess
   import asyncio
   from HoloLoom.agentic import execute_skill

   async def main():
       # Get git diff
       diff = subprocess.check_output(["git", "diff", "--cached"]).decode()

       # Generate message
       result = await execute_skill(
           "commit-message-generator",
           {"diff": diff}
       )

       print("Suggested commit message:")
       print(result.output)

       # Optionally auto-commit
       if input("Use this message? (y/n): ") == "y":
           subprocess.run(["git", "commit", "-m", result.output])

   asyncio.run(main())
   ```

The skill seamlessly integrates into any workflow that handles git diffs!
""")


async def cleanup():
    """Remove the demo skill file."""
    print("\n" + "=" * 80)
    print("CLEANUP")
    print("=" * 80)

    yaml_path = Path("HoloLoom/agentic/skills/commit-message-generator.yaml")

    if yaml_path.exists():
        choice = input("\nRemove demo skill file? (y/n): ")
        if choice.lower() == 'y':
            yaml_path.unlink()
            print(f"✓ Removed: {yaml_path}")
        else:
            print(f"✓ Kept skill file at: {yaml_path}")
            print("  You can use it with: execute_skill('commit-message-generator', ...)")


async def main():
    """Run the complete custom skill creation tutorial."""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 18 + "CUSTOM SKILL CREATION TUTORIAL" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        # Step 1: Create the skill
        yaml_path = create_skill_yaml_file()

        # Force registry reload to pick up new skill
        print("\nReloading skill registry...")
        registry = await get_registry()
        await registry.load_all_skills()
        print("✓ Skills reloaded")

        # Step 2-4: Test the skill
        await test_skill_basic()
        await test_skill_with_context()
        await test_skill_complex_change()

        # Step 5-6: Explain customization and integration
        await explain_customization()
        await show_skill_integration()

        print("\n\n" + "=" * 80)
        print("TUTORIAL COMPLETE!")
        print("=" * 80)

        print("\nWhat You Learned:")
        print("✓ How to create a skill YAML template")
        print("✓ How to structure system and user prompts")
        print("✓ How to define parameters and reasoning config")
        print("✓ How to test skills with different scenarios")
        print("✓ How to customize skills for your needs")
        print("✓ How to integrate skills into workflows")

        print("\nNext Steps:")
        print("• Create your own custom skill using the template")
        print("• Explore other skills in HoloLoom/agentic/skills/")
        print("• Read SKILLS_USAGE.md for complete guide")
        print("• Check skills/README.md for API reference")

        # Cleanup
        await cleanup()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
