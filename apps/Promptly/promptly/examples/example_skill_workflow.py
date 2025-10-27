#!/usr/bin/env python3
"""
Example: Creating and executing a Claude skill with Promptly
This demonstrates the complete workflow from skill creation to execution
"""

from promptly import Promptly
from pathlib import Path
import json

# Example executor function (you'd replace this with actual Claude API)
def example_executor(prompt, model):
    """
    Mock executor - replace with your actual Claude API call
    """
    print(f"\n{'='*70}")
    print(f"EXECUTING WITH MODEL: {model}")
    print(f"{'='*70}")
    print("PROMPT:")
    print(prompt)
    print(f"{'='*70}\n")
    
    # In production, this would be:
    # import anthropic
    # client = anthropic.Anthropic(api_key="your-key")
    # message = client.messages.create(model=model, max_tokens=1024, messages=[...])
    # return message.content[0].text
    
    return f"[Mock response from {model}] Skill executed successfully!"


def main():
    print("\n" + "="*70)
    print("Promptly Skills for Claude - Complete Example")
    print("="*70 + "\n")
    
    # Initialize Promptly
    p = Promptly()
    
    # Check if already initialized
    if not p.promptly_dir.exists():
        print("Initializing Promptly repository...")
        p.init()
        print("✓ Initialized\n")
    else:
        print("✓ Promptly already initialized\n")
    
    # Step 1: Create a skill
    print("Step 1: Creating a skill...")
    skill_name = "example_data_processor"
    
    try:
        result = p.add_skill(
            name=skill_name,
            description="Process and analyze structured data with Claude",
            metadata={
                "runtime": "claude",
                "model": "claude-3-5-sonnet-20241022",
                "version": "1.0.0",
                "tags": ["data", "analysis", "example"],
                "author": "Promptly Example"
            }
        )
        print(f"✓ {result}\n")
    except Exception as e:
        print(f"Note: {e}\n")
    
    # Step 2: Create and attach example files
    print("Step 2: Creating and attaching example files...")
    
    # Create example Python script
    example_script = """#!/usr/bin/env python3
\"\"\"
Example data processor script
\"\"\"

def process_data(data):
    \"\"\"Process the input data\"\"\"
    results = []
    for item in data:
        processed = {
            'original': item,
            'processed': item.upper(),
            'length': len(item)
        }
        results.append(processed)
    return results

if __name__ == '__main__':
    sample = ['hello', 'world', 'claude']
    print(process_data(sample))
"""
    
    # Create example config
    example_config = {
        "processing_rules": {
            "uppercase": True,
            "calculate_length": True,
            "remove_duplicates": False
        },
        "output_format": "json"
    }
    
    # Create example .sip file (custom format)
    example_sip = """# Example .sip file (Skill Instruction Protocol)
# This is a custom file format for skill instructions

SKILL: data_processor
VERSION: 1.0
RUNTIME: claude

INSTRUCTIONS:
1. Read the input data
2. Apply transformation rules from config.json
3. Execute processor.py logic
4. Return results in specified output format

EXPECTED_INPUT:
- Array of strings or objects
- Valid JSON format

EXPECTED_OUTPUT:
- Processed data structure
- Metadata about processing
"""
    
    # Save files temporarily
    script_path = Path("example_processor.py")
    config_path = Path("example_config.json")
    sip_path = Path("example_instructions.sip")
    
    script_path.write_text(example_script)
    config_path.write_text(json.dumps(example_config, indent=2))
    sip_path.write_text(example_sip)
    
    # Attach files to skill
    try:
        p.add_skill_file(skill_name, str(script_path))
        print(f"✓ Attached {script_path.name}")
        
        p.add_skill_file(skill_name, str(config_path))
        print(f"✓ Attached {config_path.name}")
        
        p.add_skill_file(skill_name, str(sip_path))
        print(f"✓ Attached {sip_path.name}\n")
    except Exception as e:
        print(f"Note: {e}\n")
    
    # Step 3: List skills
    print("Step 3: Listing all skills...")
    skills = p.list_skills()
    for skill in skills:
        print(f"  • {skill['name']} (v{skill['version']}) - {skill['commit_hash']}")
    print()
    
    # Step 4: Get skill details
    print("Step 4: Getting skill details...")
    skill = p.get_skill(skill_name)
    print(f"  Name: {skill['name']}")
    print(f"  Description: {skill['description']}")
    print(f"  Version: {skill['version']}")
    print(f"  Runtime: {skill['metadata'].get('runtime', 'N/A')}")
    print(f"  Tags: {', '.join(skill['metadata'].get('tags', []))}\n")
    
    # Step 5: Validate for Claude
    print("Step 5: Validating skill for Claude...")
    is_valid = p.validate_skill_for_claude(skill_name)
    print(f"  Valid for Claude: {'✓ Yes' if is_valid else '✗ No'}\n")
    
    # Step 6: List skill files
    print("Step 6: Listing skill files...")
    files = p.get_skill_files(skill_name)
    for f in files:
        print(f"  • {f['filename']} ({f['filetype']})")
    print()
    
    # Step 7: Prepare skill payload
    print("Step 7: Preparing skill payload...")
    payload = p.prepare_skill_payload(skill_name)
    print(f"  Skill: {payload['skill_name']}")
    print(f"  Files loaded: {len(payload['files'])}")
    print(f"  Total content size: {sum(len(f['content']) for f in payload['files'])} chars\n")
    
    # Step 8: Execute skill
    print("Step 8: Executing skill...")
    result = p.execute_skill(
        skill_name=skill_name,
        user_input="Process the following data: ['apple', 'banana', 'cherry']",
        model="claude-3-5-sonnet-20241022",
        executor_func=example_executor
    )
    
    print("\nExecution Result:")
    print(f"  Model: {result['model']}")
    print(f"  Prompt length: {len(result['prompt'])} chars")
    print(f"  Result: {result['result']}\n")
    
    # Cleanup temporary files
    print("Cleaning up example files...")
    script_path.unlink()
    config_path.unlink()
    sip_path.unlink()
    print("✓ Cleanup complete\n")
    
    print("="*70)
    print("Example Complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Replace example_executor with real Claude API call")
    print("  2. Create your own skills with actual code/data files")
    print("  3. Use 'python promptly_cli.py skill' commands for CLI access")
    print("  4. Explore versioning and branching for skill evolution")
    print("\nSee SKILLS.md for full documentation!")
    print()


if __name__ == '__main__':
    main()
