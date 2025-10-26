#!/usr/bin/env python3
"""
Extended Promptly CLI with ultraprompt command
Usage: python promptly_cli.py ultraprompt [ollama|llm] "your request" [--execute]
"""

import sys
import subprocess
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the ultraprompt functions
try:
    from ultraprompt_ollama import ultraprompt_with_ollama, check_ollama
except ImportError:
    ultraprompt_with_ollama = None
    check_ollama = None

try:
    from ultraprompt_llm import ultraprompt_with_llm
except ImportError:
    ultraprompt_with_llm = None

# Import base promptly
from promptly import cli, Promptly


def skill_command():
    """Handle skill commands"""
    if len(sys.argv) < 3:
        print("Usage: python promptly_cli.py skill <command> [args]")
        print("\nCommands:")
        print("  add <name> [description]          - Add a new skill")
        print("  list                               - List all skills")
        print("  get <name> [version]              - Get skill details")
        print("  add-file <skill> <filepath>       - Attach file to skill")
        print("  files <skill>                     - List skill files")
        print("\nExamples:")
        print('  python promptly_cli.py skill add "data_analysis" "Analyze CSV data"')
        print('  python promptly_cli.py skill add-file "data_analysis" ./script.py')
        print('  python promptly_cli.py skill list')
        sys.exit(1)
    
    cmd = sys.argv[2].lower()
    promptly = Promptly()
    
    try:
        if cmd == "add":
            if len(sys.argv) < 4:
                print("Error: skill name required")
                sys.exit(1)
            name = sys.argv[3]
            description = sys.argv[4] if len(sys.argv) > 4 else None
            # Set Claude as default runtime
            metadata = {"runtime": "claude"}
            result = promptly.add_skill(name, description, metadata)
            print(result)
        
        elif cmd == "list":
            skills = promptly.list_skills()
            if not skills:
                print("No skills found")
            else:
                print(f"{'Name':<30} {'Version':<10} {'Commit':<15} {'Created'}")
                print("-" * 80)
                for s in skills:
                    print(f"{s['name']:<30} v{s['version']:<9} {s['commit_hash']:<15} {s['created_at'][:19]}")
        
        elif cmd == "get":
            if len(sys.argv) < 4:
                print("Error: skill name required")
                sys.exit(1)
            name = sys.argv[3]
            version = int(sys.argv[4]) if len(sys.argv) > 4 else None
            skill = promptly.get_skill(name, version=version)
            if not skill:
                print(f"Skill '{name}' not found")
                sys.exit(1)
            print(f"Name: {skill['name']}")
            print(f"Description: {skill.get('description', 'N/A')}")
            print(f"Version: {skill['version']}")
            print(f"Branch: {skill['branch']}")
            print(f"Commit: {skill['commit_hash']}")
            print(f"Created: {skill['created_at']}")
            print(f"Metadata: {skill.get('metadata', {})}")
        
        elif cmd == "add-file":
            if len(sys.argv) < 5:
                print("Error: skill name and filepath required")
                sys.exit(1)
            skill_name = sys.argv[3]
            filepath = sys.argv[4]
            result = promptly.add_skill_file(skill_name, filepath)
            print(result)
        
        elif cmd == "files":
            if len(sys.argv) < 4:
                print("Error: skill name required")
                sys.exit(1)
            skill_name = sys.argv[3]
            files = promptly.get_skill_files(skill_name)
            if not files:
                print(f"No files attached to skill '{skill_name}'")
            else:
                print(f"{'Filename':<40} {'Type':<10} {'Created'}")
                print("-" * 80)
                for f in files:
                    print(f"{f['filename']:<40} {f['filetype']:<10} {f['created_at'][:19]}")
        
        else:
            print(f"Error: Unknown command '{cmd}'")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def ultraprompt_command():
    """Handle ultraprompt command"""
    if len(sys.argv) < 4:
        print("Usage: python promptly_cli.py ultraprompt [ollama|llm] \"your request\" [--execute] [--advanced]")
        print("\nExamples:")
        print('  python promptly_cli.py ultraprompt ollama "write a fibonacci function"')
        print('  python promptly_cli.py ultraprompt ollama "explain quantum computing" --execute')
        print('  python promptly_cli.py ultraprompt ollama "medical diagnosis guide" --advanced')
        print('  python promptly_cli.py ultraprompt llm "create a REST API"')
        print("\nFlags:")
        print("  --execute, -e   Execute the expanded prompt")
        print("  --advanced, -a  Use ultraprompt-advanced template (detailed)")
        sys.exit(1)
    
    backend = sys.argv[2].lower()
    request = sys.argv[3]
    execute = "--execute" in sys.argv or "-e" in sys.argv
    advanced = "--advanced" in sys.argv or "-a" in sys.argv
    
    # Find model if specified
    model = None
    for i, arg in enumerate(sys.argv[4:], 4):
        if not arg.startswith("-") and arg not in ["--execute", "-e", "--advanced", "-a"]:
            model = arg
            break
    
    if backend == "ollama":
        if not ultraprompt_with_ollama:
            print("Error: ultraprompt_ollama module not found")
            sys.exit(1)
        
        if not check_ollama():
            print("Error: Ollama not available")
            sys.exit(1)
        
        template = "ultraprompt-advanced" if advanced else "ultraprompt"
        result = ultraprompt_with_ollama(
            request,
            model=model or "llama3.2:3b",
            execute=execute,
            template=template
        )
        
        if result:
            sys.exit(0)
        else:
            sys.exit(1)
    
    elif backend == "llm":
        if not ultraprompt_with_llm:
            print("Error: ultraprompt_llm module not found")
            sys.exit(1)
        
        result = ultraprompt_with_llm(request, model=model)
        
        if result:
            sys.exit(0)
        else:
            sys.exit(1)
    
    else:
        print(f"Error: Unknown backend '{backend}'. Use 'ollama' or 'llm'")
        sys.exit(1)


if __name__ == '__main__':
    # Check if this is an ultraprompt command
    if len(sys.argv) > 1 and sys.argv[1] == "ultraprompt":
        ultraprompt_command()
    elif len(sys.argv) > 1 and sys.argv[1] == "skill":
        skill_command()
    else:
        # Pass through to original promptly CLI
        cli()
