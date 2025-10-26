"""
Fix Claude Desktop Config
==========================
Creates proper UTF-8 JSON config without BOM.
"""

import json
from pathlib import Path

# Configuration
config = {
    "mcpServers": {
        "holoLoom-memory": {
            "command": "C:/Users/blake/Documents/mythRL/.venv/Scripts/python.exe",
            "args": [
                "c:/Users/blake/Documents/mythRL/HoloLoom/memory/mcp_server_standalone.py"
            ],
            "env": {
                "PYTHONPATH": "c:\\Users\\blake\\Documents\\mythRL"
            }
        }
    }
}

# Write to Claude config location
config_path = Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"
config_path.parent.mkdir(parents=True, exist_ok=True)

# Write with proper UTF-8 encoding (no BOM)
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2)

print(f"✅ Config written to: {config_path}")
print("\n📄 Contents:")
print(config_path.read_text(encoding='utf-8'))

# Validate JSON
try:
    with open(config_path, 'r', encoding='utf-8') as f:
        test = json.load(f)
    print("\n✅ JSON is valid!")
    print(f"✅ Server configured: {list(test['mcpServers'].keys())}")
except json.JSONDecodeError as e:
    print(f"\n❌ JSON error: {e}")
