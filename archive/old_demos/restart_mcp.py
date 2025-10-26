"""
Restart MCP Server for Claude
==============================
Kill the current MCP server process and let Claude restart it.
"""

import subprocess
import sys

print("🔄 Restarting MCP Server...")

# Find and kill python processes running mcp_server_standalone.py
result = subprocess.run(
    ['powershell', '-Command', 
     "Get-Process python | Where-Object { $_.Path -like '*mythRL*' } | Stop-Process -Force"],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("✅ Old server process killed")
else:
    print("⚠️  No running server found (or already stopped)")

print("\n📋 Next steps:")
print("1. Claude Desktop will auto-restart the server")
print("2. Try your command again in Claude")
print("3. Check logs: Get-Content \"$env:APPDATA\\Claude\\logs\\mcp-server-holoLoom-memory.log\" -Tail 20")

print("\n✨ Server ready to restart!")
