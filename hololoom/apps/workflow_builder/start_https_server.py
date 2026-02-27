"""
Start the dashboard with HTTPS for Firefox microphone access
"""
import subprocess
import sys
import os

print("=" * 60)
print("  Starting Dashboard with HTTPS")
print("=" * 60)
print()
print("This will:")
print("  1. Generate a self-signed SSL certificate")
print("  2. Start server on https://localhost:8002")
print("  3. You'll need to accept the security warning in Firefox")
print()

# Generate self-signed certificate
print("Generating SSL certificate...")
subprocess.run([
    "openssl", "req", "-x509", "-newkey", "rsa:4096",
    "-keyout", "key.pem", "-out", "cert.pem",
    "-days", "365", "-nodes",
    "-subj", "/CN=localhost"
], check=True)

print("✓ Certificate generated!")
print()
print("Starting server with SSL...")
print("Open: https://localhost:8002")
print("(You'll need to click 'Advanced' → 'Accept the Risk' in Firefox)")
print()

# Start with SSL
os.system("uvicorn agentic_server:app --host 0.0.0.0 --port 8002 --ssl-keyfile key.pem --ssl-certfile cert.pem")
