#!/usr/bin/env python3
"""
Elle AR Setup Verification Script
==================================
Verifies that everything is ready for iPhone AR demo testing.

Checks:
1. PC WiFi IP address (warns if WSL/virtual)
2. Backend server running (port 8000)
3. WebSocket connection (/ws/ar)
4. Web server running (port 8080)
5. demo.html configuration

Usage:
    python verify_setup.py
"""

import asyncio
import socket
import subprocess
import sys
from pathlib import Path
import re
import json

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("⚠️  websockets library not available. Install with: pip install websockets")
    print("   Skipping WebSocket connection test...\n")


def get_wifi_ip():
    """Get PC's WiFi IP address"""
    try:
        # Try to connect to external server to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            return ip
    except Exception as e:
        print(f"❌ Could not detect IP: {e}")
        return None


def check_port_in_use(port):
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return False
        except OSError:
            return True


async def test_websocket(url, timeout=3):
    """Test WebSocket connection"""
    if not WEBSOCKETS_AVAILABLE:
        return False, "websockets library not installed"

    try:
        async with asyncio.timeout(timeout):
            async with websockets.connect(url) as ws:
                return True, "Connected successfully"
    except asyncio.TimeoutError:
        return False, f"Connection timeout ({timeout}s)"
    except Exception as e:
        return False, str(e)


def check_demo_html_config(ip):
    """Check if demo.html has correct IP configured"""
    demo_path = Path(__file__).parent / "demo.html"

    if not demo_path.exists():
        return None, "demo.html not found"

    try:
        content = demo_path.read_text(encoding='utf-8')

        # Find BACKEND_URL line
        match = re.search(r"const BACKEND_URL = '(ws://[^']+)'", content)
        if not match:
            return None, "BACKEND_URL not found in demo.html"

        configured_url = match.group(1)
        expected_url = f"ws://{ip}:8000/ws/ar"

        if configured_url == expected_url:
            return True, configured_url
        else:
            return False, f"Configured: {configured_url}\nExpected: {expected_url}"

    except Exception as e:
        return None, f"Error reading demo.html: {e}"


async def main():
    print("=" * 60)
    print("  Elle AR Setup Verification")
    print("=" * 60)
    print()

    results = {
        "ip_valid": False,
        "backend_running": False,
        "websocket_working": False,
        "web_server_running": False,
        "demo_configured": False
    }

    # Step 1: Get WiFi IP
    print("[1/5] Detecting PC WiFi IP address...")
    ip = get_wifi_ip()

    if ip:
        print(f"     ✅ Your PC IP: {ip}")

        # Check if it's a WSL/virtual network IP
        if ip.startswith("172.17.") or ip.startswith("172.18."):
            print(f"     ⚠️  WARNING: {ip} looks like WSL/virtual network")
            print("        Your actual WiFi IP might be different.")
            results["ip_valid"] = False
        else:
            results["ip_valid"] = True
    else:
        print("     ❌ Could not detect IP address")
        return

    print()

    # Step 2: Check backend server
    print("[2/5] Checking if backend server is running...")
    backend_running = check_port_in_use(8000)

    if backend_running:
        print("     ✅ Port 8000 in use (backend likely running)")
        results["backend_running"] = True
    else:
        print("     ❌ Port 8000 free (backend NOT running)")
        print("        Start with: python -m HoloLoom.server.ar_api")
        results["backend_running"] = False

    print()

    # Step 3: Test WebSocket connection
    print("[3/5] Testing WebSocket connection...")
    ws_url = f"ws://{ip}:8000/ws/ar"

    if WEBSOCKETS_AVAILABLE and backend_running:
        success, message = await test_websocket(ws_url)

        if success:
            print(f"     ✅ WebSocket connection successful!")
            print(f"        URL: {ws_url}")
            results["websocket_working"] = True
        else:
            print(f"     ❌ WebSocket connection failed: {message}")
            print(f"        URL: {ws_url}")
            results["websocket_working"] = False
    else:
        if not backend_running:
            print(f"     ⚠️  Skipped (backend not running)")
        else:
            print(f"     ⚠️  Skipped (websockets library not installed)")
        results["websocket_working"] = False

    print()

    # Step 4: Check web server
    print("[4/5] Checking if web server is running...")
    web_server_running = check_port_in_use(8080)

    if web_server_running:
        print("     ✅ Port 8080 in use (web server likely running)")
        print(f"        Demo URL: http://{ip}:8080/demo.html")
        results["web_server_running"] = True
    else:
        print("     ❌ Port 8080 free (web server NOT running)")
        print("        Start with: python -m http.server 8080")
        results["web_server_running"] = False

    print()

    # Step 5: Check demo.html configuration
    print("[5/5] Checking demo.html configuration...")
    configured, details = check_demo_html_config(ip)

    if configured is True:
        print(f"     ✅ demo.html configured correctly")
        print(f"        BACKEND_URL: {details}")
        results["demo_configured"] = True
    elif configured is False:
        print(f"     ⚠️  demo.html IP mismatch:")
        print(f"        {details}")
        results["demo_configured"] = False
    else:
        print(f"     ❌ {details}")
        results["demo_configured"] = False

    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print()

    # Calculate score
    score = sum(results.values())
    total = len(results)

    print(f"Status: {score}/{total} checks passed")
    print()

    if score == total:
        print("✅ ALL SYSTEMS GO! Ready to test on iPhone.")
        print()
        print("Next steps:")
        print("1. Open Safari on iPhone")
        print(f"2. Navigate to: http://{ip}:8080/demo.html")
        print("3. Grant camera permission")
        print("4. Tap 'Start AR Session'")
        print()
        print("⚠️  Note: iPhone Safari requires HTTPS for camera via IP")
        print("   If camera fails, you'll need to:")
        print("   - Use React app (npm run dev)")
        print("   - Deploy to Vercel")
        print("   - Or use ngrok for HTTPS tunnel")
    else:
        print("⚠️  ISSUES DETECTED - Review failures above")
        print()
        print("Quick fixes:")

        if not results["backend_running"]:
            print("  • Start backend: python -m HoloLoom.server.ar_api")

        if not results["web_server_running"]:
            print("  • Start web server: python -m http.server 8080")

        if not results["demo_configured"]:
            print(f"  • Update demo.html line 294 to: ws://{ip}:8000/ws/ar")

        if not results["websocket_working"] and results["backend_running"]:
            print("  • Check firewall: netsh advfirewall firewall add rule name='Elle AR' dir=in action=allow protocol=TCP localport=8000")

    print()
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nVerification cancelled by user.")
        sys.exit(1)
