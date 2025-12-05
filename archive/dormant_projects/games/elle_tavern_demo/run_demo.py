#!/usr/bin/env python3
"""
Elle Tavern Demo - Convenience Startup Script

One-command startup for the complete demo:
1. Checks if Elle service is running
2. Starts Elle service if needed
3. Starts game server
4. Opens browser automatically

Usage:
    python run_demo.py
    python run_demo.py --no-browser  # Don't open browser automatically
    python run_demo.py --port 8080   # Use custom port

Created: 2025-11-16
"""

import os
import sys
import time
import subprocess
import webbrowser
import argparse
import signal
from pathlib import Path
import requests


class DemoRunner:
    """Manages demo startup and lifecycle."""

    def __init__(self, game_port=8001, elle_port=8000, auto_browser=True):
        self.game_port = game_port
        self.elle_port = elle_port
        self.auto_browser = auto_browser
        self.elle_process = None
        self.game_process = None
        self.repo_root = Path(__file__).parent.parent.parent

    def check_port_available(self, port):
        """Check if port is available."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return True
            except OSError:
                return False

    def check_elle_service(self):
        """Check if Elle service is running."""
        try:
            response = requests.get(f'http://localhost:{self.elle_port}/health', timeout=2)
            if response.status_code == 200:
                print(f"✅ Elle service running on port {self.elle_port}")
                return True
        except requests.ConnectionError:
            pass
        except requests.Timeout:
            pass

        return False

    def start_elle_service(self):
        """Start Elle service in background."""
        print(f"\n🚀 Starting Elle service on port {self.elle_port}...")

        elle_dir = self.repo_root / 'apps' / 'elle_game_engine'

        if not elle_dir.exists():
            print(f"❌ Elle service directory not found: {elle_dir}")
            return False

        # Set environment variables
        env = os.environ.copy()
        env['ELLE_LLM_PROVIDER'] = env.get('ELLE_LLM_PROVIDER', 'dummy')
        env['ELLE_VOICE_BACKEND'] = env.get('ELLE_VOICE_BACKEND', 'dummy')

        # Start Elle service
        try:
            self.elle_process = subprocess.Popen(
                [
                    sys.executable, '-m', 'uvicorn',
                    'service:app',
                    '--host', '0.0.0.0',
                    '--port', str(self.elle_port)
                ],
                cwd=str(elle_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for service to start
            print("   Waiting for Elle service to start...", end='', flush=True)
            for i in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                if self.check_elle_service():
                    print(" ✅")
                    return True
                print('.', end='', flush=True)

            print(" ❌")
            print("❌ Elle service failed to start")
            return False

        except Exception as e:
            print(f"❌ Failed to start Elle service: {e}")
            return False

    def start_game_server(self):
        """Start game server in background."""
        print(f"\n🎮 Starting game server on port {self.game_port}...")

        game_dir = self.repo_root / 'games' / 'elle_tavern_demo'

        if not game_dir.exists():
            print(f"❌ Game directory not found: {game_dir}")
            return False

        # Set environment variables
        env = os.environ.copy()
        env['ELLE_API_URL'] = f'http://localhost:{self.elle_port}'

        # Start game server
        try:
            self.game_process = subprocess.Popen(
                [
                    sys.executable, '-m', 'uvicorn',
                    'server:app',
                    '--host', '0.0.0.0',
                    '--port', str(self.game_port)
                ],
                cwd=str(game_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for service to start
            print("   Waiting for game server to start...", end='', flush=True)
            for i in range(20):  # Wait up to 20 seconds
                time.sleep(1)
                try:
                    response = requests.get(f'http://localhost:{self.game_port}/api/health', timeout=1)
                    if response.status_code == 200:
                        print(" ✅")
                        return True
                except:
                    pass
                print('.', end='', flush=True)

            print(" ❌")
            print("❌ Game server failed to start")
            return False

        except Exception as e:
            print(f"❌ Failed to start game server: {e}")
            return False

    def open_browser(self):
        """Open browser to game URL."""
        url = f'http://localhost:{self.game_port}'
        print(f"\n🌐 Opening browser to {url}...")

        time.sleep(2)  # Give server a moment to fully initialize

        try:
            webbrowser.open(url)
            print("✅ Browser opened")
        except Exception as e:
            print(f"⚠️ Could not open browser automatically: {e}")
            print(f"   Please open manually: {url}")

    def cleanup(self):
        """Stop all services."""
        print("\n🛑 Stopping services...")

        if self.game_process:
            print("   Stopping game server...", end='', flush=True)
            self.game_process.terminate()
            try:
                self.game_process.wait(timeout=5)
                print(" ✅")
            except subprocess.TimeoutExpired:
                self.game_process.kill()
                print(" ⚠️ (forced)")

        if self.elle_process:
            print("   Stopping Elle service...", end='', flush=True)
            self.elle_process.terminate()
            try:
                self.elle_process.wait(timeout=5)
                print(" ✅")
            except subprocess.TimeoutExpired:
                self.elle_process.kill()
                print(" ⚠️ (forced)")

    def run(self):
        """Main run loop."""
        print("=" * 60)
        print("🎮 Elle Tavern Demo - Startup")
        print("=" * 60)

        # Check ports
        print("\n📡 Checking ports...")
        if not self.check_port_available(self.game_port):
            print(f"❌ Port {self.game_port} already in use")
            print(f"   Game may already be running, or use --port to specify different port")
            return 1

        # Check/start Elle service
        if not self.check_elle_service():
            if not self.start_elle_service():
                return 1
        else:
            print("   (Already running)")

        # Start game server
        if not self.start_game_server():
            self.cleanup()
            return 1

        # Open browser
        if self.auto_browser:
            self.open_browser()

        # Show info
        print("\n" + "=" * 60)
        print("✅ Demo running!")
        print("=" * 60)
        print(f"\n🌐 Game URL: http://localhost:{self.game_port}")
        print(f"📡 Elle API: http://localhost:{self.elle_port}/docs")
        print("\n💡 Tips:")
        print("   - Press Ctrl+C to stop")
        print("   - Check logs for any errors")
        print("   - Use ELLE_LLM_PROVIDER=openai for real LLM (requires API key)")
        print()

        # Wait for Ctrl+C
        try:
            print("⏳ Running... (Press Ctrl+C to stop)\n")
            while True:
                # Check if processes are still running
                if self.game_process and self.game_process.poll() is not None:
                    print("\n❌ Game server crashed")
                    break
                if self.elle_process and self.elle_process.poll() is not None:
                    print("\n❌ Elle service crashed")
                    break

                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\n⏸️ Stopping demo...")

        finally:
            self.cleanup()
            print("\n👋 Goodbye!\n")

        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Elle Tavern Demo - One-command startup'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8001,
        help='Game server port (default: 8001)'
    )
    parser.add_argument(
        '--elle-port',
        type=int,
        default=8000,
        help='Elle service port (default: 8000)'
    )
    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )

    args = parser.parse_args()

    runner = DemoRunner(
        game_port=args.port,
        elle_port=args.elle_port,
        auto_browser=not args.no_browser
    )

    return runner.run()


if __name__ == '__main__':
    sys.exit(main())
