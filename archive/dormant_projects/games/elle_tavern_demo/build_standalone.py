"""
Elle Tavern Demo - Standalone Build Script

Creates a standalone executable for distribution on Windows, macOS, and Linux.

Usage:
    python build_standalone.py

Output:
    - dist/elle-tavern-demo (executable)
    - dist/elle-tavern-demo.zip (distribution package)

Created: 2025-11-16
"""

import os
import sys
import shutil
import zipfile
import platform
from pathlib import Path


def check_pyinstaller():
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller
        print("✅ PyInstaller found")
        return True
    except ImportError:
        print("❌ PyInstaller not found. Installing...")
        os.system(f"{sys.executable} -m pip install pyinstaller")
        return True


def clean_build_dirs():
    """Clean previous build artifacts."""
    print("\n🧹 Cleaning build directories...")

    dirs_to_clean = ['build', 'dist', '__pycache__']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"  - Removed {dir_name}/")

    print("✅ Clean complete")


def collect_data_files():
    """Collect all data files needed for distribution."""
    print("\n📦 Collecting data files...")

    data_files = []

    # Static files (HTML, CSS, JS)
    if os.path.exists('static'):
        data_files.append(('static', 'static'))
        print("  - Added static/")

    # Configuration files
    for file in ['requirements.txt', 'README.md', 'DEPLOYMENT.md']:
        if os.path.exists(file):
            data_files.append((file, '.'))
            print(f"  - Added {file}")

    # Game data files
    for file in ['world_data.py', 'quests.py', 'game_engine.py', 'integration.py']:
        if os.path.exists(file):
            data_files.append((file, '.'))
            print(f"  - Added {file}")

    return data_files


def create_spec_file(data_files):
    """Create PyInstaller spec file."""
    print("\n📝 Creating PyInstaller spec file...")

    spec_content = f"""
# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['server.py'],
    pathex=[],
    binaries=[],
    datas={data_files},
    hiddenimports=[
        'fastapi',
        'uvicorn',
        'pydantic',
        'aiohttp',
        'httpx',
        'games.elle_tavern_demo.world_data',
        'games.elle_tavern_demo.quests',
        'games.elle_tavern_demo.game_engine',
        'games.elle_tavern_demo.integration',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='elle-tavern-demo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
"""

    with open('elle-tavern-demo.spec', 'w') as f:
        f.write(spec_content)

    print("✅ Spec file created")


def build_executable():
    """Build executable using PyInstaller."""
    print("\n🔨 Building executable...")
    print(f"   Platform: {platform.system()}")
    print(f"   Python: {sys.version.split()[0]}")

    # Run PyInstaller
    os.system("pyinstaller --clean --noconfirm elle-tavern-demo.spec")

    # Check if build succeeded
    exe_name = 'elle-tavern-demo.exe' if platform.system() == 'Windows' else 'elle-tavern-demo'
    exe_path = os.path.join('dist', exe_name)

    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"\n✅ Build successful!")
        print(f"   Executable: {exe_path}")
        print(f"   Size: {size_mb:.1f} MB")
        return True
    else:
        print("\n❌ Build failed - executable not found")
        return False


def create_readme():
    """Create README for distribution."""
    readme_content = """# Elle Tavern Demo - Standalone Distribution

## Quick Start

### Windows
Double-click `elle-tavern-demo.exe`

### macOS / Linux
```bash
chmod +x elle-tavern-demo
./elle-tavern-demo
```

Then open browser to: http://localhost:8001

## Requirements

This standalone distribution includes everything needed to run the game.

**Optional**: For full LLM integration, you need:
- OpenAI API key (set OPENAI_API_KEY environment variable)
- Or Anthropic API key (set ANTHROPIC_API_KEY environment variable)

## Configuration

Create a `.env` file in the same directory:

```
ELLE_LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
ELLE_VOICE_BACKEND=openai
```

Or use dummy mode (no API key needed):

```
ELLE_LLM_PROVIDER=dummy
```

## Troubleshooting

### Port Already in Use
```bash
# Windows
netstat -ano | findstr :8001

# macOS / Linux
lsof -i :8001
```

### Firewall Blocking
Allow the executable through your firewall.

### Performance Issues
- Close other applications
- Check CPU/memory usage
- Use dummy LLM provider for testing

## Support

See DEPLOYMENT.md for complete documentation.

---
**Version**: 1.0.0
**Built**: {platform.system()} {platform.machine()}
"""

    with open('dist/README.txt', 'w') as f:
        f.write(readme_content)

    print("✅ README created")


def create_distribution_package():
    """Create final distribution ZIP file."""
    print("\n📦 Creating distribution package...")

    zip_name = f'elle-tavern-demo-{platform.system().lower()}.zip'
    zip_path = os.path.join('dist', zip_name)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add executable
        exe_name = 'elle-tavern-demo.exe' if platform.system() == 'Windows' else 'elle-tavern-demo'
        exe_path = os.path.join('dist', exe_name)
        if os.path.exists(exe_path):
            zipf.write(exe_path, exe_name)
            print(f"  - Added {exe_name}")

        # Add README
        if os.path.exists('dist/README.txt'):
            zipf.write('dist/README.txt', 'README.txt')
            print("  - Added README.txt")

        # Add static files if they exist
        if os.path.exists('static'):
            for root, dirs, files in os.walk('static'):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path)
                    zipf.write(file_path, arcname)
                    print(f"  - Added {arcname}")

        # Add deployment guide
        if os.path.exists('DEPLOYMENT.md'):
            zipf.write('DEPLOYMENT.md', 'DEPLOYMENT.md')
            print("  - Added DEPLOYMENT.md")

    if os.path.exists(zip_path):
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"\n✅ Distribution package created!")
        print(f"   Package: {zip_path}")
        print(f"   Size: {size_mb:.1f} MB")
        return True
    else:
        print("\n❌ Failed to create distribution package")
        return False


def main():
    """Main build process."""
    print("=" * 60)
    print("Elle Tavern Demo - Standalone Build")
    print("=" * 60)

    # Check PyInstaller
    if not check_pyinstaller():
        print("❌ Failed to install PyInstaller")
        return 1

    # Clean previous builds
    clean_build_dirs()

    # Collect data files
    data_files = collect_data_files()

    # Create spec file
    create_spec_file(data_files)

    # Build executable
    if not build_executable():
        print("\n❌ Build failed")
        return 1

    # Create README
    create_readme()

    # Create distribution package
    if not create_distribution_package():
        print("\n⚠️ Distribution package creation failed")

    print("\n" + "=" * 60)
    print("✅ Build complete!")
    print("=" * 60)
    print("\nDistribution files:")
    print("  - dist/elle-tavern-demo (executable)")
    print(f"  - dist/elle-tavern-demo-{platform.system().lower()}.zip (package)")
    print("\nTo distribute:")
    print("  1. Test the executable")
    print("  2. Upload the .zip file to itch.io, GitHub Releases, etc.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
