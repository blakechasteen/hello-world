#!/usr/bin/env python3
"""
Demo: Thanksgiving 2025 Planning

Your actual Thanksgiving plan as executable code
"""

import sys
from pathlib import Path

# Add sous package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sous.app import plan_thanksgiving


def main():
    """Run Thanksgiving planning demo"""
    print("\n🎉 SOUS Demo: Your Thanksgiving 2025 Plan\n")
    print("This demo shows how your actual Thanksgiving planning")
    print("becomes executable SOUS code.\n")

    plan_thanksgiving()

    print("\n💡 What just happened:")
    print("   1. Loaded your Thanksgiving event (Nov 27, 2025)")
    print("   2. Loaded 9 recipes with timing constraints")
    print("   3. Generated shopping lists for 6 stores")
    print("   4. Created day-by-day schedule (Monday-Thursday)")
    print("   5. Detected appliance conflicts (if any)")
    print("\n🎯 All patterns from your manual planning are now automated!")


if __name__ == "__main__":
    main()
