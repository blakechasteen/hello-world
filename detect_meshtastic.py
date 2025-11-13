#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect and connect to Meshtastic device
"""
import sys
import time
import io

# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Try importing meshtastic
try:
    import meshtastic
    import meshtastic.serial_interface
    print("[OK] Meshtastic module found")
except ImportError as e:
    print(f"[ERROR] Meshtastic not found: {e}")
    print("\nInstall with: pip install meshtastic")
    sys.exit(1)

# List available serial ports
try:
    import serial.tools.list_ports

    print("\n=== Available Serial Ports ===")
    ports = list(serial.tools.list_ports.comports())

    if not ports:
        print("No serial ports found!")
        print("\nMake sure:")
        print("  1. Device is plugged in via USB")
        print("  2. Drivers are installed")
        print("  3. Device shows up in Device Manager")
        sys.exit(1)

    for port in ports:
        print(f"\n  Port: {port.device}")
        print(f"  Description: {port.description}")
        print(f"  Manufacturer: {port.manufacturer}")

        # Check if it looks like a Heltec device
        if any(keyword in str(port).lower() for keyword in ['heltec', 'cp210', 'ch340', 'usb']):
            print(f"  >>> Possible Meshtastic device!")

    print("\n=== Attempting Connection ===")
    print("Trying to auto-detect device...")

    # Try to connect
    try:
        interface = meshtastic.serial_interface.SerialInterface()
        print("\n✓ Successfully connected to device!")

        # Get device info
        print("\n=== Device Information ===")
        if hasattr(interface, 'myInfo') and interface.myInfo:
            print(f"  Node ID: {interface.myInfo.my_node_num}")
            if hasattr(interface.myInfo, 'user'):
                print(f"  Long Name: {interface.myInfo.user.long_name}")
                print(f"  Short Name: {interface.myInfo.user.short_name}")

        # Get node info
        if hasattr(interface, 'nodes'):
            print(f"\n  Total nodes in mesh: {len(interface.nodes)}")

        print("\n✓ Device is online and ready!")
        print("\nNext steps:")
        print("  1. Configure your device name and settings")
        print("  2. Join or create a mesh network")
        print("  3. Set your region (US, EU, etc.)")

        interface.close()

    except Exception as e:
        print(f"\n✗ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check that no other program is using the device")
        print("  2. Try unplugging and replugging the device")
        print("  3. Check Device Manager for driver issues")
        print("  4. Try a different USB cable/port")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
