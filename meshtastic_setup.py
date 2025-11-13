# -*- coding: utf-8 -*-
"""Simple Meshtastic device detection"""
import sys

try:
    import meshtastic
    import meshtastic.serial_interface
    import serial.tools.list_ports

    print("="*50)
    print("MESHTASTIC DEVICE DETECTOR")
    print("="*50)

    # List serial ports
    print("\nDetecting serial ports...")
    ports = list(serial.tools.list_ports.comports())

    if not ports:
        print("ERROR: No serial ports found!")
        print("\nMake sure:")
        print("  - Device is plugged in via USB")
        print("  - USB drivers are installed")
        sys.exit(1)

    print(f"\nFound {len(ports)} port(s):")
    for i, port in enumerate(ports, 1):
        print(f"\n  [{i}] {port.device}")
        print(f"      {port.description}")
        if port.manufacturer:
            print(f"      Manufacturer: {port.manufacturer}")

    # Try to connect
    print("\n" + "="*50)
    print("CONNECTING TO DEVICE...")
    print("="*50)

    try:
        print("\nAttempting auto-detection...")
        interface = meshtastic.serial_interface.SerialInterface()

        print("\nSUCCESS! Device connected!")
        print("\n" + "="*50)
        print("DEVICE INFORMATION")
        print("="*50)

        # Wait a moment for info to populate
        import time
        time.sleep(2)

        if hasattr(interface, 'myInfo') and interface.myInfo:
            print(f"\nNode ID: {interface.myInfo.my_node_num}")
            if hasattr(interface.myInfo, 'user'):
                print(f"Long Name: {interface.myInfo.user.long_name}")
                print(f"Short Name: {interface.myInfo.user.short_name}")

        if hasattr(interface, 'nodes'):
            print(f"\nNodes in mesh: {len(interface.nodes)}")

        print("\n" + "="*50)
        print("DEVICE IS ONLINE!")
        print("="*50)
        print("\nNext Steps:")
        print("  1. Set your device name")
        print("  2. Configure your region (US/EU/etc)")
        print("  3. Join or create a mesh network")

        interface.close()

    except Exception as e:
        print(f"\nERROR: Could not connect to device")
        print(f"Details: {e}")
        print("\nTroubleshooting:")
        print("  - Unplug and replug the device")
        print("  - Close any other apps using the device")
        print("  - Check Device Manager for issues")
        print("  - Try a different USB port/cable")

except ImportError as e:
    print(f"ERROR: Missing module - {e}")
    print("\nRun: pip install meshtastic")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
