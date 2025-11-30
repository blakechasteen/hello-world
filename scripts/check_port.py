# -*- coding: utf-8 -*-
"""Check if COM3 is available"""
import serial.tools.list_ports
import serial

print("Checking COM3 status...\n")

# List all ports
ports = list(serial.tools.list_ports.comports())
for port in ports:
    if 'COM3' in port.device:
        print(f"Found: {port.device}")
        print(f"  Description: {port.description}")
        print(f"  In use: Checking...")

        # Try to open briefly
        try:
            test = serial.Serial(port.device, timeout=1)
            test.close()
            print("  Status: AVAILABLE (ready to connect!)")
            print("\nYou can now run: python test_connection.py")
        except serial.SerialException as e:
            print(f"  Status: LOCKED (in use by another program)")
            print(f"\n  Error: {e}")
            print("\nTo fix:")
            print("  1. Close web flasher browser tab")
            print("  2. Wait 10 seconds")
            print("  OR")
            print("  1. Unplug USB cable")
            print("  2. Wait 5 seconds")
            print("  3. Plug back in")
