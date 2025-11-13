# -*- coding: utf-8 -*-
"""Configure second device after flashing"""
import sys
import time

print("="*60)
print("CONFIGURE SECOND DEVICE")
print("="*60)

print("\nMake sure:")
print("  - Second device is plugged in (first unplugged)")
print("  - Firmware has been flashed")
print("  - Device has been unplugged/replugged after flash")

input("\nPress Enter to continue...")

try:
    import meshtastic
    import meshtastic.serial_interface
    import serial.tools.list_ports

    # Auto-detect port
    ports = list(serial.tools.list_ports.comports())
    com_port = None

    for port in ports:
        if 'CP210x' in port.description or 'CH340' in port.description:
            com_port = port.device
            break

    if not com_port:
        print("\nNo device found! Make sure it's plugged in.")
        sys.exit(1)

    print(f"\nConnecting to {com_port}...")
    interface = meshtastic.serial_interface.SerialInterface(com_port)

    print("\nConnected! Configuring device...")
    time.sleep(2)

    # Get node ID
    node_id = None
    if hasattr(interface, 'myInfo') and interface.myInfo:
        node_id = interface.myInfo.my_node_num
        print(f"Node ID: !{node_id:08x}")

    interface.close()

    # Now configure via CLI
    print("\n" + "="*60)
    print("SETTING CONFIGURATION")
    print("="*60)

    import subprocess

    # Set region
    print("\n1. Setting region to US...")
    subprocess.run([
        "C:\\Users\\blake\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
        "-m", "meshtastic",
        "--set", "lora.region", "US"
    ])

    time.sleep(2)

    # Set name
    print("\n2. Setting device name...")
    subprocess.run([
        "C:\\Users\\blake\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
        "-m", "meshtastic",
        "--set-owner", "Blake Node 2",
        "--set-owner-short", "BLK2"
    ])

    print("\n" + "="*60)
    print("SECOND DEVICE CONFIGURED!")
    print("="*60)

    print("\n✓ Region: US")
    print("✓ Name: Blake Node 2 (BLK2)")
    print(f"✓ Node ID: !{node_id:08x}" if node_id else "")

    print("\n" + "="*60)
    print("TESTING MESH COMMUNICATION")
    print("="*60)

    print("\nNow:")
    print("  1. Keep this device plugged in")
    print("  2. Plug in the FIRST device too")
    print("  3. Wait 30 seconds")
    print("  4. They should discover each other!")
    print("\nCheck device screens - you should see 2 nodes!")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
