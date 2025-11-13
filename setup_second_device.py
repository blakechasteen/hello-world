# -*- coding: utf-8 -*-
"""Setup second Meshtastic device"""
import sys
import time

print("="*60)
print("SECOND HELTEC DEVICE SETUP")
print("="*60)

print("\nSTEPS:")
print("  1. UNPLUG the first device (disconnect USB)")
print("  2. PLUG IN the second device")
print("  3. Press Enter when ready...")

input()

print("\nChecking for device...")

try:
    import meshtastic
    import meshtastic.serial_interface
    import serial.tools.list_ports

    # Find COM port
    ports = list(serial.tools.list_ports.comports())
    com_port = None

    for port in ports:
        if 'CP210x' in port.description or 'CH340' in port.description:
            com_port = port.device
            print(f"\nFound device on: {com_port}")
            print(f"Description: {port.description}")
            break

    if not com_port:
        print("\nERROR: No device found!")
        print("Make sure:")
        print("  - Second device is plugged in")
        print("  - First device is unplugged")
        sys.exit(1)

    print("\n" + "="*60)
    print("CONNECTING TO SECOND DEVICE")
    print("="*60)

    print(f"\nConnecting to {com_port}...")
    interface = meshtastic.serial_interface.SerialInterface(com_port)

    print("\nSUCCESS! Second device connected!")
    print("\nWaiting for device info...")
    time.sleep(3)

    # Get info
    print("\n" + "="*60)
    print("SECOND DEVICE INFORMATION")
    print("="*60)

    if hasattr(interface, 'myInfo') and interface.myInfo:
        node_id = interface.myInfo.my_node_num
        print(f"\nNode ID: !{node_id:08x}")

        if hasattr(interface.myInfo, 'user'):
            user = interface.myInfo.user
            print(f"Current Name: {user.long_name}")
            print(f"Hardware: {user.hw_model}")

    interface.close()
    print("\n" + "="*60)
    print("SECOND DEVICE DETECTED!")
    print("="*60)

    print(f"\nYour second device is on: {com_port}")
    print("\nNow flash firmware:")
    print("  1. Go to: https://flasher.meshtastic.org")
    print("  2. Select HELTEC_V3 (or your model)")
    print(f"  3. Connect to {com_port}")
    print("  4. Flash firmware")
    print("\nAfter flashing:")
    print("  - Unplug and replug device")
    print("  - Run: python configure_second_device.py")

except Exception as e:
    print(f"\nError: {e}")
    print("\nIf device needs firmware:")
    print("  1. Note which COM port it's on")
    print("  2. Flash via https://flasher.meshtastic.org")
    print("  3. Unplug/replug")
    print("  4. Run this script again")
    import traceback
    traceback.print_exc()
