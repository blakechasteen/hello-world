# -*- coding: utf-8 -*-
"""Auto-detect second device"""
import sys
import time

print("="*60)
print("SECOND DEVICE DETECTION")
print("="*60)

print("\nMake sure:")
print("  - First device is UNPLUGGED")
print("  - Second device is PLUGGED IN")

try:
    import serial.tools.list_ports

    print("\nScanning for devices...")
    time.sleep(1)

    ports = list(serial.tools.list_ports.comports())

    print(f"\nFound {len(ports)} serial port(s):")

    for port in ports:
        print(f"\n  {port.device}")
        print(f"  {port.description}")
        if port.manufacturer:
            print(f"  Manufacturer: {port.manufacturer}")

    # Look for Heltec device
    heltec_port = None
    for port in ports:
        if 'CP210x' in str(port.description) or 'CH340' in str(port.description):
            heltec_port = port.device
            print(f"\n>>> Found Heltec device on: {heltec_port}")
            break

    if not heltec_port:
        print("\nNo Heltec device found!")
        print("\nCheck:")
        print("  - Is second device plugged in?")
        print("  - Is first device unplugged?")
        print("  - Check Device Manager")
        sys.exit(1)

    # Try to connect
    print("\n" + "="*60)
    print("TESTING CONNECTION")
    print("="*60)

    try:
        import meshtastic
        import meshtastic.serial_interface

        print(f"\nConnecting to {heltec_port}...")
        interface = meshtastic.serial_interface.SerialInterface(heltec_port)

        print("\n✓ SUCCESS! Device has Meshtastic firmware!")

        time.sleep(2)

        if hasattr(interface, 'myInfo') and interface.myInfo:
            node_id = interface.myInfo.my_node_num
            print(f"\nNode ID: !{node_id:08x}")

        interface.close()

        print("\n" + "="*60)
        print("SECOND DEVICE IS READY!")
        print("="*60)
        print("\nDevice is already flashed with Meshtastic.")
        print("\nNext: Configure it by running:")
        print("  python configure_second_device.py")

    except Exception as e:
        if "Timed out" in str(e) or "No module named" in str(e):
            print("\n⚠ Device needs Meshtastic firmware!")
            print("\n" + "="*60)
            print("FLASH FIRMWARE NOW")
            print("="*60)
            print(f"\n1. Go to: https://flasher.meshtastic.org")
            print(f"2. Select: HELTEC_V3")
            print(f"3. Connect to: {heltec_port}")
            print(f"4. Click 'Flash'")
            print(f"\nAfter flashing:")
            print(f"  - Unplug and replug device")
            print(f"  - Run: python configure_second_device.py")
        else:
            raise

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
