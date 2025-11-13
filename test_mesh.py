# -*- coding: utf-8 -*-
"""Test mesh network with both devices"""
import sys
import time

print("="*60)
print("MESH NETWORK TEST")
print("="*60)

print("\nPlease ensure:")
print("  - BOTH devices are plugged in")
print("  - They are within ~10 feet of each other")
print("  - Wait 30-60 seconds for discovery")

print("\nWaiting 5 seconds before connecting...")
time.sleep(5)

try:
    import meshtastic
    import meshtastic.serial_interface
    import serial.tools.list_ports

    # Find all Heltec devices
    ports = list(serial.tools.list_ports.comports())
    heltec_ports = []

    for port in ports:
        if 'CP210x' in str(port.description) or 'CH340' in str(port.description):
            heltec_ports.append(port.device)

    print(f"\nFound {len(heltec_ports)} Heltec device(s): {heltec_ports}")

    if len(heltec_ports) < 2:
        print("\nWARNING: Only found 1 device!")
        print("Make sure both devices are plugged in.")
        if len(heltec_ports) == 1:
            print(f"\nWill connect to the one device found: {heltec_ports[0]}")
        else:
            print("\nNo devices found!")
            sys.exit(1)

    # Connect to first available device
    test_port = heltec_ports[0]
    print(f"\nConnecting to {test_port}...")

    interface = meshtastic.serial_interface.SerialInterface(test_port)

    print("\nWaiting for node info to populate...")
    time.sleep(5)

    print("\n" + "="*60)
    print("MESH NETWORK STATUS")
    print("="*60)

    # Get current device info
    if hasattr(interface, 'myInfo') and interface.myInfo:
        my_node = interface.myInfo.my_node_num
        print(f"\nConnected to Node: !{my_node:08x}")

        if hasattr(interface.myInfo, 'user'):
            user = interface.myInfo.user
            print(f"Device Name: {user.long_name}")

    # Check for other nodes
    if hasattr(interface, 'nodes'):
        nodes = interface.nodes
        print(f"\n\nTotal nodes in mesh: {len(nodes)}")

        if len(nodes) > 1:
            print("\n SUCCESS! Multiple nodes detected!")
            print("\nNodes in your mesh:")
            for node_id, node_info in nodes.items():
                if hasattr(node_info, 'user'):
                    name = node_info.user.long_name if hasattr(node_info.user, 'long_name') else 'Unknown'
                    print(f"  - {node_id}: {name}")
        else:
            print("\n Only 1 node detected so far.")
            print("\nIf both devices are plugged in:")
            print("  - Wait 30-60 more seconds")
            print("  - Devices need to discover each other")
            print("  - Check that both have LoRa antennas attached!")
            print("  - Make sure they're close together (within 10 feet)")

    interface.close()

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)

    if len(heltec_ports) >= 2:
        print("\n Both devices detected on USB!")
        print("\nTo test messaging:")
        print("  1. Keep both plugged in")
        print("  2. Wait 60 seconds for mesh discovery")
        print("  3. Send test message (see below)")

        print("\n\nTo send a message from one device:")
        print("  python -m meshtastic --sendtext 'Hello from Node 1!'")

        print("\n\nTo monitor messages:")
        print("  python -m meshtastic --listen")
    else:
        print("\n Only found 1 device on USB.")
        print("\nOptions:")
        print("  1. Unplug second device - it will continue on battery")
        print("  2. Use Meshtastic mobile app to see both nodes")
        print("  3. They should still mesh via LoRa radio")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
