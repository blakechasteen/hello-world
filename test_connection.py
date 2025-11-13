# -*- coding: utf-8 -*-
"""Quick test after flashing"""
import sys
import time

print("Waiting 3 seconds for you to close the web flasher...")
time.sleep(3)

try:
    import meshtastic
    import meshtastic.serial_interface

    print("\n" + "="*60)
    print("TESTING MESHTASTIC CONNECTION")
    print("="*60)

    print("\nAttempting to connect to COM3...")
    print("(This should work now that firmware is flashed!)\n")

    interface = meshtastic.serial_interface.SerialInterface("COM3")

    print("\n" + "="*60)
    print("SUCCESS! MESHTASTIC IS ONLINE!")
    print("="*60)

    print("\nWaiting for device info to populate...")
    time.sleep(3)

    # Get device info
    print("\n" + "="*60)
    print("DEVICE INFORMATION")
    print("="*60)

    if hasattr(interface, 'myInfo') and interface.myInfo:
        node_id = interface.myInfo.my_node_num
        print(f"\nNode ID (hex): !{node_id:08x}")
        print(f"Node ID (decimal): {node_id}")

        if hasattr(interface.myInfo, 'user'):
            user = interface.myInfo.user
            print(f"\nLong Name: {user.long_name}")
            print(f"Short Name: {user.short_name}")
            print(f"Hardware Model: {user.hw_model}")

    # Show nodes in mesh
    if hasattr(interface, 'nodes') and interface.nodes:
        print(f"\n\nTotal nodes in mesh: {len(interface.nodes)}")
        print("(Just you for now - you'll see others as you join networks)")

    print("\n" + "="*60)
    print("YOUR DEVICE IS READY!")
    print("="*60)

    print("\nNext steps:")
    print("  1. Set your name (see below)")
    print("  2. Configure region (IMPORTANT!)")
    print("  3. Join a mesh network or create your own")

    print("\n" + "="*60)
    print("QUICK CONFIGURATION COMMANDS")
    print("="*60)

    print("\nTo set your name:")
    print('  python -m meshtastic --set-owner "Your Name" --set-owner-short "YN"')

    print("\nTo set region (REQUIRED):")
    print("  For US:")
    print("    python -m meshtastic --set lora.region US")
    print("  For EU:")
    print("    python -m meshtastic --set lora.region EU_868")

    print("\nTo see all settings:")
    print("  python -m meshtastic --info")

    print("\nTo send a test message:")
    print('  python -m meshtastic --sendtext "Hello mesh!"')

    interface.close()
    print("\n\nConnection closed. Device is ready for use!")

except PermissionError:
    print("\n" + "="*60)
    print("PORT STILL LOCKED")
    print("="*60)
    print("\nThe web flasher is still using COM3!")
    print("\nPlease:")
    print("  1. Close the web flasher browser tab completely")
    print("  2. Wait 5 seconds")
    print("  3. Run this script again:")
    print("     python test_connection.py")

except Exception as e:
    print(f"\nError: {e}")
    print("\nIf you just flashed:")
    print("  1. Unplug the device")
    print("  2. Wait 5 seconds")
    print("  3. Plug it back in")
    print("  4. Run this script again")
    import traceback
    traceback.print_exc()
