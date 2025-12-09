# -*- coding: utf-8 -*-
"""Connect to Meshtastic device with extended timeout"""
import sys
import time

try:
    import meshtastic
    import meshtastic.serial_interface

    print("="*60)
    print("CONNECTING TO HELTEC DEVICE ON COM3")
    print("="*60)
    print("\nThis may take 20-30 seconds on first connection...")
    print("(Device firmware may be initializing)\n")

    # Try connecting with explicit port
    try:
        print("Connecting to COM3...")
        interface = meshtastic.serial_interface.SerialInterface(
            devPath="COM3",
            debugOut=sys.stdout  # Show debug info
        )

        print("\n" + "="*60)
        print("SUCCESS! DEVICE CONNECTED!")
        print("="*60)

        # Give it time to fully initialize
        print("\nWaiting for device info...")
        time.sleep(5)

        # Get info
        if hasattr(interface, 'myInfo') and interface.myInfo:
            print(f"\nNode ID: {interface.myInfo.my_node_num}")
            if hasattr(interface.myInfo, 'user'):
                print(f"Long Name: {interface.myInfo.user.long_name}")
                print(f"Short Name: {interface.myInfo.user.short_name}")

        print("\n" + "="*60)
        print("YOUR DEVICE IS ONLINE!")
        print("="*60)

        print("\nKeeping connection open for 10 seconds...")
        print("(You can check the device screen)")
        time.sleep(10)

        interface.close()
        print("\nConnection closed successfully.")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        if 'interface' in locals():
            interface.close()

    except Exception as e:
        print(f"\n{'='*60}")
        print("CONNECTION FAILED")
        print("="*60)
        print(f"\nError: {e}")
        print("\nPossible issues:")
        print("  1. Device firmware may need to be flashed/updated")
        print("  2. Device may be in bootloader mode")
        print("  3. Another program may be using COM3")
        print("  4. Device may need a hard reset")
        print("\nTry:")
        print("  - Hold the reset button on device for 3 seconds")
        print("  - Unplug USB, wait 5 seconds, replug")
        print("  - Check if device screen shows anything")
        print("  - Visit: https://flasher.meshtastic.org to flash firmware")

except ImportError:
    print("ERROR: meshtastic module not installed")
    print("Run: pip install meshtastic")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
