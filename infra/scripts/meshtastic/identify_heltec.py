# -*- coding: utf-8 -*-
"""Identify Heltec LoRa device model and chip type"""
import sys

try:
    import serial.tools.list_ports

    print("="*60)
    print("HELTEC DEVICE IDENTIFIER")
    print("="*60)

    # Get detailed port info
    ports = list(serial.tools.list_ports.comports())

    for port in ports:
        if 'COM3' in port.device:
            print(f"\nDevice on COM3:")
            print(f"  Description: {port.description}")
            print(f"  Manufacturer: {port.manufacturer}")
            print(f"  VID:PID: {hex(port.vid) if port.vid else 'N/A'}:{hex(port.pid) if port.pid else 'N/A'}")
            print(f"  Serial Number: {port.serial_number if port.serial_number else 'N/A'}")

            # Check USB chip (gives clues about ESP32 model)
            if port.vid and port.pid:
                vid = port.vid
                pid = port.pid

                print(f"\nUSB Chip Analysis:")
                # Silicon Labs CP210x is common on Heltec devices
                if vid == 0x10C4:  # Silicon Labs
                    print("  ✓ Silicon Labs CP210x detected")
                    print("  Common on: HELTEC V3, Wireless Stick, etc.")

                # CH340 chip
                elif vid == 0x1A86:  # QinHeng Electronics
                    print("  ✓ CH340 USB chip detected")
                    print("  Common on: Various ESP32 boards")

    print("\n" + "="*60)
    print("IDENTIFY YOUR DEVICE")
    print("="*60)

    print("\nCheck the physical device for labels/markings:")
    print("\n1. LOOK AT THE PCB:")
    print("   - Look for text printed on the board")
    print("   - Common Heltec LoRa models:")
    print("     • WiFi LoRa 32 V3 (ESP32-S3)")
    print("     • Wireless Stick Lite V3 (ESP32-S3)")
    print("     • Wireless Stick V3 (ESP32-S3)")
    print("     • WiFi LoRa 32 V2 (ESP32, older)")

    print("\n2. CHECK THE SCREEN:")
    print("   - V3 models: Usually have OLED display")
    print("   - 'Lite' models: May have no screen")

    print("\n3. CHIP MARKINGS:")
    print("   ESP32-S3 chips are marked:")
    print("   - Look for largest black chip on board")
    print("   - Should say 'ESP32-S3' on it")
    print("   - Older V2 boards have 'ESP32-WROOM' or 'ESP32-WROVER'")

    print("\n4. ANTENNA:")
    print("   - LoRa antenna connector (small gold SMA/U.FL)")
    print("   - WiFi/BT antenna (built-in or external)")

    print("\n" + "="*60)
    print("COMMON HELTEC MODELS FOR MESHTASTIC")
    print("="*60)

    models = {
        "WiFi LoRa 32 V3": {
            "chip": "ESP32-S3",
            "screen": "Yes (0.96\" OLED)",
            "buttons": "2 (PRG + RST)",
            "firmware": "HELTEC_V3"
        },
        "Wireless Stick V3": {
            "chip": "ESP32-S3",
            "screen": "Yes (0.96\" OLED)",
            "buttons": "2 (PRG + RST)",
            "firmware": "HELTEC_WIRELESS_STICK_V3"
        },
        "Wireless Stick Lite V3": {
            "chip": "ESP32-S3",
            "screen": "No",
            "buttons": "2 (PRG + RST)",
            "firmware": "HELTEC_WIRELESS_STICK_LITE_V3"
        },
        "Wireless Tracker V1.0": {
            "chip": "ESP32-S3",
            "screen": "Yes (1.3\" OLED)",
            "buttons": "2 (USER + RST)",
            "firmware": "HELTEC_WIRELESS_TRACKER_V1_0"
        },
        "WiFi LoRa 32 V2 (older)": {
            "chip": "ESP32 (NOT S3)",
            "screen": "Yes (0.96\" OLED)",
            "buttons": "2 (PRG + RST)",
            "firmware": "HELTEC_V2"
        }
    }

    for name, specs in models.items():
        print(f"\n{name}:")
        print(f"  Chip: {specs['chip']}")
        print(f"  Screen: {specs['screen']}")
        print(f"  Buttons: {specs['buttons']}")
        print(f"  Firmware ID: {specs['firmware']}")

    print("\n" + "="*60)
    print("QUICK IDENTIFICATION GUIDE")
    print("="*60)

    print("\nIf you have:")
    print("  • Screen + 2 buttons + looks newish → Probably V3 (ESP32-S3)")
    print("  • No screen + 2 buttons → Probably Wireless Stick Lite V3 (ESP32-S3)")
    print("  • Large screen (1.3\") + GPS → Wireless Tracker (ESP32-S3)")
    print("  • Older device (2-3+ years) → Might be V2 (ESP32, not S3)")

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)

    print("\n1. Identify your model from the list above")
    print("2. Go to: https://flasher.meshtastic.org")
    print("3. Select the matching firmware ID")
    print("4. Flash to COM3")

    print("\nNOTE: If unsure between V2 and V3:")
    print("  V3 devices are ESP32-S3 (newer, 2022+)")
    print("  V2 devices are ESP32 classic (older, pre-2022)")
    print("  When in doubt, try V3 firmware first!")

    print("\nNeed more help?")
    print("  • Check purchase date (V3 released mid-2022)")
    print("  • Look at product photos on Heltec website")
    print("  • Check your order receipt/product page")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
