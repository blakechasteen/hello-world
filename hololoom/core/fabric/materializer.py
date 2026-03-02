"""
Materializer - Turning Spacetime into Matter
============================================

The Materializer bridges the gap between the internal `Spacetime` reasoning 
state and the external file system. It takes `Artifact` objects and writes 
them to disk, handling file operations, permissions, and directory creation.

Security Level: FRONTIER
- Risk Engine (Entropy, SAST, Signatures)
- Quarantine Protocol
- Immutable Provenance
- Audit Logging

Component of: HoloLoom "Maker" Capability
"""

import os
import logging
import base64
import json
import math
import ast
import hashlib
import unicodedata
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field

from hololoom.fabric.spacetime import Spacetime, Artifact, ArtifactType

logger = logging.getLogger(__name__)

# --- Frontier Security Constants ---

BLOCKED_EXTENSIONS = {
    '.exe', '.dll', '.bin', '.sh', '.bat', '.cmd', '.ps1', '.vbs', '.js', 
    '.jar', '.msi', '.com', '.scr', '.pif'
}

# Magic Numbers (Start of file bytes)
MAGIC_NUMBERS = {
    b'MZ': 'Windows Executable',
    b'\x7fELF': 'Linux Executable',
    b'\xca\xfe\xba\xbe': 'Java Class',
    b'\xfe\xed\xfa\xce': 'Mach-O Binary',
    b'\xfe\xed\xfa\xcf': 'Mach-O Binary',
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

SENSITIVE_PATTERNS = [
    re.compile(r'AKIA[0-9A-Z]{16}'),  # AWS Key
    re.compile(r'-----BEGIN PRIVATE KEY-----'),  # Private Key
    re.compile(r'AIza[0-9A-Za-z-_]{35}'),  # Google API Key
]

class RiskLevel(Enum):
    SAFE = "SAFE"             # 0-20
    WARN = "WARN"             # 21-50
    QUARANTINE = "QUARANTINE" # 51-80
    BLOCK = "BLOCK"           # 81-100

@dataclass
class RiskAssessment:
    level: RiskLevel
    score: int
    signals: List[str]

# --- Risk Engine ---

class RiskEngine:
    """
    intelligent security gateway that assesses the intent and risk of artifacts.
    """
    
    @staticmethod
    def assess(artifact: Artifact, content_bytes: bytes) -> RiskAssessment:
        score = 0
        signals = []
        
        # 1. Extension Risk
        ext = Path(artifact.name).suffix.lower()
        if ext in BLOCKED_EXTENSIONS:
            score += 100
            signals.append(f"Blocked Extension: {ext}")
        elif ext == '.py':
            score += 10
            signals.append("Executable Script (.py)")
            
        # 2. Magic Number Risk
        for magic, name in MAGIC_NUMBERS.items():
            if content_bytes.startswith(magic):
                score += 80
                signals.append(f"Magic Number Detected: {name}")
                break
                
        # 3. Size Risk
        if len(content_bytes) > MAX_FILE_SIZE:
            score += 100
            signals.append("Exceeds Max File Size")

        # 4. Entropy Risk (Any content)
        entropy = RiskEngine._calculate_entropy(content_bytes)
        if entropy > 7.5:
            score += 60 # Increased from 50 to ensure quarantine
            signals.append(f"High Entropy ({entropy:.2f}): Potential Malware/Packed")
            
        # 5. Content Risk (Text only)
        try:
            text_content = content_bytes.decode('utf-8')
            
            # Secret Scanning
            for pattern in SENSITIVE_PATTERNS:
                if pattern.search(text_content):
                    score += 60
                    signals.append("Sensitive Pattern Detected")
                    break
            
            # SAST - Keyword Risk
            if 'eval(' in text_content or 'exec(' in text_content:
                score += 20
                signals.append("Dangerous Function: eval/exec")
                
            if 'subprocess' in text_content and 'socket' in text_content:
                score += 60 # Increased from 40 to ensure quarantine (10+60=70 > 50)
                signals.append("Behavioral: socket+subprocess (Reverse Shell Signature)")
                
        except UnicodeDecodeError:
            # Binary content that isn't magic-blocked
            if artifact.type == ArtifactType.CODE:
                # Code should be text
                score += 30
                signals.append("Binary content in Code artifact")
                
        # Determine Level
        if score <= 20: level = RiskLevel.SAFE
        elif score <= 50: level = RiskLevel.WARN
        elif score <= 80: level = RiskLevel.QUARANTINE
        else: level = RiskLevel.BLOCK
        
        return RiskAssessment(level, score, signals)

    @staticmethod
    def _calculate_entropy(data: bytes) -> float:
        if not data: return 0
        entropy = 0
        for x in range(256):
            p_x = float(data.count(x)) / len(data)
            if p_x > 0:
                entropy += - p_x * math.log(p_x, 2)
        return entropy

# --- Audit & Provenance ---

class ProvenanceManager:
    @staticmethod
    def generate_provenance(artifact: Artifact, assessment: RiskAssessment, target_path: Path) -> Dict[str, Any]:
        """Generate provenance metadata."""
        return {
            "schema_version": "1.0",
            "artifact_name": artifact.name,
            "timestamp": datetime.now().isoformat(),
            "tool_used": "unknown", # Could be passed via context
            "content_hash": hashlib.sha256(artifact.content.encode() if isinstance(artifact.content, str) else artifact.content).hexdigest(),
            "risk_assessment": {
                "level": assessment.level.value,
                "score": assessment.score,
                "signals": assessment.signals
            },
            "target_path": str(target_path)
        }

class AuditLogger:
    def __init__(self, root_dir: Path):
        self.log_path = root_dir / ".holo_audit.jsonl"
        
    def log(self, action: str, artifact: str, result: str, details: str):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "artifact": artifact,
            "result": result,
            "details": details
        }
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + "\n")

# --- Materializer ---

class Materializer:
    """
    Handles the physical realization of artifacts with Frontier Safety.
    """
    
    def __init__(self, root_dir: Union[str, Path]):
        self.root_dir = Path(root_dir)
        self.logger = logging.getLogger(__name__)
        self.audit = AuditLogger(self.root_dir)
        
    def materialize(self, fabric: Spacetime, dry_run: bool = False, force_overwrite: bool = False) -> Dict[str, List[str]]:
        results = {t.value: [] for t in ArtifactType}
        
        if not fabric.artifacts:
            self.logger.info("No artifacts to materialize.")
            return results
            
        self.logger.info(f"Materializing {len(fabric.artifacts)} artifacts... (Frontier Safety Active)")
        
        for artifact in fabric.artifacts:
            try:
                path = self._process_artifact(artifact, dry_run, force_overwrite)
                if path:
                    results[artifact.type.value].append(str(path))
            except Exception as e:
                self.logger.error(f"Failed to materialize '{artifact.name}': {e}")
                self.audit.log("materialize", artifact.name, "error", str(e))
                
        return results
        
    def _process_artifact(self, artifact: Artifact, dry_run: bool, force_overwrite: bool) -> Optional[Path]:
        if not artifact.content:
            return None
            
        # 0. Prepare Content Bytes
        content_bytes = artifact.content
        if isinstance(content_bytes, str):
            content_bytes = content_bytes.encode('utf-8')
            
        # 1. RISK ASSESSMENT
        assessment = RiskEngine.assess(artifact, content_bytes)
        self.logger.info(f"Risk Assessment for {artifact.name}: {assessment.level.value} (Score: {assessment.score})")
        
        if assessment.level == RiskLevel.BLOCK:
            self.logger.error(f"FRONTIER BLOCK: {artifact.name} (Risk: {assessment.score}) - {assessment.signals}")
            self.audit.log("write", artifact.name, "blocked", str(assessment.signals))
            return None
            
        # 2. Path Resolution
        # Unicode Normalization (NFKC) to prevent homograph attacks
        clean_name = unicodedata.normalize('NFKC', artifact.name)
        clean_dest = unicodedata.normalize('NFKC', artifact.destination_path) if artifact.destination_path else clean_name
        
        rel_path = Path(clean_dest)
        if rel_path.is_absolute() or '..' in rel_path.parts:
             self.logger.error(f"Security Block: Path traversal detected '{clean_dest}'.")
             self.audit.log("write", artifact.name, "blocked", "Path traversal")
             return None

        # Resolve
        final_path = (self.root_dir / rel_path).resolve()
        
        # Determine effective write path (Quarantine or Target)
        write_path = final_path
        quarantined = False
        
        if assessment.level == RiskLevel.QUARANTINE:
            quarantine_dir = self.root_dir / ".quarantine"
            quarantine_dir.mkdir(exist_ok=True)
            write_path = quarantine_dir / f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{clean_name}"
            quarantined = True
            self.logger.warning(f"QUARANTINE: Redirecting {artifact.name} to {write_path}")

        # Check symlinks (don't follow)
        if write_path.is_symlink():
             self.logger.error("Security Block: Target is a symlink.")
             self.audit.log("write", artifact.name, "blocked", "Symlink target")
             return None

        # 3. Dry Run Check
        if dry_run:
            self.logger.info(f"[DRY-RUN] would write to {write_path}")
            return write_path

        # 4. Overwrite Protection & Backups (Only if not quarantined)
        if not quarantined and write_path.exists():
            if not force_overwrite:
                self.logger.info(f"Skipping existing file {write_path}")
                return write_path
            else:
                # Create Backup
                backup_path = write_path.with_suffix(write_path.suffix + ".bak")
                import shutil
                shutil.copy2(write_path, backup_path)
                self.logger.info(f"Backup created: {backup_path}")

        # 5. Write Execution
        try:
            write_path.parent.mkdir(parents=True, exist_ok=True)
            mode = 'wb'
            with open(write_path, mode) as f:
                f.write(content_bytes)
                
            # 6. Provenance Sidecar (If not quarantined)
            if not quarantined:
                prov = ProvenanceManager.generate_provenance(artifact, assessment, write_path)
                prov_path = write_path.with_suffix(write_path.suffix + ".provenance")
                with open(prov_path, 'w') as f:
                    json.dump(prov, f, indent=2)
                    
            # 7. Warning File (If Check was WARN)
            if assessment.level == RiskLevel.WARN and not quarantined:
                warn_path = write_path.with_suffix(write_path.suffix + ".WARNING.txt")
                with open(warn_path, 'w') as f:
                    f.write(f"WARNING: This file has high risk score ({assessment.score}).\nSignals: {assessment.signals}")

            action_res = "quarantined" if quarantined else "written"
            self.audit.log("write", artifact.name, action_res, f"Score: {assessment.score}")
            
            return write_path

        except Exception as e:
            self.logger.error(f"Write failed: {e}")
            self.audit.log("write", artifact.name, "failed", str(e))
            return None
