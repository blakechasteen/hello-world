"""Minimal motif detection abstractions and simple implementations.

This file provides lightweight, test-friendly motif detectors and a
`create_motif_detector` factory used by the orchestrator. The real
project may use spaCy or regex-based detectors; here we implement
small, dependency-free stubs so the example runs.
"""
from __future__ import annotations
from typing import List, Optional
import re

from holoLoom.documentation.types import Motif


class MotifDetector:
	"""Protocol/base class for motif detectors."""
	async def detect(self, text: str) -> List[Motif]:
		raise NotImplementedError()


class RegexMotifDetector(MotifDetector):
	"""Simple regex-based motif detector.

	It looks for a few basic keywords and returns Motif objects with a
	crude span and score. This is intentionally tiny and deterministic.
	"""
	KEYWORDS = [
		"question", "answer", "algorithm", "thompson", "sampling",
		"hive", "inspection", "calculate", "compute"
	]

	def __init__(self):
		self._patterns = [re.compile(rf"\b{re.escape(k)}\b", re.I) for k in self.KEYWORDS]

	async def detect(self, text: str) -> List[Motif]:
		matches: List[Motif] = []
		for pat in self._patterns:
			for m in pat.finditer(text):
				motif = Motif(pattern=pat.pattern.strip('\\b').strip(), span=(m.start(), m.end()), score=0.9)
				matches.append(motif)
		return matches


class SpacyMotifDetector(MotifDetector):
	"""Stub for a spaCy-backed detector. Behaves like RegexMotifDetector if spaCy not present."""
	def __init__(self):
		self.inner = RegexMotifDetector()

	async def detect(self, text: str) -> List[Motif]:
		return await self.inner.detect(text)


class HybridMotifDetector(MotifDetector):
	"""Combines different detectors; currently delegates to RegexMotifDetector."""
	def __init__(self):
		self.detector = RegexMotifDetector()

	async def detect(self, text: str) -> List[Motif]:
		return await self.detector.detect(text)


def create_motif_detector(mode: Optional[str] = None) -> MotifDetector:
	"""Factory: return a motif detector instance based on `mode`.

	Modes: 'regex' (default), 'spacy', 'hybrid'
	"""
	mode = (mode or "regex").lower()
	if mode == "spacy":
		return SpacyMotifDetector()
	if mode == "hybrid":
		return HybridMotifDetector()
	return RegexMotifDetector()

