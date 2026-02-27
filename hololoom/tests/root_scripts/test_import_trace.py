#!/usr/bin/env python3
"""Trace imports to find the hanging module."""

import sys

print("Step 1: Import dataclasses")
from dataclasses import dataclass
print("OK dataclasses")

print("\nStep 2: Import typing")
from typing import List, Dict
print("OK typing")

print("\nStep 3: Import numpy")
import numpy as np
print("OK numpy")

print("\nStep 4: Import networkx")
import networkx as nx
print("OK networkx")

print("\nStep 5: Import HoloLoom.documentation.types")
sys.stdout.flush()
from hololoom.documentation.types import Vector
print("OK documentation.types")

print("\nAll imports successful!")
