"""Hive data models — frozen dataclasses, no bus dependency."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class HiveInspection:
    """A single hive inspection record."""
    hive_id: str
    date: str                        # ISO date
    queen_seen: bool = True
    eggs_seen: bool = True
    brood_pattern: str = "solid"     # solid | spotty | absent
    varroa_count: float = 0.0        # per 100 bees (sugar roll / alcohol wash)
    honey_stores: str = "adequate"   # light | adequate | heavy
    temperament: str = "calm"        # calm | nervous | aggressive
    queen_cells: int = 0
    disease_signs: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class Hive:
    """A hive in the apiary."""
    id: str
    name: str
    yard: str = "main"
    inspections: List[HiveInspection] = field(default_factory=list)

    @property
    def latest_inspection(self) -> Optional[HiveInspection]:
        if not self.inspections:
            return None
        return max(self.inspections, key=lambda i: i.date)


def load_hives(path: Path) -> List[Hive]:
    """Load hives from JSON file."""
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    hives = []
    for h in data.get("hives", []):
        inspections = [
            HiveInspection(**insp) for insp in h.get("inspections", [])
        ]
        hives.append(Hive(
            id=h["id"],
            name=h.get("name", h["id"]),
            yard=h.get("yard", "main"),
            inspections=inspections,
        ))
    return hives


def save_hives(hives: List[Hive], path: Path) -> None:
    """Save hives to JSON file (atomic write)."""
    data = {"hives": []}
    for hive in hives:
        h = {"id": hive.id, "name": hive.name, "yard": hive.yard}
        h["inspections"] = [asdict(i) for i in hive.inspections]
        data["hives"].append(h)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


SAMPLE_HIVES = [
    Hive(id="hive-1", name="Sunflower", yard="south", inspections=[
        HiveInspection(hive_id="hive-1", date="2026-03-10",
                       queen_seen=True, eggs_seen=True, brood_pattern="solid",
                       varroa_count=2.0, honey_stores="adequate", queen_cells=0),
    ]),
    Hive(id="hive-2", name="Clover", yard="south", inspections=[
        HiveInspection(hive_id="hive-2", date="2026-03-10",
                       queen_seen=False, eggs_seen=False, brood_pattern="spotty",
                       varroa_count=6.0, honey_stores="light", queen_cells=2,
                       disease_signs=["chalkbrood"]),
    ]),
    Hive(id="hive-3", name="Wildflower", yard="north", inspections=[
        HiveInspection(hive_id="hive-3", date="2026-03-10",
                       queen_seen=True, eggs_seen=True, brood_pattern="solid",
                       varroa_count=1.0, honey_stores="heavy"),
    ]),
]
