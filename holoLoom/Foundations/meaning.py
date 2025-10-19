from dataclasses import dataclass
from typing import Literal, Dict

Role = Literal["is_a", "part_of", "causes", "synonym", "unit_of"]

@dataclass(frozen=True)
class Concept:
    id: str
    label: str
    attrs: Dict[str, str] | None = None

@dataclass(frozen=True)
class Relation:
    head: str
    role: Role
    tail: str
    weight: float = 1.0

@dataclass(frozen=True)
class Typing:
    concept_id: str
    type_label: str
