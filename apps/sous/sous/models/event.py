"""Event and Appliance models"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional


@dataclass
class Appliance:
    """
    Kitchen appliance with capacity constraints

    Attributes:
        id: Unique identifier (e.g., "oven_main")
        name: Display name (e.g., "Main Oven")
        type: Appliance type (oven, burner, microwave)
        capacity: Number of simultaneous tasks (usually 1)
        notes: Additional notes (e.g., "only one good burner")
    """

    id: str
    name: str
    type: str  # oven, burner, microwave, etc.
    capacity: int = 1
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Appliance":
        """Create Appliance from JSON dict"""
        return cls(
            id=data["id"],
            name=data["name"],
            type=data["type"],
            capacity=data.get("capacity", 1),
            notes=data.get("notes", ""),
        )

    def to_dict(self) -> dict:
        """Convert to JSON dict"""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "capacity": self.capacity,
            "notes": self.notes,
        }


@dataclass
class Event:
    """
    Cooking event (e.g., Thanksgiving dinner)

    Attributes:
        id: Unique identifier
        name: Event name
        date: Event date
        prep_days: List of dates available for prep
        appliances: Available kitchen appliances
    """

    id: str
    name: str
    date: date
    prep_days: List[date] = field(default_factory=list)
    appliances: List[Appliance] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        """Create Event from JSON dict"""
        # Parse dates
        event_date = date.fromisoformat(data["date"])
        prep_days = [date.fromisoformat(d) for d in data.get("prep_days", [])]

        # Parse appliances
        appliances = [
            Appliance.from_dict(a) for a in data.get("appliances", [])
        ]

        return cls(
            id=data["id"],
            name=data["name"],
            date=event_date,
            prep_days=prep_days,
            appliances=appliances,
        )

    def to_dict(self) -> dict:
        """Convert to JSON dict"""
        return {
            "id": self.id,
            "name": self.name,
            "date": self.date.isoformat(),
            "prep_days": [d.isoformat() for d in self.prep_days],
            "appliances": [a.to_dict() for a in self.appliances],
        }

    def get_appliance(self, appliance_id: str) -> Optional[Appliance]:
        """Get appliance by ID"""
        for appliance in self.appliances:
            if appliance.id == appliance_id:
                return appliance
        return None

    def get_appliances_by_type(self, appliance_type: str) -> List[Appliance]:
        """Get all appliances of given type"""
        return [a for a in self.appliances if a.type == appliance_type]

    def days_before_event(self, day: date) -> int:
        """Calculate days before event"""
        return (self.date - day).days

    def is_prep_day(self, day: date) -> bool:
        """Check if date is a prep day"""
        return day in self.prep_days

    def is_event_day(self, day: date) -> bool:
        """Check if date is the event day"""
        return day == self.date
