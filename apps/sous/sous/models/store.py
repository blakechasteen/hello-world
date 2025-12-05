"""Store model"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Store:
    """
    Grocery store or local farm with routing metadata

    Attributes:
        id: Unique identifier (e.g., "store_amish", "farm_smiths")
        name: Display name (e.g., "Amish Market", "Smith Family Farm")
        roles: Store categories (anchor, fill, specialty, bulk, produce, local_farm, farmers_market)
        priority: Lower number = preferred (1 = highest priority)
        is_local_farm: Whether this is a local farm/farmers market
        seasonal_availability: Months when farm is open (1-12), empty list = year-round
        specialties: What this farm is known for (e.g., ["tomatoes", "eggs", "honey"])
        notes: Additional info (e.g., "Open Saturdays 8am-noon", "Cash only")
    """

    id: str
    name: str
    roles: List[str] = field(default_factory=list)
    priority: int = 999
    is_local_farm: bool = False
    seasonal_availability: List[int] = field(default_factory=list)
    specialties: List[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Store":
        """Create Store from JSON dict"""
        return cls(
            id=data["id"],
            name=data["name"],
            roles=data.get("roles", []),
            priority=data.get("priority", 999),
            is_local_farm=data.get("is_local_farm", False),
            seasonal_availability=data.get("seasonal_availability", []),
            specialties=data.get("specialties", []),
            notes=data.get("notes", ""),
        )

    def to_dict(self) -> dict:
        """Convert to JSON dict"""
        return {
            "id": self.id,
            "name": self.name,
            "roles": self.roles,
            "priority": self.priority,
            "is_local_farm": self.is_local_farm,
            "seasonal_availability": self.seasonal_availability,
            "specialties": self.specialties,
            "notes": self.notes,
        }

    def has_role(self, role: str) -> bool:
        """Check if store has given role"""
        return role in self.roles

    def __lt__(self, other: "Store") -> bool:
        """Enable sorting by priority (lower is better)"""
        return self.priority < other.priority

    def __repr__(self) -> str:
        return f"Store({self.name}, priority={self.priority}, roles={self.roles})"
