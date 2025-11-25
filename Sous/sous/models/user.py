#!/usr/bin/env python3
"""
User and Household Models for Multi-User Kitchen Management

Supports:
- Individual user profiles with skills and preferences
- Household management with shared resources
- Task assignment based on skill levels
- Gamification (points, achievements, levels)
- Schedule coordination across household members
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Optional, Set
from enum import Enum


class SkillLevel(Enum):
    """Cooking skill levels"""
    BEGINNER = 1.0
    INTERMEDIATE = 2.0
    ADVANCED = 3.0
    EXPERT = 4.0


class ResourceType(Enum):
    """Kitchen resource types"""
    OVEN = "oven"
    STOVETOP = "stovetop"
    COUNTER = "counter"
    FRIDGE = "fridge"
    MICROWAVE = "microwave"


@dataclass
class AvailabilityWindow:
    """Time window when user is available for cooking"""
    day_of_week: int  # 0=Monday, 6=Sunday
    start_time: time
    end_time: time
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "day_of_week": self.day_of_week,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AvailabilityWindow':
        return cls(
            day_of_week=data["day_of_week"],
            start_time=time.fromisoformat(data["start_time"]),
            end_time=time.fromisoformat(data["end_time"]),
            notes=data.get("notes", "")
        )


@dataclass
class UserProfile:
    """
    Individual user profile with cooking skills and preferences.

    Tracks:
    - Identity (user_id, name, household_id)
    - Skills (per-technique cooking abilities)
    - Preferences (dietary, cuisines, schedule)
    - Statistics (for gamification)
    """
    user_id: str
    name: str
    household_id: str

    # Cooking skills (technique → proficiency 0.0-4.0)
    skills: Dict[str, float] = field(default_factory=dict)

    # Dietary preferences and restrictions
    dietary_restrictions: List[str] = field(default_factory=list)
    favorite_cuisines: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)

    # Schedule availability
    availability: List[AvailabilityWindow] = field(default_factory=list)
    preferred_meal_times: Dict[str, time] = field(default_factory=dict)  # breakfast, lunch, dinner

    # Gamification stats
    total_points: int = 0
    level: int = 1
    tasks_completed: int = 0
    recipes_mastered: Set[str] = field(default_factory=set)
    achievements: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

    def get_skill_level(self, technique: str) -> float:
        """Get proficiency for a specific cooking technique."""
        return self.skills.get(technique, 1.0)  # Default to BEGINNER

    def update_skill(self, technique: str, delta: float):
        """Update skill level (can increase or decrease)."""
        current = self.skills.get(technique, 1.0)
        new_level = max(1.0, min(4.0, current + delta))  # Clamp to 1.0-4.0
        self.skills[technique] = new_level

    def add_points(self, points: int, reason: str = ""):
        """Add gamification points and check for level up."""
        self.total_points += points

        # Level up every 1000 points
        new_level = 1 + (self.total_points // 1000)
        if new_level > self.level:
            self.level = new_level
            self.achievements.append(f"Reached Level {new_level}")

    def complete_task(self, points: int = 10):
        """Record task completion."""
        self.tasks_completed += 1
        self.add_points(points, "Task completion")
        self.last_active = datetime.now()

    def master_recipe(self, recipe_id: str):
        """Mark a recipe as mastered (cooked 5+ times successfully)."""
        self.recipes_mastered.add(recipe_id)
        self.add_points(50, f"Mastered recipe: {recipe_id}")
        self.achievements.append(f"Mastered: {recipe_id}")

    def is_available_at(self, day: int, hour: int) -> bool:
        """Check if user is available at given day/time."""
        check_time = time(hour=hour, minute=0)
        for window in self.availability:
            if window.day_of_week == day:
                if window.start_time <= check_time <= window.end_time:
                    return True
        return False

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "household_id": self.household_id,
            "skills": self.skills,
            "dietary_restrictions": self.dietary_restrictions,
            "favorite_cuisines": self.favorite_cuisines,
            "allergies": self.allergies,
            "availability": [w.to_dict() for w in self.availability],
            "preferred_meal_times": {k: v.isoformat() for k, v in self.preferred_meal_times.items()},
            "total_points": self.total_points,
            "level": self.level,
            "tasks_completed": self.tasks_completed,
            "recipes_mastered": list(self.recipes_mastered),
            "achievements": self.achievements,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'UserProfile':
        """Deserialize from dictionary."""
        profile = cls(
            user_id=data["user_id"],
            name=data["name"],
            household_id=data["household_id"],
            skills=data.get("skills", {}),
            dietary_restrictions=data.get("dietary_restrictions", []),
            favorite_cuisines=data.get("favorite_cuisines", []),
            allergies=data.get("allergies", []),
            total_points=data.get("total_points", 0),
            level=data.get("level", 1),
            tasks_completed=data.get("tasks_completed", 0),
            recipes_mastered=set(data.get("recipes_mastered", [])),
            achievements=data.get("achievements", [])
        )

        # Parse availability windows
        if "availability" in data:
            profile.availability = [
                AvailabilityWindow.from_dict(w) for w in data["availability"]
            ]

        # Parse preferred meal times
        if "preferred_meal_times" in data:
            profile.preferred_meal_times = {
                k: time.fromisoformat(v) for k, v in data["preferred_meal_times"].items()
            }

        # Parse timestamps
        if "created_at" in data:
            profile.created_at = datetime.fromisoformat(data["created_at"])
        if "last_active" in data:
            profile.last_active = datetime.fromisoformat(data["last_active"])

        return profile


@dataclass
class ResourceSchedule:
    """Schedule for a kitchen resource (oven, stovetop, etc.)"""
    resource_type: ResourceType
    bookings: List[Dict[str, any]] = field(default_factory=list)  # {user_id, start, end, task_id}

    def is_available(self, start: datetime, end: datetime) -> bool:
        """Check if resource is available in time window."""
        for booking in self.bookings:
            if not (end <= booking["start"] or start >= booking["end"]):
                return False  # Overlap detected
        return True

    def book(self, user_id: str, task_id: str, start: datetime, end: datetime) -> bool:
        """Book resource if available."""
        if not self.is_available(start, end):
            return False

        self.bookings.append({
            "user_id": user_id,
            "task_id": task_id,
            "start": start,
            "end": end
        })
        self.bookings.sort(key=lambda b: b["start"])
        return True

    def release(self, task_id: str):
        """Release resource booking for a task."""
        self.bookings = [b for b in self.bookings if b["task_id"] != task_id]


@dataclass
class Household:
    """
    Household profile for shared kitchen management.

    Tracks:
    - Members and roles
    - Shared resources (oven, stovetop, counter)
    - Active recipes and tasks
    - Household-level statistics
    """
    household_id: str
    name: str
    members: List[str] = field(default_factory=list)  # List of user_ids

    # Kitchen resources
    resources: Dict[ResourceType, ResourceSchedule] = field(default_factory=dict)

    # Active cooking
    active_recipes: List[str] = field(default_factory=list)
    pending_tasks: List[str] = field(default_factory=list)

    # Household statistics
    total_meals_cooked: int = 0
    waste_prevented_kg: float = 0.0
    food_shared_kg: float = 0.0
    composted_kg: float = 0.0
    carbon_saved_kg: float = 0.0

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    address: str = ""

    def add_member(self, user_id: str):
        """Add a member to the household."""
        if user_id not in self.members:
            self.members.append(user_id)

    def remove_member(self, user_id: str):
        """Remove a member from the household."""
        if user_id in self.members:
            self.members.remove(user_id)

    def initialize_resources(self, oven_count: int = 1, stovetop_count: int = 1):
        """Initialize kitchen resources."""
        for _ in range(oven_count):
            key = ResourceType.OVEN
            if key not in self.resources:
                self.resources[key] = ResourceSchedule(resource_type=key)

        for _ in range(stovetop_count):
            key = ResourceType.STOVETOP
            if key not in self.resources:
                self.resources[key] = ResourceSchedule(resource_type=key)

        # Counter space (unlimited for now)
        self.resources[ResourceType.COUNTER] = ResourceSchedule(resource_type=ResourceType.COUNTER)

    def book_resource(
        self,
        resource_type: ResourceType,
        user_id: str,
        task_id: str,
        start: datetime,
        end: datetime
    ) -> bool:
        """Book a kitchen resource."""
        if resource_type not in self.resources:
            return False
        return self.resources[resource_type].book(user_id, task_id, start, end)

    def release_resource(self, resource_type: ResourceType, task_id: str):
        """Release a kitchen resource."""
        if resource_type in self.resources:
            self.resources[resource_type].release(task_id)

    def record_meal(self):
        """Record a completed meal."""
        self.total_meals_cooked += 1

    def record_waste_prevented(self, kg: float):
        """Record waste prevention (food saved from being thrown away)."""
        self.waste_prevented_kg += kg
        self.carbon_saved_kg += kg * 2.5  # Rough estimate: 2.5 kg CO2e per kg food waste

    def record_food_shared(self, kg: float):
        """Record food shared with neighbors."""
        self.food_shared_kg += kg

    def record_composted(self, kg: float):
        """Record unavoidable waste composted."""
        self.composted_kg += kg

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "household_id": self.household_id,
            "name": self.name,
            "members": self.members,
            "active_recipes": self.active_recipes,
            "pending_tasks": self.pending_tasks,
            "total_meals_cooked": self.total_meals_cooked,
            "waste_prevented_kg": self.waste_prevented_kg,
            "food_shared_kg": self.food_shared_kg,
            "composted_kg": self.composted_kg,
            "carbon_saved_kg": self.carbon_saved_kg,
            "created_at": self.created_at.isoformat(),
            "address": self.address
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Household':
        """Deserialize from dictionary."""
        household = cls(
            household_id=data["household_id"],
            name=data["name"],
            members=data.get("members", []),
            active_recipes=data.get("active_recipes", []),
            pending_tasks=data.get("pending_tasks", []),
            total_meals_cooked=data.get("total_meals_cooked", 0),
            waste_prevented_kg=data.get("waste_prevented_kg", 0.0),
            food_shared_kg=data.get("food_shared_kg", 0.0),
            composted_kg=data.get("composted_kg", 0.0),
            carbon_saved_kg=data.get("carbon_saved_kg", 0.0),
            address=data.get("address", "")
        )

        if "created_at" in data:
            household.created_at = datetime.fromisoformat(data["created_at"])

        # Initialize resources
        household.initialize_resources()

        return household
