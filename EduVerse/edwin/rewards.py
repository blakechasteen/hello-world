"""
EdWIN Reward System

Virtual rewards for student motivation (avatars, themes, power-ups, titles).

**Implementation Date**: November 15, 2025
**Agent**: Agent C
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum


class RewardType(Enum):
    """Types of rewards"""
    AVATAR = "avatar"  # Profile avatars
    THEME = "theme"  # Dashboard themes
    TITLE = "title"  # Display titles
    POWER_UP = "power_up"  # XP boosters, streak freezes
    BADGE = "badge"  # Special badges
    CONTENT = "content"  # Exclusive learning content


class RewardRarity(Enum):
    """Reward rarity (affects cost)"""
    COMMON = "common"  # 100-500 XP
    UNCOMMON = "uncommon"  # 500-1000 XP
    RARE = "rare"  # 1000-2500 XP
    EPIC = "epic"  # 2500-5000 XP
    LEGENDARY = "legendary"  # 5000+ XP


@dataclass
class Reward:
    """A single unlockable reward"""
    id: str
    name: str
    description: str
    reward_type: RewardType
    rarity: RewardRarity
    xp_cost: int
    icon: str
    unlocked_by_default: bool = False
    unlock_requirement: Optional[str] = None  # "level_10", "achievement_xyz", etc.
    metadata: Dict = field(default_factory=dict)


# Reward catalog
REWARD_CATALOG = {
    # Avatars (20)
    "avatar_robot": Reward("avatar_robot", "Robot Scholar", "Futuristic robot avatar", RewardType.AVATAR, RewardRarity.COMMON, 100, "🤖"),
    "avatar_wizard": Reward("avatar_wizard", "Knowledge Wizard", "Magical wizard avatar", RewardType.AVATAR, RewardRarity.UNCOMMON, 500, "🧙"),
    "avatar_ninja": Reward("avatar_ninja", "Learning Ninja", "Stealthy ninja avatar", RewardType.AVATAR, RewardRarity.RARE, 1000, "🥷"),
    "avatar_astronaut": Reward("avatar_astronaut", "Space Explorer", "Astronaut avatar", RewardType.AVATAR, RewardRarity.EPIC, 2500, "👨‍🚀"),
    "avatar_superhero": Reward("avatar_superhero", "Super Learner", "Superhero avatar", RewardType.AVATAR, RewardRarity.LEGENDARY, 5000, "🦸"),

    # Themes (15)
    "theme_dark": Reward("theme_dark", "Dark Mode", "Sleek dark theme", RewardType.THEME, RewardRarity.COMMON, 200, "🌙"),
    "theme_ocean": Reward("theme_ocean", "Ocean Breeze", "Calming ocean theme", RewardType.THEME, RewardRarity.UNCOMMON, 600, "🌊"),
    "theme_forest": Reward("theme_forest", "Forest Green", "Nature-inspired theme", RewardType.THEME, RewardRarity.UNCOMMON, 600, "🌲"),
    "theme_sunset": Reward("theme_sunset", "Sunset Glow", "Warm sunset colors", RewardType.THEME, RewardRarity.RARE, 1200, "🌅"),
    "theme_galaxy": Reward("theme_galaxy", "Galaxy", "Cosmic galaxy theme", RewardType.THEME, RewardRarity.EPIC, 3000, "🌌"),

    # Titles (25)
    "title_scholar": Reward("title_scholar", "Scholar", "Display 'Scholar' title", RewardType.TITLE, RewardRarity.COMMON, 150, "📚", unlocked_by_default=True),
    "title_genius": Reward("title_genius", "Genius", "Display 'Genius' title", RewardType.TITLE, RewardRarity.UNCOMMON, 700, "🧠"),
    "title_master": Reward("title_master", "Master", "Display 'Master' title", RewardType.TITLE, RewardRarity.RARE, 1500, "🎓"),
    "title_legend": Reward("title_legend", "Legend", "Display 'Legend' title", RewardType.TITLE, RewardRarity.EPIC, 4000, "👑"),
    "title_math_whiz": Reward("title_math_whiz", "Math Whiz", "Math expert title", RewardType.TITLE, RewardRarity.UNCOMMON, 800, "🔢", unlock_requirement="achievement_math_master"),

    # Power-ups (20)
    "powerup_xp_2x_1h": Reward("powerup_xp_2x_1h", "2x XP Booster (1h)", "Double XP for 1 hour", RewardType.POWER_UP, RewardRarity.UNCOMMON, 500, "⚡"),
    "powerup_xp_3x_1h": Reward("powerup_xp_3x_1h", "3x XP Booster (1h)", "Triple XP for 1 hour", RewardType.POWER_UP, RewardRarity.RARE, 1500, "⚡⚡"),
    "powerup_streak_freeze": Reward("powerup_streak_freeze", "Streak Freeze", "Protect your streak for 1 day", RewardType.POWER_UP, RewardRarity.COMMON, 300, "❄️"),
    "powerup_second_chance": Reward("powerup_second_chance", "Second Chance", "Retry a failed objective without penalty", RewardType.POWER_UP, RewardRarity.UNCOMMON, 600, "🔄"),

    # Exclusive Content (10)
    "content_advanced_math": Reward("content_advanced_math", "Advanced Math Pack", "Unlock advanced math problems", RewardType.CONTENT, RewardRarity.RARE, 2000, "📐", unlock_requirement="level_20"),
    "content_ai_tutorials": Reward("content_ai_tutorials", "AI Tutorial Series", "Exclusive AI learning content", RewardType.CONTENT, RewardRarity.EPIC, 3500, "🤖", unlock_requirement="level_30"),

    # Special Badges (10)
    "badge_golden_star": Reward("badge_golden_star", "Golden Star", "Exclusive golden star badge", RewardType.BADGE, RewardRarity.LEGENDARY, 10000, "⭐", unlock_requirement="level_100"),
}


@dataclass
class StudentInventory:
    """Student's owned rewards"""
    student_id: str
    owned_rewards: List[str] = field(default_factory=list)
    active_avatar: Optional[str] = None
    active_theme: Optional[str] = None
    active_title: Optional[str] = None
    active_power_ups: Dict[str, datetime] = field(default_factory=dict)  # power_up_id -> expiration


class RewardStore:
    """
    Reward store for purchasing and managing rewards

    Example:
        >>> store = RewardStore()
        >>> inventory = StudentInventory(student_id="student_001")
        >>>
        >>> # Purchase reward
        >>> result = store.purchase_reward(
        ...     reward_id="avatar_robot",
        ...     student_inventory=inventory,
        ...     student_xp=150
        ... )
        >>>
        >>> # Equip avatar
        >>> store.equip_reward("avatar_robot", inventory)
    """

    def __init__(self):
        self.catalog = REWARD_CATALOG

    def get_available_rewards(
        self,
        student_level: int,
        student_achievements: List[str]
    ) -> List[Reward]:
        """Get rewards available for purchase"""
        available = []

        for reward in self.catalog.values():
            # Check unlock requirements
            if reward.unlock_requirement:
                if reward.unlock_requirement.startswith("level_"):
                    required_level = int(reward.unlock_requirement.split("_")[1])
                    if student_level < required_level:
                        continue
                elif reward.unlock_requirement.startswith("achievement_"):
                    if reward.unlock_requirement not in student_achievements:
                        continue

            available.append(reward)

        return available

    def purchase_reward(
        self,
        reward_id: str,
        student_inventory: StudentInventory,
        student_xp: int
    ) -> Dict:
        """Purchase a reward"""
        if reward_id not in self.catalog:
            return {"success": False, "error": "Reward not found"}

        reward = self.catalog[reward_id]

        # Already owned?
        if reward_id in student_inventory.owned_rewards:
            return {"success": False, "error": "Already owned"}

        # Can afford?
        if student_xp < reward.xp_cost:
            return {
                "success": False,
                "error": f"Insufficient XP (need {reward.xp_cost}, have {student_xp})"
            }

        # Purchase!
        student_inventory.owned_rewards.append(reward_id)

        return {
            "success": True,
            "reward_id": reward_id,
            "reward_name": reward.name,
            "xp_spent": reward.xp_cost,
            "xp_remaining": student_xp - reward.xp_cost
        }

    def equip_reward(
        self,
        reward_id: str,
        student_inventory: StudentInventory
    ) -> bool:
        """Equip a reward (avatar, theme, or title)"""
        if reward_id not in student_inventory.owned_rewards:
            return False

        reward = self.catalog.get(reward_id)
        if not reward:
            return False

        # Equip based on type
        if reward.reward_type == RewardType.AVATAR:
            student_inventory.active_avatar = reward_id
        elif reward.reward_type == RewardType.THEME:
            student_inventory.active_theme = reward_id
        elif reward.reward_type == RewardType.TITLE:
            student_inventory.active_title = reward_id

        return True

    def activate_power_up(
        self,
        power_up_id: str,
        student_inventory: StudentInventory
    ) -> Optional[datetime]:
        """Activate a power-up (returns expiration time)"""
        if power_up_id not in student_inventory.owned_rewards:
            return None

        reward = self.catalog.get(power_up_id)
        if not reward or reward.reward_type != RewardType.POWER_UP:
            return None

        # Determine duration (from metadata or default 1 hour)
        duration_hours = reward.metadata.get("duration_hours", 1)
        expiration = datetime.utcnow() + timedelta(hours=duration_hours)

        student_inventory.active_power_ups[power_up_id] = expiration

        # Remove from owned (consumable)
        student_inventory.owned_rewards.remove(power_up_id)

        return expiration


# Helper functions

def create_reward_store() -> RewardStore:
    """Create reward store"""
    return RewardStore()


if __name__ == "__main__":
    print("=== EdWIN Reward Store Demo ===\n")

    store = RewardStore()
    inventory = StudentInventory(student_id="student_001")

    # Show available rewards
    print("Available Rewards (Level 1):")
    available = store.get_available_rewards(student_level=1, student_achievements=[])
    for reward in available[:5]:
        print(f"  {reward.icon} {reward.name} - {reward.xp_cost} XP ({reward.rarity.value})")

    # Purchase avatar
    print("\nPurchasing Robot Avatar...")
    result = store.purchase_reward("avatar_robot", inventory, student_xp=150)
    if result["success"]:
        print(f"  ✅ Purchased! {result['xp_remaining']} XP remaining")

    # Equip avatar
    print("\nEquipping Robot Avatar...")
    if store.equip_reward("avatar_robot", inventory):
        print(f"  ✅ Equipped! Active avatar: {inventory.active_avatar}")

    print(f"\nInventory: {len(inventory.owned_rewards)} items")
