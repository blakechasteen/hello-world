"""
Schedule Parser for Elle Core

Reads coz/schedule.md and extracts:
- Monthly focus areas
- Seasonal priorities
- Recommendations based on current month

Created: 2025-11-15
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from enum import Enum


class Month(Enum):
    """Months of the year"""
    JANUARY = 1
    FEBRUARY = 2
    MARCH = 3
    APRIL = 4
    MAY = 5
    JUNE = 6
    JULY = 7
    AUGUST = 8
    SEPTEMBER = 9
    OCTOBER = 10
    NOVEMBER = 11
    DECEMBER = 12


class Season(Enum):
    """Seasons"""
    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"


@dataclass
class MonthlyFocus:
    """Monthly focus area"""
    month: Month
    focus: str
    notes: str = ""

    def get_season(self) -> Season:
        """Get season for this month"""
        if self.month in [Month.DECEMBER, Month.JANUARY, Month.FEBRUARY]:
            return Season.WINTER
        elif self.month in [Month.MARCH, Month.APRIL, Month.MAY]:
            return Season.SPRING
        elif self.month in [Month.JUNE, Month.JULY, Month.AUGUST]:
            return Season.SUMMER
        else:
            return Season.FALL

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "month": self.month.name,
            "month_number": self.month.value,
            "focus": self.focus,
            "season": self.get_season().value,
            "notes": self.notes,
        }


class ScheduleParser:
    """Parser for schedule.md file"""

    def __init__(self, schedule_path: str = "coz/schedule.md"):
        self.schedule_path = Path(schedule_path)
        self.monthly_focus: Dict[Month, MonthlyFocus] = {}

    def parse(self) -> Dict[Month, MonthlyFocus]:
        """Parse schedule.md file"""
        if not self.schedule_path.exists():
            raise FileNotFoundError(f"Schedule file not found: {self.schedule_path}")

        with open(self.schedule_path, 'r') as f:
            content = f.read()

        self.monthly_focus = self._parse_schedule_table(content)
        return self.monthly_focus

    def _parse_schedule_table(self, content: str) -> Dict[Month, MonthlyFocus]:
        """Parse the monthly schedule table"""
        monthly_focus = {}

        # Find the table
        table_start = content.find('| Month |')
        if table_start == -1:
            return monthly_focus

        # Find the end of table
        table_end = content.find('\n\n', table_start)
        if table_end == -1:
            table_end = len(content)

        table_text = content[table_start:table_end]

        # Parse table rows
        lines = table_text.split('\n')
        for line in lines[2:]:  # Skip header and separator
            if not line.strip() or line.startswith('|---'):
                continue

            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 4:
                continue

            try:
                month_str = parts[1].strip()
                focus = parts[2].strip()
                notes = parts[3].strip() if len(parts) > 3 else ""

                month = self._parse_month(month_str)
                if month:
                    monthly_focus[month] = MonthlyFocus(
                        month=month,
                        focus=focus,
                        notes=notes,
                    )
            except (ValueError, IndexError):
                continue

        return monthly_focus

    def _parse_month(self, month_str: str) -> Optional[Month]:
        """Parse month string"""
        month_str = month_str.lower().strip()

        month_map = {
            'january': Month.JANUARY,
            'february': Month.FEBRUARY,
            'march': Month.MARCH,
            'april': Month.APRIL,
            'may': Month.MAY,
            'june': Month.JUNE,
            'july': Month.JULY,
            'august': Month.AUGUST,
            'september': Month.SEPTEMBER,
            'october': Month.OCTOBER,
            'november': Month.NOVEMBER,
            'december': Month.DECEMBER,
        }

        for key, month in month_map.items():
            if key in month_str:
                return month

        return None

    def get_current_focus(self, now: Optional[datetime] = None) -> Optional[MonthlyFocus]:
        """Get current month's focus"""
        if now is None:
            now = datetime.now()

        current_month = Month(now.month)
        return self.monthly_focus.get(current_month)

    def get_focus_by_month(self, month: Month) -> Optional[MonthlyFocus]:
        """Get focus for specific month"""
        return self.monthly_focus.get(month)

    def get_focus_by_month_number(self, month_num: int) -> Optional[MonthlyFocus]:
        """Get focus by month number (1-12)"""
        try:
            month = Month(month_num)
            return self.monthly_focus.get(month)
        except ValueError:
            return None

    def get_seasonal_focus(self, season: Season) -> List[MonthlyFocus]:
        """Get all focuses for a season"""
        return [
            focus for focus in self.monthly_focus.values()
            if focus.get_season() == season
        ]

    def get_current_season(self, now: Optional[datetime] = None) -> Season:
        """Get current season"""
        if now is None:
            now = datetime.now()

        month = Month(now.month)
        focus = self.monthly_focus.get(month)
        if focus:
            return focus.get_season()

        # Fallback calculation
        if now.month in [12, 1, 2]:
            return Season.WINTER
        elif now.month in [3, 4, 5]:
            return Season.SPRING
        elif now.month in [6, 7, 8]:
            return Season.SUMMER
        else:
            return Season.FALL

    def get_upcoming_focuses(self, months_ahead: int = 3, now: Optional[datetime] = None) -> List[MonthlyFocus]:
        """Get upcoming monthly focuses"""
        if now is None:
            now = datetime.now()

        current_month_num = now.month
        upcoming = []

        for i in range(1, months_ahead + 1):
            next_month_num = ((current_month_num - 1 + i) % 12) + 1
            focus = self.get_focus_by_month_number(next_month_num)
            if focus:
                upcoming.append(focus)

        return upcoming

    def get_recommendations_for_current_month(self, now: Optional[datetime] = None) -> Dict:
        """Get recommendations based on current month"""
        if now is None:
            now = datetime.now()

        current_focus = self.get_current_focus(now)
        current_season = self.get_current_season(now)
        upcoming = self.get_upcoming_focuses(3, now)

        return {
            "current_month": Month(now.month).name,
            "current_focus": current_focus.to_dict() if current_focus else None,
            "current_season": current_season.value,
            "upcoming_focuses": [f.to_dict() for f in upcoming],
            "recommendations": self._generate_recommendations(current_focus, current_season),
        }

    def _generate_recommendations(self, focus: Optional[MonthlyFocus], season: Season) -> Dict:
        """Generate recommendations based on focus and season"""
        recommendations = {
            "primary_focus": focus.focus if focus else "No focus defined",
            "suggested_tasks": [],
            "seasonal_tips": [],
        }

        if focus:
            # Add specific recommendations based on focus
            focus_lower = focus.focus.lower()

            if "planning" in focus_lower or "maintenance" in focus_lower:
                recommendations["suggested_tasks"].extend([
                    "Review and update Standard Operating Procedures",
                    "Audit tools and equipment condition",
                    "Plan next month's production schedule",
                    "Review financial performance",
                ])

            elif "seed" in focus_lower or "nursery" in focus_lower:
                recommendations["suggested_tasks"].extend([
                    "Prepare seed starting materials",
                    "Set up growing beds",
                    "Plan transplanting schedule",
                    "Acquire seeds and supplies",
                ])

            elif "honey" in focus_lower:
                recommendations["suggested_tasks"].extend([
                    "Harvest honey from hives",
                    "Process and bottle honey",
                    "Prepare gift packaging",
                    "Monitor hive health",
                ])

            elif "canning" in focus_lower or "preserve" in focus_lower:
                recommendations["suggested_tasks"].extend([
                    "Sterilize jars and equipment",
                    "Prepare preservation recipes",
                    "Create preservation station",
                    "Label and organize inventory",
                ])

            elif "bread" in focus_lower or "meal" in focus_lower:
                recommendations["suggested_tasks"].extend([
                    "Bake test batches",
                    "Refine recipes",
                    "Prepare packaging",
                    "Schedule production batches",
                ])

            elif "biochar" in focus_lower or "compost" in focus_lower:
                recommendations["suggested_tasks"].extend([
                    "Prepare biochar kiln",
                    "Inoculate with beneficial microbes",
                    "Test compost readiness",
                    "Package and label",
                ])

        # Seasonal tips
        if season == Season.WINTER:
            recommendations["seasonal_tips"].extend([
                "Use slower season to plan and maintain equipment",
                "Focus on indoor production and storage",
                "Review and improve SOPs",
            ])
        elif season == Season.SPRING:
            recommendations["seasonal_tips"].extend([
                "Prepare for growing season",
                "Clean and repair outdoor equipment",
                "Start seed propagation",
            ])
        elif season == Season.SUMMER:
            recommendations["seasonal_tips"].extend([
                "Maximize outdoor production",
                "Monitor water and pest issues",
                "Plan for harvest preservation",
            ])
        else:  # FALL
            recommendations["seasonal_tips"].extend([
                "Prepare for winter dormancy",
                "Focus on harvest and preservation",
                "Prepare for reduced outdoor production",
            ])

        return recommendations

    def get_summary(self) -> Dict:
        """Get schedule summary"""
        seasons = {}
        for season in Season:
            focuses = self.get_seasonal_focus(season)
            seasons[season.value] = [f.focus for f in focuses]

        return {
            "total_months_defined": len(self.monthly_focus),
            "months_defined": list(self.monthly_focus.keys()),
            "by_season": seasons,
        }
