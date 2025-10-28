"""
Example WeatherDataProvider protocol implementation.

Demonstrates how to create a custom weather integration following
the protocol-based design pattern.
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta

from apps.keep.protocols import WeatherDataProvider


class MockWeatherProvider:
    """
    Mock weather provider for testing and demonstration.

    In production, replace with real weather API (OpenWeatherMap, etc.)
    """

    def __init__(self, location: str = "default"):
        """
        Initialize weather provider.

        Args:
            location: Default location
        """
        self.location = location
        self.api_key = None  # Would be real API key in production

    async def get_current_weather(self, location: str) -> Dict[str, Any]:
        """
        Get current weather conditions.

        Args:
            location: Location to query

        Returns:
            Weather data dict
        """
        # Mock implementation - would call real API
        return {
            "location": location,
            "temperature": 72.0,
            "humidity": 65,
            "wind_speed": 5.0,
            "conditions": "Partly Cloudy",
            "precipitation": 0.0,
            "timestamp": datetime.now().isoformat(),
        }

    async def get_forecast(
        self,
        location: str,
        days: int
    ) -> List[Dict[str, Any]]:
        """
        Get weather forecast.

        Args:
            location: Location to query
            days: Number of days

        Returns:
            List of forecast data dicts
        """
        # Mock implementation
        forecast = []
        base_temp = 70.0

        for i in range(days):
            date = datetime.now() + timedelta(days=i)
            forecast.append({
                "date": date.date().isoformat(),
                "temperature_high": base_temp + (i % 3) * 5,
                "temperature_low": base_temp - 10,
                "conditions": ["Sunny", "Cloudy", "Rainy"][i % 3],
                "precipitation_chance": [10, 30, 70][i % 3],
            })

        return forecast

    def is_suitable_for_inspection(
        self,
        weather: Dict[str, Any]
    ) -> bool:
        """
        Determine if weather is suitable for hive inspection.

        Args:
            weather: Weather data dict

        Returns:
            True if suitable for inspection
        """
        # Best inspection conditions:
        # - Temperature between 60-85°F
        # - Wind speed < 15 mph
        # - No precipitation
        # - Not overcast/threatening

        temp = weather.get("temperature", 0)
        wind = weather.get("wind_speed", 999)
        precip = weather.get("precipitation", 0)
        conditions = weather.get("conditions", "").lower()

        if temp < 60 or temp > 85:
            return False

        if wind > 15:
            return False

        if precip > 0:
            return False

        if any(word in conditions for word in ["storm", "rain", "snow"]):
            return False

        return True


class OpenWeatherMapProvider:
    """
    Real OpenWeatherMap API implementation.

    Requires API key: https://openweathermap.org/api
    """

    def __init__(self, api_key: str):
        """
        Initialize with API key.

        Args:
            api_key: OpenWeatherMap API key
        """
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"

    async def get_current_weather(self, location: str) -> Dict[str, Any]:
        """Get current weather from OpenWeatherMap."""
        import aiohttp

        url = f"{self.base_url}/weather"
        params = {
            "q": location,
            "appid": self.api_key,
            "units": "imperial"  # Fahrenheit
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()

                return {
                    "location": location,
                    "temperature": data["main"]["temp"],
                    "humidity": data["main"]["humidity"],
                    "wind_speed": data["wind"]["speed"],
                    "conditions": data["weather"][0]["description"],
                    "precipitation": data.get("rain", {}).get("1h", 0.0),
                    "timestamp": datetime.now().isoformat(),
                }

    async def get_forecast(
        self,
        location: str,
        days: int
    ) -> List[Dict[str, Any]]:
        """Get forecast from OpenWeatherMap."""
        import aiohttp

        url = f"{self.base_url}/forecast"
        params = {
            "q": location,
            "appid": self.api_key,
            "units": "imperial",
            "cnt": days * 8  # 3-hour intervals
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()

                # Group by day
                daily_forecast = []
                current_date = None
                day_data = []

                for item in data["list"]:
                    dt = datetime.fromtimestamp(item["dt"])
                    if current_date is None or dt.date() != current_date:
                        if day_data:
                            daily_forecast.append(self._aggregate_day(day_data))
                        current_date = dt.date()
                        day_data = []

                    day_data.append(item)

                if day_data:
                    daily_forecast.append(self._aggregate_day(day_data))

                return daily_forecast[:days]

    def _aggregate_day(self, intervals: List[Dict]) -> Dict[str, Any]:
        """Aggregate 3-hour intervals into daily summary."""
        temps = [item["main"]["temp"] for item in intervals]
        conditions = [item["weather"][0]["description"] for item in intervals]

        # Find most common condition
        from collections import Counter
        most_common = Counter(conditions).most_common(1)[0][0]

        return {
            "date": datetime.fromtimestamp(intervals[0]["dt"]).date().isoformat(),
            "temperature_high": max(temps),
            "temperature_low": min(temps),
            "conditions": most_common,
            "precipitation_chance": sum(
                item.get("pop", 0) for item in intervals
            ) / len(intervals) * 100,
        }

    def is_suitable_for_inspection(
        self,
        weather: Dict[str, Any]
    ) -> bool:
        """Check if weather is suitable for inspection."""
        # Same logic as mock implementation
        temp = weather.get("temperature", 0)
        wind = weather.get("wind_speed", 999)
        precip = weather.get("precipitation", 0)

        return 60 <= temp <= 85 and wind < 15 and precip == 0


# =============================================================================
# Usage Example
# =============================================================================

async def demonstrate_weather_integration():
    """Demonstrate weather provider usage."""
    # Use mock provider for demo
    provider = MockWeatherProvider()

    # Get current weather
    current = await provider.get_current_weather("San Francisco, CA")
    print(f"Current weather: {current['temperature']}°F, {current['conditions']}")

    # Check if suitable for inspection
    suitable = provider.is_suitable_for_inspection(current)
    print(f"Suitable for inspection: {suitable}")

    # Get forecast
    forecast = await provider.get_forecast("San Francisco, CA", days=3)
    print("\n3-day forecast:")
    for day in forecast:
        print(f"  {day['date']}: {day['temperature_high']}°F, {day['conditions']}")

    # Find best inspection day
    print("\nBest days for inspection:")
    for day in forecast:
        if day["precipitation_chance"] < 20 and 60 <= day["temperature_high"] <= 85:
            print(f"  {day['date']} - {day['conditions']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_weather_integration())
