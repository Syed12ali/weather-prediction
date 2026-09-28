import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()


class OpenWeatherClient:
    """Wrapper for fetching weather data from OpenWeatherMap's API."""

    def __init__(self, api_key: Optional[str] = None, latitude: float = 52.52, longitude: float = 13.41):
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        self.latitude = latitude
        self.longitude = longitude
        self.base_url = "https://api.openweathermap.org/data/2.5"

    def _synthetic_hourly_forecast(self, hours: int = 48) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        for idx in range(hours):
            time_point = now + timedelta(hours=idx)
            temp = 22 + 8 * (idx / max(hours, 1))
            humidity = 52 + 25 * ((idx % 7) / 7)
            rain = 0.3 + (idx % 5) * 0.7
            if idx in {2, 5, 9}:
                rain += 18.0
                temp += 8.0
                humidity += 20
            wind_speed = 7 + (idx % 4) * 2.5
            weather_main = "Thunderstorm" if idx in {2, 9} else "Rain" if idx % 3 == 0 else "Clear"
            records.append(
                {
                    "dt_txt": time_point.strftime("%Y-%m-%d %H:%M:%S"),
                    "main": {"temp": round(temp, 2), "humidity": round(humidity, 2), "pressure": 1013 + (idx % 3) * 4},
                    "wind": {"speed": round(wind_speed, 2)},
                    "rain": {"3h": round(rain, 2)},
                    "weather": [{"main": weather_main}],
                }
            )
        return records

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("OPENWEATHER_API_KEY is missing. Add it to your .env file or environment.")

        payload = {"appid": self.api_key, "lat": self.latitude, "lon": self.longitude}
        if params:
            payload.update(params)

        response = requests.get(f"{self.base_url}/{endpoint}", params=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_current_weather(self) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "main": {"temp": 22.5, "humidity": 60, "pressure": 1015},
                "wind": {"speed": 9.0},
                "weather": [{"main": "Clouds"}],
            }
        return self._request("weather", {"units": "metric"})

    def get_hourly_forecast(self, hours: int = 48) -> List[Dict[str, Any]]:
        if not self.api_key:
            return self._synthetic_hourly_forecast(hours=hours)
        payload = self._request("forecast", {"units": "metric", "cnt": hours})
        return payload.get("list", [])

    def get_daily_forecast(self, days: int = 7) -> Dict[str, Any]:
        return self._request("forecast/daily", {"units": "metric", "cnt": days})

    def get_historical_weather(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch daily aggregated observations for a date range using the historical weather API."""
        if not self.api_key:
            raise ValueError("OPENWEATHER_API_KEY is missing.")

        records: List[Dict[str, Any]] = []
        cursor = start
        while cursor <= end:
            ts = int(cursor.timestamp())
            payload = self._request("onecall/timemachine", {"dt": ts, "units": "metric"})
            for item in payload.get("data", []):
                records.append(
                    {
                        "timestamp": datetime.fromtimestamp(item["dt"], tz=timezone.utc).isoformat(),
                        "temperature": item.get("temp"),
                        "feels_like": item.get("feels_like"),
                        "humidity": item.get("humidity"),
                        "pressure": item.get("pressure"),
                        "wind_speed": item.get("wind_speed"),
                        "rain_1h": item.get("rain", {}).get("1h", 0.0),
                        "weather_main": item.get("weather", [{}])[0].get("main", "Unknown"),
                    }
                )
            cursor += timedelta(days=1)
        return pd.DataFrame(records)
