from __future__ import annotations

from typing import Iterable, List, Tuple

import pandas as pd


class ClimateFeatureEngineer:
    """Build physically meaningful climate features from raw weather records."""

    @staticmethod
    def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
        features = df.copy()
        if "temperature" not in features.columns:
            raise KeyError("DataFrame must include 'temperature'.")

        features["temperature_sq"] = features["temperature"] ** 2
        features["humidity_sq"] = features["humidity"].fillna(0) ** 2
        features["rain_intensity"] = features.get("rain_1h", 0.0).fillna(0.0)
        features["heat_index"] = features["temperature"] + 0.1 * (features["humidity"].fillna(0) - 50)
        features["pressure_anomaly"] = features["pressure"].fillna(1013) - 1013
        features["wind_x"] = features.get("wind_speed", 0.0).fillna(0.0) * 0.8
        features["wind_y"] = features.get("wind_speed", 0.0).fillna(0.0) * 0.2

        if "weather_main" in features.columns:
            weather_map = {
                "Clear": 0,
                "Clouds": 1,
                "Rain": 2,
                "Drizzle": 3,
                "Thunderstorm": 4,
                "Snow": 5,
            }
            features["weather_main_encoded"] = features["weather_main"].fillna("Clear").map(weather_map).fillna(0)
        return features

    @staticmethod
    def create_targets(df: pd.DataFrame) -> pd.DataFrame:
        targets = df.copy()
        targets["extreme_rainfall"] = (targets.get("rain_1h", 0.0) >= 10).astype(int)
        targets["heatwave_risk"] = ((targets.get("temperature", 0.0) >= 30) & (targets.get("humidity", 0.0) >= 40)).astype(int)
        targets["flood_risk"] = ((targets.get("rain_1h", 0.0) >= 5) | (targets.get("temperature", 0.0) >= 28)).astype(int)
        return targets

    @staticmethod
    def build_sequences(df: pd.DataFrame, target_col: str, window: int = 12) -> Tuple[pd.DataFrame, pd.Series]:
        if window <= 1:
            raise ValueError("window must be greater than 1")

        rows = []
        labels = []
        for i in range(len(df) - window):
            sequence = df.iloc[i : i + window].copy()
            rows.append(sequence.drop(columns=[target_col], errors="ignore").to_numpy().flatten())
            labels.append(df.iloc[i + window][target_col])
        return pd.DataFrame(rows), pd.Series(labels)
