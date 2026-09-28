from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

if __package__ in {None, ""}:
    project_src = Path(__file__).resolve().parent.parent
    if str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
    from climate_xai.data.openweather_client import OpenWeatherClient
    from climate_xai.data.preprocessing import ClimateFeatureEngineer
    from climate_xai.models.piml import PhysicsInformedModel
    from climate_xai.xai.explain import XAIExplainer
else:
    from .data.openweather_client import OpenWeatherClient
    from .data.preprocessing import ClimateFeatureEngineer
    from .models.piml import PhysicsInformedModel
    from .xai.explain import XAIExplainer


@dataclass
class ExtremeWeatherPipeline:
    """End-to-end pipeline for climate forecasting and explainability."""

    latitude: float = 52.52
    longitude: float = 13.41
    api_key: str | None = None

    def __post_init__(self) -> None:
        self.client = OpenWeatherClient(api_key=self.api_key, latitude=self.latitude, longitude=self.longitude)
        self.feature_engineer = ClimateFeatureEngineer()
        self.model = PhysicsInformedModel()

    def fetch_forecast_data(self, hours: int = 48) -> pd.DataFrame:
        raw = self.client.get_hourly_forecast(hours=hours)
        rows = []
        for item in raw:
            main = item.get("main", {})
            weather = item.get("weather", [{}])[0] if item.get("weather") else {}
            row = {
                "timestamp": item.get("dt_txt"),
                "temperature": main.get("temp"),
                "humidity": main.get("humidity"),
                "pressure": main.get("pressure"),
                "wind_speed": item.get("wind", {}).get("speed"),
                "rain_1h": item.get("rain", {}).get("3h", 0.0) if item.get("rain") else 0.0,
                "weather_main": weather.get("main"),
            }
            rows.append(row)

        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(
                {
                    "timestamp": [],
                    "temperature": [],
                    "humidity": [],
                    "pressure": [],
                    "wind_speed": [],
                    "rain_1h": [],
                    "weather_main": [],
                }
            )
        return frame

    def build_training_dataset(self, history: pd.DataFrame, target: str) -> pd.DataFrame:
        processed = self.feature_engineer.add_physical_features(history)
        targets = self.feature_engineer.create_targets(processed)
        dataset = processed.merge(targets[[target]], left_index=True, right_index=True)
        return dataset

    def train(self, dataset: pd.DataFrame, target: str) -> Dict[str, object]:
        X_train, y_train, X_test, y_test = self.model.prepare(dataset, target)
        trained = self.model.train(X_train, y_train)
        metrics = self.model.evaluate(trained, X_test, y_test)
        return {"train": X_train, "labels": y_train, "test": X_test, "truth": y_test, "models": trained, "metrics": metrics}

    def explain(self, model_dict: Dict[str, object], X: pd.DataFrame, model_name: str = "random_forest") -> Dict[str, object]:
        explainer = XAIExplainer(model_dict[model_name])
        return explainer.explain(X)

    def save_summary(self, output_dir: str = "outputs") -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return out


if __name__ == "__main__":
    pipeline = ExtremeWeatherPipeline(api_key=None)
    forecast = pipeline.fetch_forecast_data(hours=12)
    dataset = pipeline.build_training_dataset(forecast, "extreme_rainfall")
    result = pipeline.train(dataset, "extreme_rainfall")
    print("Forecast rows:", len(forecast))
    print("Model metrics:")
    for name, payload in result["metrics"].items():
        print(name, payload["accuracy"], payload["f1"])
