import os
from pathlib import Path

import pandas as pd

from climate_xai.pipeline import ExtremeWeatherPipeline


def main() -> None:
    print("Loading climate forecasting demo...")
    pipeline = ExtremeWeatherPipeline(latitude=52.52, longitude=13.41)

    raw_forecast = pipeline.fetch_forecast_data(hours=24)
    print(raw_forecast.head())

    dataset = pipeline.build_training_dataset(raw_forecast, "extreme_rainfall")
    print(dataset.head())

    trained = pipeline.train(dataset, "extreme_rainfall")
    print(trained["metrics"])

    sample = trained["test"].head(5)
    explanation = pipeline.explain(trained["models"], sample, model_name="random_forest")
    print({"feature_count": len(explanation["feature_names"]), "expected_value": explanation["expected_value"]})

    out_dir = pipeline.save_summary("outputs")
    print(f"Artifacts saved under: {out_dir}")


if __name__ == "__main__":
    main()
