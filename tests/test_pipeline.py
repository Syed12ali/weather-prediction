from climate_xai.pipeline import ExtremeWeatherPipeline


def test_fetch_forecast_data_works_without_api_key():
    pipeline = ExtremeWeatherPipeline(api_key=None)
    df = pipeline.fetch_forecast_data(hours=6)

    assert not df.empty
    assert len(df) == 6
    assert {"temperature", "humidity", "pressure", "wind_speed", "rain_1h"}.issubset(set(df.columns))


def test_training_pipeline_handles_synthetic_weather_data():
    pipeline = ExtremeWeatherPipeline(api_key=None)
    df = pipeline.fetch_forecast_data(hours=12)
    dataset = pipeline.build_training_dataset(df, "extreme_rainfall")

    result = pipeline.train(dataset, "extreme_rainfall")

    assert "metrics" in result
    assert set(result["metrics"]).issubset({"logistic_regression", "random_forest"})
