# weather-prediction

A climate-focused Python project integrating physics-informed machine learning and Explainable AI (XAI) for predicting extreme rainfall, heatwaves, and flood risk using open weather data APIs.

## Project goals
- Forecast extreme rainfall, heatwave risk, and flood risk events.
- Blend physically meaningful climate features with machine learning.
- Use explainability tools to interpret model behavior and improve trust.
- Work with OpenWeather API data in a reproducible, extensible Python pipeline.

## Project structure
- `src/climate_xai/data/` — data collection and feature engineering
- `src/climate_xai/models/` — models and training logic
- `src/climate_xai/xai/` — explainability and SHAP-based introspection
- `src/climate_xai/pipeline.py` — end-to-end forecasting pipeline
- `examples/demo.py` — quick demonstration pipeline
- `tests/` — automated validation for the Python pipeline

## Setup

1. Create a virtual environment:
   python -m venv .venv
   source .venv/bin/activate

2. Install dependencies:
   pip install -r requirements.txt

3. Configure API access:
   cp .env.example .env
   Add your OpenWeather API key to `.env`.

4. Run the CLI demo:
   PYTHONPATH=src python examples/demo.py

5. Run the pipeline directly:
   python src/climate_xai/pipeline.py

## Notes
- The current implementation is a practical prototype built around OpenWeather APIs and scikit-learn.
- It uses a simple physics-informed feature set: temperature, humidity, pressure anomaly, rain intensity, heat index, and wind terms.
- SHAP is included to explain feature attribution for key predictions.
- When no API key is available, the pipeline falls back to synthetic climate scenarios so it still runs locally.

## Future extensions
- Add ERA5 / weather station ingestion
- Incorporate spatiotemporal graph models for weather dynamics
- Add uncertainty estimation and calibration
- Extend into a richer climate analytics toolkit
