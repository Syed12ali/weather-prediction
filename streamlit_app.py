from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from climate_xai.pipeline import ExtremeWeatherPipeline

st.set_page_config(page_title="Climate Risk Forecast", page_icon="🌦️", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top left, #123d6a 0%, #0b1f32 30%, #071721 100%);
            color: #ecf8ff;
        }
        .stSidebar {
            background: rgba(3, 12, 23, 0.95);
            border-right: 1px solid rgba(126, 249, 198, 0.18);
        }
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3, p {
            color: #eefbff;
        }
        [data-testid="stMetricValue"] {
            color: #9ef7c4;
            font-weight: 800;
            font-size: 1.5rem;
        }
        [data-testid="stMetricLabel"] {
            color: #dfeefb;
        }
        div.stButton > button {
            background: linear-gradient(90deg, #00c2ff 0%, #7ef9c6 100%);
            color: #031828;
            border: none;
            border-radius: 12px;
            font-weight: 800;
            padding: 0.8rem 1.4rem;
            box-shadow: 0 10px 25px rgba(0, 194, 255, 0.3);
        }
        div.stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 28px rgba(126, 249, 198, 0.38);
        }
        .stSuccess {
            background: rgba(19, 74, 64, 0.9);
            border: 1px solid rgba(126, 249, 198, 0.2);
            color: #dbfff4;
        }
        .stInfo {
            background: rgba(11, 35, 62, 0.92);
            border: 1px solid rgba(0, 194, 255, 0.25);
            color: #dff8ff;
        }
        .stDataFrame {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .main .block-container {
            max-width: 1400px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="background: linear-gradient(135deg, rgba(0,194,255,0.16), rgba(126,249,198,0.12)); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;">
        <h1 style="margin:0; color:#ffffff;">🌦️ Climate Risk Forecast Dashboard</h1>
        <p style="margin:0.4rem 0 0 0; color:#dceffc;">Physics-informed forecasting for extreme rainfall, heatwaves, and flood risk</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Weather settings")
    latitude = st.number_input("Latitude", value=52.52, format="%.2f")
    longitude = st.number_input("Longitude", value=13.41, format="%.2f")
    hours = st.slider("Forecast hours", 6, 72, 24)
    run = st.button("Run forecast", use_container_width=True)

    st.caption("Live data is enabled through the environment configuration, not the dashboard UI.")

api_key = None
if run:
    pipeline = ExtremeWeatherPipeline(latitude=float(latitude), longitude=float(longitude), api_key=api_key)
    data = pipeline.fetch_forecast_data(hours=int(hours))
    dataset = pipeline.build_training_dataset(data, "extreme_rainfall")
    model_result = pipeline.train(dataset, "extreme_rainfall")

    st.success("Forecast completed successfully.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Forecast window", f"{hours} h")
    col2.metric("Location", f"{latitude:.2f}, {longitude:.2f}")
    col3.metric("Data source", "Environment data" if api_key else "Synthetic fallback")

    st.subheader("Forecast sample")
    st.dataframe(data.head(10), use_container_width=True)

    st.subheader("Model performance")
    metrics = model_result["metrics"]
    metric_df = pd.DataFrame(
        [
            {
                "Model": name,
                "Accuracy": payload["accuracy"],
                "F1": payload["f1"],
            }
            for name, payload in metrics.items()
        ]
    )
    st.dataframe(metric_df, use_container_width=True)

    sample = model_result["test"].head(5)
    explanation = pipeline.explain(model_result["models"], sample, model_name="random_forest")
    st.subheader("Explainability")
    st.json({"feature_count": len(explanation["feature_names"]), "expected_value": explanation["expected_value"]})
else:
    st.info("Use the sidebar to configure a location and start the forecast. If you want live OpenWeather data, set the environment variable before launching the app.")
