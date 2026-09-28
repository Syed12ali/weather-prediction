from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


class XAIExplainer:
    """Explain model decisions with SHAP values and feature importance plots."""

    def __init__(self, model):
        self.model = model

    def explain(self, X: pd.DataFrame, sample_index: int = 0) -> Dict[str, object]:
        explainer = shap.Explainer(self.model, X)
        values = explainer(X.iloc[[sample_index]])
        base_values = np.asarray(values.base_values).reshape(-1)
        return {
            "values": values,
            "base_values": values.base_values,
            "feature_names": list(X.columns),
            "expected_value": float(base_values[0]) if base_values.size else 0.0,
        }

    def plot_summary(self, X: pd.DataFrame, output_path: str = "outputs/shap_summary.png") -> str:
        explainer = shap.Explainer(self.model, X)
        shap_values = explainer(X)
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(shap_values, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        return output_path

    def feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        explainer = shap.Explainer(self.model, X)
        shap_values = explainer(X)
        values = np.asarray(shap_values.values)
        if values.ndim == 3:
            values = values[..., 0]
        mean_abs = np.abs(values).mean(axis=0)

        summary = pd.DataFrame(
            {
                "feature": X.columns,
                "mean_abs_shap": mean_abs,
            }
        ).sort_values("mean_abs_shap", ascending=False)
        return summary.reset_index(drop=True)
