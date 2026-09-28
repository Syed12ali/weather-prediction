from __future__ import annotations

from typing import Dict

import pandas as pd


def summarize_metrics(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for name, payload in metrics.items():
        rows.append(
            {
                "model": name,
                "accuracy": payload.get("accuracy", 0.0),
                "f1": payload.get("f1", 0.0),
            }
        )
    return pd.DataFrame(rows)
