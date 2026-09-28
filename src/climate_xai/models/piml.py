from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


@dataclass
class PhysicsInformedModel:
    """Simple physics-informed ensemble model for extreme climate events."""

    random_state: int = 42

    def prepare(self, df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        feature_frame = df.copy()
        feature_frame = feature_frame.drop(columns=["timestamp"], errors="ignore")

        explicit_exclusions = ["weather_main", "weather_main_encoded"]
        feature_frame = feature_frame.drop(columns=[col for col in explicit_exclusions if col in feature_frame.columns], errors="ignore")

        non_numeric_cols = feature_frame.select_dtypes(exclude=["number"]).columns.tolist()
        if target in non_numeric_cols:
            non_numeric_cols.remove(target)
        if non_numeric_cols:
            feature_frame = feature_frame.drop(columns=non_numeric_cols, errors="ignore")

        features = feature_frame.drop(columns=[target], errors="ignore")
        labels = feature_frame[target].astype(int)

        if labels.nunique() < 2:
            labels = pd.Series(np.where(labels == 0, 0, 1), index=labels.index)

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=self.random_state, stratify=labels
        )
        return X_train, y_train, X_test, y_test

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
        if y_train.nunique() < 2:
            fallback = DummyClassifier(strategy="most_frequent")
            fallback.fit(X_train, y_train)
            return {"dummy_classifier": fallback}

        models = {
            "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=2,
                random_state=self.random_state,
                class_weight="balanced",
            ),
        }

        trained = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained[name] = model

        return trained

    def evaluate(self, model_dict: Dict[str, object], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        results = {}
        for name, model in model_dict.items():
            preds = model.predict(X_test)
            results[name] = {
                "accuracy": float(accuracy_score(y_test, preds)),
                "f1": float(f1_score(y_test, preds, zero_division=0)),
                "report": classification_report(y_test, preds, zero_division=0, output_dict=True),
            }
        return results

    def predict_risk(self, model_dict: Dict[str, object], features: pd.DataFrame) -> Dict[str, np.ndarray]:
        return {name: model.predict(features) for name, model in model_dict.items()}
