import json
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import joblib

from utils.nasa import fetch_koi_dataframe  

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

FEATURES = [
    "koi_period","koi_prad","koi_model_snr",
    "koi_depth","koi_duration","koi_steff","koi_slogg","koi_srad"
]

def clean_and_label(df: pd.DataFrame):
    df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
    df["label"] = (df["koi_disposition"] == "CONFIRMED").astype(int)
    return df

def main():
    print("Fetching KOI data from NASA TAP…")
    df = fetch_koi_dataframe()
    df = clean_and_label(df)

    X = df[FEATURES].copy()
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # transformer(s) -> final estimator
    base = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08)
    #base = LogisticRegression(max_iter=1000, class_weight="balanced")
    calibrated = CalibratedClassifierCV(base, cv=3, method="isotonic")

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", calibrated) 
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, digits=3))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Brier score (lower is better): {brier_score_loss(y_test, y_proba):.4f}")

    
    joblib.dump(pipe, MODEL_DIR/"model.pkl")
    (MODEL_DIR/"feature_columns.json").write_text(json.dumps(FEATURES, indent=2))
    print("Saved model to model/model.pkl")

if __name__ == "__main__":
    main()
