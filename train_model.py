# train_model.py
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

from utils.nasa import fetch_koi_dataframe

MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

FEATURES = [
    "koi_period","koi_prad","koi_model_snr",
    "koi_depth","koi_duration","koi_steff","koi_slogg","koi_srad"
]

def clean_and_label(df: pd.DataFrame):
    # Keep only definite labels; drop "CANDIDATE/NOT DISPOSITIONED"
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

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))

    # Save artifacts
    joblib.dump(pipe, MODEL_DIR/"model.pkl")
    (MODEL_DIR/"feature_columns.json").write_text(json.dumps(FEATURES, indent=2))
    print("Saved model to model/model.pkl")

if __name__ == "__main__":
    main()
