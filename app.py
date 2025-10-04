import json
import io
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from utils.transit import bls_search
from utils.orbit import semimajor_axis_au, ellipse_xyz

# --- Sidebar / Title ---
st.set_page_config(page_title="Exo-Seeker", page_icon="🪐", layout="wide")
st.title("🪐 Exo-Seeker")
st.caption("Exoplanet exploration using NASA dataset")

# --- Load model ---
MODEL_PATH = Path("model/model.pkl")
FEAT_PATH = Path("model/feature_columns.json")
model = None
features = None
if MODEL_PATH.exists() and FEAT_PATH.exists():
    model = joblib.load(MODEL_PATH)
    features = json.loads(FEAT_PATH.read_text())

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Explore & Predict", "Upload Light Curve", "3D Orbit"])

# ---------- Tab 1: Explore & Predict ----------
with tab1:
    st.subheader("Predict KOI candidacy from NASA features")
    st.write("Paste KOI feature values or load an example row.")

    col1, col2 = st.columns([2,1])

    with col1:
        default_row = {
            "koi_period": 10.0,
            "koi_prad": 1.5,
            "koi_model_snr": 10.0,
            "koi_depth": 500.0,     # ppm
            "koi_duration": 3.0,    # hours
            "koi_steff": 5700.0,
            "koi_slogg": 4.4,
            "koi_srad": 1.0
        }
        user_vals = {}
        for k,v in default_row.items():
            user_vals[k] = st.number_input(k, value=float(v))

        if st.button("Predict", type="primary", disabled=model is None):
            X = pd.DataFrame([user_vals])
            proba = float(model.predict_proba(X)[0,1])
            pred = int(model.predict(X)[0])
            st.success(f"Predicted **{'CONFIRMED-like' if pred==1 else 'FALSE-POSITIVE-like'}** with probability {proba:.2f}")

    with col2:
        st.info("Tip: You can train/update the model via `train_model.py` and redeploy.")

# ---------- Tab 2: Upload Light Curve ----------
with tab2:
    st.subheader("Quick transit search (BLS)")
    uploaded = st.file_uploader("Upload light curve CSV with columns: time, flux", type=["csv"])
    if uploaded is not None:
        lc = pd.read_csv(uploaded)
    else:
        if st.button("Load sample"):
            lc = pd.read_csv("data/sample_lightcurve.csv")
        else:
            lc = None

    if lc is not None and {"time","flux"}.issubset(lc.columns):
        info, phase, flux = bls_search(lc["time"].values, lc["flux"].values)

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lc["time"], y=lc["flux"], mode="markers", name="Light curve"))
            fig.update_layout(xaxis_title="Time (days)", yaxis_title="Relative flux", title="Raw light curve")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=phase, y=flux, mode="markers", name="Phase-folded"))
            fig2.update_layout(xaxis_title="Phase (P)", yaxis_title="Flux", title="Phase-folded (BLS)")
            st.plotly_chart(fig2, use_container_width=True)

        st.json(info)
        st.caption("BLS: standard approach for finding transiting planets.")  # astropy docs

# ---------- Tab 3: 3D Orbit ----------
with tab3:
    st.subheader("3D orbital visualization")
    st.write("Enter period (days) and star mass (solar masses), or set a directly (AU).")
    mode = st.radio("Choose input", ["Period + Star mass", "Semi-major axis (AU)"], horizontal=True)

    if mode == "Period + Star mass":
        P = st.number_input("Orbital period (days)", value=10.0, min_value=0.1)
        Mstar = st.number_input("Star mass (Solar masses)", value=1.0, min_value=0.1)
        a_au = semimajor_axis_au(P, Mstar)
    else:
        a_au = st.number_input("Semi-major axis (AU)", value=0.1, min_value=0.01)

    e = st.slider("Eccentricity", 0.0, 0.9, 0.0, 0.01)
    inc = st.slider("Inclination (deg)", 0.0, 90.0, 30.0, 1.0)

    x, y, z = ellipse_xyz(a_au, e=e, inc_deg=inc)
    star = go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=5), name="Star")
    orbit = go.Scatter3d(x=x, y=y, z=z, mode="lines", name="Orbit")

    fig3 = go.Figure(data=[star, orbit])
    fig3.update_layout(
        scene=dict(
            xaxis_title="AU", yaxis_title="AU", zaxis_title="AU",
            aspectmode="data"
        ),
        title=f"Semi-major axis ≈ {a_au:.3f} AU"
    )
    st.plotly_chart(fig3, use_container_width=True)
