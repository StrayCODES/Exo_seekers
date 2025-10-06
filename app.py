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

# Sidebar / Title
st.set_page_config(page_title="Exo-Seeker", page_icon="🪐", layout="wide")
st.title("🪐 Exo-Seeker")
st.caption("Exoplanet exploration using NASA dataset")

# Load model
MODEL_PATH = Path("model/model.pkl")
FEAT_PATH = Path("model/feature_columns.json")
model = None
features = None
if MODEL_PATH.exists() and FEAT_PATH.exists():
    model = joblib.load(MODEL_PATH)
    features = json.loads(FEAT_PATH.read_text())

# Tabs
tab1, tab2, tab3 = st.tabs(["Explore & Predict", "Upload Light Curve", "3D Orbit"])

# Tab 1: Explore & Predict
with tab1:
    st.subheader("Estimate how likely a candidate is a planet")
    st.write("Enter features (from NASA KOI/PS tables) or try the defaults.")

    col1, col2 = st.columns([2,1])

    with col1:
        default_row = {
            "Orbital period (days)": 9.48,
            "Planet radius (Earth radii)": 2.26,
            "Signal-to-noise (model)": 35.8,
            "Transit depth (ppm)": 615.8,
            "Transit duration (hours)": 2.95,
            "Star temperature (K)": 5455.0,
            "Star surface gravity (log g)": 446,
            "Star radius (Solar radii)": .93
        }
        ui_to_feat = {
        "Orbital period (days)": "koi_period",
        "Planet radius (Earth radii)": "koi_prad",
        "Signal-to-noise (model)": "koi_model_snr",
        "Transit depth (ppm)": "koi_depth",
        "Transit duration (hours)": "koi_duration",
        "Star temperature (K)": "koi_steff",
        "Star surface gravity (log g)": "koi_slogg",
        "Star radius (Solar radii)": "koi_srad",
    }
        
    inputs = {}
    cols = st.columns(2)
    for i, (label, value) in enumerate(default_row.items()):
        with cols[i % 2]:
            help_text = {
                "Orbital period (days)": "Time for one orbit around the star.",
                "Planet radius (Earth radii)": "Estimated planet size.",
                "Signal-to-noise (model)": "Strength of the signal vs noise.",
                "Transit depth (ppm)": "How much the star dims during transit.",
                "Transit duration (hours)": "How long the dimming lasts.",
                "Star temperature (K)": "Surface temperature of the star.",
                "Star surface gravity (log g)": "Gravity at the star’s surface.",
                "Star radius (Solar radii)": "Star size compared to the Sun."
            }.get(label, None)

            inputs[label] = st.number_input(label, value=float(value), help=help_text)

    if st.button("Estimate planet likelihood", type="primary", disabled=model is None):
        row = {ui_to_feat[k]: v for k, v in inputs.items()}
        X = pd.DataFrame([row])
        prob = float(model.predict_proba(X)[0,1])
        pred = "Likely a planet" if prob >= 0.5 else "Unlikely a planet"
        st.success(f"**{pred}** — Estimated probability: **{prob:.2f}**")
    
# Tab 2: Light Curve
with tab2:
    st.subheader("Find transits in a light curve")
    st.write("Upload a CSV with columns **time, flux** (brightness vs time), or load our sample.")

    uploaded = st.file_uploader("Upload light curve CSV", type=["csv"])
    if uploaded:
        lc = pd.read_csv(uploaded)
    elif st.button("Load sample light curve"):
        lc = pd.read_csv("data/sample_lightcurve.csv")
    else:
        lc = None

    if lc is not None and {"time","flux"}.issubset(lc.columns):
        info, phase, flux = bls_search(lc["time"].values, lc["flux"].values)

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lc["time"], y=lc["flux"], mode="markers", name="Light curve"))
            fig.update_layout(title="Brightness over time (light curve)", xaxis_title="Time (days)", yaxis_title="Relative flux")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=phase, y=flux, mode="markers", name="Phase-folded"))
            fig2.update_layout(title="Transit signal (phase-folded)", xaxis_title="Phase (0–1)", yaxis_title="Flux")
            st.plotly_chart(fig2, use_container_width=True)

        st.json({
            "Best period (days)": round(info["period"], 6),
            "Transit start (t0)": round(info["t0"], 6),
            "Transit duration (days)": round(info["duration"], 6),
            "Transit depth": round(info["depth"], 6),
            "Signal-to-noise": round(info["snr"], 3)
        })
        st.caption("Method: Box Least Squares (standard for finding planet transits).")

    
# ---------- Tab 3: 3D Orbit ----------
with tab3:
    st.subheader("3D Orbit Viewer")
    st.write("Visualize a planet’s orbit around its star. Choose how to provide the orbit size.")

    mode = st.radio("Orbit size input", ["Calculate from period & star mass", "Enter semi-major axis (AU)"], horizontal=True)

    if mode == "Calculate from period & star mass":
        P = st.number_input("Orbital period (days)", min_value=0.1, value=10.0, help="Time for one full orbit.")
        Mstar = st.number_input("Star mass (Solar masses)", min_value=0.1, value=1.0, help="Star mass compared to the Sun.")
        a_au = semimajor_axis_au(P, Mstar)
    else:
        a_au = st.number_input("Semi-major axis (AU)", min_value=0.01, value=0.1, help="Average distance from star.")

    e = st.slider("Eccentricity (0=circle)", 0.0, 0.9, 0.0, 0.01)
    inc = st.slider("Tilt / inclination (degrees)", 0.0, 90.0, 30.0, 1.0)

    x, y, z = ellipse_xyz(a_au, e=e, inc_deg=inc)
    star = go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", marker=dict(size=5), name="Star")
    orbit = go.Scatter3d(x=x, y=y, z=z, mode="lines", name="Planet path")

    fig3 = go.Figure(data=[star, orbit])
    fig3.update_layout(
        title=f"Semi-major axis ≈ {a_au:.3f} AU",
        scene=dict(xaxis_title="AU", yaxis_title="AU", zaxis_title="AU", aspectmode="data")
    )
    st.plotly_chart(fig3, use_container_width=True)