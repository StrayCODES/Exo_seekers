# utils/transit.py
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares

def bls_search(time, flux):
    # Clean NaNs
    m = np.isfinite(time) & np.isfinite(flux)
    t, y = np.array(time)[m], np.array(flux)[m]

    # Period grid (in days) — quick default
    periods = np.linspace(0.5, 30.0, 5000)
    durations = np.linspace(0.05, 0.3, 10)  # fraction of period

    bls = BoxLeastSquares(t, y)
    res = bls.power(periods, durations)
    i = np.argmax(res.power)

    out = {
        "period": float(res.period[i]),
        "t0": float(res.transit_time[i]),
        "duration": float(res.duration[i]),
        "depth": float(res.depth[i]),
        "snr": float(res.depth_snr[i])
    }

    # Phase-fold for plotting
    phase = ((t - out["t0"] + 0.5*out["period"]) % out["period"]) / out["period"]
    order = np.argsort(phase)
    return out, phase[order], y[order]
