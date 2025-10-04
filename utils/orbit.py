# utils/orbit.py
import numpy as np
import pandas as pd

G = 6.67430e-11
M_SUN = 1.98847e30
AU = 1.495978707e11
DAY = 86400.0

def semimajor_axis_au(period_days: float, star_mass_solar: float = 1.0):
    P = period_days * DAY
    M = star_mass_solar * M_SUN
    a_m = (G * M * P**2 / (4 * np.pi**2)) ** (1.0/3.0)
    return a_m / AU

def ellipse_xyz(a, e=0.0, inc_deg=0.0, npts=600):
    # simple Keplerian ellipse in 3D with inclination
    b = a * np.sqrt(1 - e**2)
    theta = np.linspace(0, 2*np.pi, npts)
    x = a*np.cos(theta) - a*e      # focus at origin
    y = b*np.sin(theta)
    z = np.zeros_like(x)
    # rotate by inclination around x-axis
    inc = np.deg2rad(inc_deg)
    y_i = y*np.cos(inc)
    z_i = y*np.sin(inc)
    return x, y_i, z_i
