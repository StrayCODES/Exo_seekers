# 🪐 ExoAI Explorer

An interactive Streamlit web app that combines **NASA exoplanet data**, **machine learning**, and **visual exploration tools** to help users analyze, predict, and visualize exoplanetary systems.

Built to be **deployed in a day**, ExoAI Explorer pulls real data from the **NASA Exoplanet Archive (KOI cumulative table via TAP)**, trains a classifier to distinguish confirmed planets from false positives, and provides tools for **transit detection** and **3D orbital visualization** — all inside a single Streamlit interface.

---

## 🚀 Features

* **Real NASA Data Integration**

  * Uses the [Table Access Protocol (TAP)](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html) to fetch the latest **Kepler Object of Interest (KOI)** cumulative dataset.
* **Machine Learning Classifier**

  * Trains a lightweight Hist Gradient Boosting Classifier model to classify KOI candidates as *CONFIRMED-like* or *FALSE POSITIVE-like*.
* **Transit Search (BLS)**

  * Implements the **Box Least Squares** algorithm (via [Astropy](https://docs.astropy.org/en/latest/api/astropy.timeseries.BoxLeastSquares.html)) to identify periodic dips in uploaded light curves.
* **3D Orbital Visualization**

  * Renders circular or elliptical orbits in 3D using **Plotly**, based on Kepler’s Third Law.
* **Streamlit UI**

  * Intuitive web interface with tabs for **Prediction**, **Transit Search**, and **3D Orbit View**.
* **Fast Deployment**

  * Fully deployable to **Streamlit Community Cloud** via GitHub with a single click.

---

## 🧠 How It Works

### 1. Data Source

Data comes from NASA’s **Exoplanet Archive – KOI cumulative table** accessed through the official TAP interface.

Example TAP query:

```sql
SELECT koi_disposition, koi_period, koi_prad, koi_model_snr,
       koi_depth, koi_duration, koi_steff, koi_slogg, koi_srad
FROM cumulative
```

### 2. Model Training

* Labels:

  * `CONFIRMED → 1`
  * `FALSE POSITIVE → 0`
* Features:

  * `koi_period, koi_prad, koi_model_snr, koi_depth, koi_duration, koi_steff, koi_slogg, koi_srad`
* Algorithm: Logistic Regression (with scaling + imputation)
* Output: `model/model.pkl` and `feature_columns.json`

### 3. Transit Detection

Uses **Astropy’s BoxLeastSquares** to search uploaded light curves for periodic transits and display both raw and phase-folded plots.

### 4. Orbit Visualization

Computes orbital paths in 3D using Kepler’s Third Law:
[
a = \left(\frac{G M_\star P^2}{4\pi^2}\right)^{1/3}
]
with adjustable eccentricity and inclination sliders.

---

## 🗂️ Project Structure

```
exoai-explorer/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data/
│   └── sample_lightcurve.csv
│
├── model/
│   ├── model.pkl
│   └── feature_columns.json
│
└── utils/
    ├── __init__.py
    ├── nasa.py
    ├── transit.py
    └── orbit.py
```

---

## ⚙️ Installation & Setup

### Prerequisites

* Python 3.10+
* Git & VS Code (recommended)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/exoai-explorer.git
cd exoai-explorer
```

### 2. Create and activate virtual environment

**Windows PowerShell:**

```powershell
py -3.10 -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_model.py
```

### 5. Run the app locally

```bash
streamlit run app.py
```

Then open the local URL (default: [http://localhost:8501](http://localhost:8501)).

---

## ☁️ Deploying on Streamlit Community Cloud

1. Push this project to a public GitHub repository.
2. Go to [Streamlit Cloud](https://share.streamlit.io).
3. Click **“New app”** → connect your GitHub → select `app.py`.
4. Ensure `requirements.txt` exists in the root directory.
5. Deploy!

You can set a custom subdomain and badge in Streamlit’s settings.

---

## 🧩 Example Usage

### 1. Predict KOI Outcome

Enter feature values in the *Explore & Predict* tab → click **Predict** → see probability of being a confirmed planet.

### 2. Transit Search

Upload a CSV with columns `time, flux` (or load the sample) → app runs BLS and shows phase-folded light curve + detected parameters.

### 3. Orbit Visualization

Enter orbital period and star mass → interactive 3D orbit appears with inclination and eccentricity controls.

---

## 📦 Dependencies

| Package           | Purpose               |
| ----------------- | --------------------- |
| `streamlit`       | Web app framework     |
| `pandas`, `numpy` | Data handling         |
| `scikit-learn`    | ML model              |
| `requests`        | NASA TAP API          |
| `plotly`          | 3D visualization      |
| `astropy`         | BLS transit detection |

---

## 🧩 Future Improvements

* Add a “Known Planets” explorer using NASA’s `ps` table.
* Cache NASA queries using `st.cache_data`.
* Support Kepler, TESS, and custom missions for broader data coverage.
* Integrate RandomForest or XGBoost models for better classification.

---

## 📖 References & Acknowledgements

* [NASA Exoplanet Archive – TAP Interface](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html)
* [Kepler Object of Interest (KOI) Table Columns](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html)
* [Astropy BoxLeastSquares Documentation](https://docs.astropy.org/en/latest/api/astropy.timeseries.BoxLeastSquares.html)
* [Streamlit Community Cloud Deployment Guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)
* [Kepler’s Laws of Planetary Motion – NASA](https://science.nasa.gov/kepler/mission/)

---

## 🧑‍🚀 Author

**Mahir Ghotia**
📍 Vancouver, BC, Canada
🔗 [LinkedIn](https://linkedin.com/in/mahirghotia) | [GitHub](https://github.com/mahirghotia)

---

## 🪐 License

This project uses publicly available NASA open data.
Code is open-source under the **MIT License**.

---

> “Exploring the universe starts with the right data — ExoAI makes it interactive.”
