
import requests
import pandas as pd
import io  

TAP_BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

def fetch_koi_dataframe():
    cols = ",".join([
        "koi_disposition","koi_period","koi_prad","koi_model_snr",
        "koi_depth","koi_duration","koi_steff","koi_slogg","koi_srad"
    ])
    query = f"select+{cols}+from+cumulative"
    url = f"{TAP_BASE}?query={query}&format=csv"

    print("Fetching KOI data from:", url)
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    df = pd.read_csv(io.StringIO(r.text))
    return df
