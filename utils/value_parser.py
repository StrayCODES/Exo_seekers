
import re
from typing import Any, Dict, Optional, Tuple, Union

Number = Union[int, float]

# Pre-compiled regexes for speed/readability
RE_SCI = re.compile(r"^[\s]*[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?[\s]*$")
RE_PM  = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*±\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$")
RE_ASYM = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\+\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*-\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)

# Also allow forms like "0.146 +0.318-0.146" (no spaces before last minus)
RE_ASYM_TIGHT = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\+\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)-\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)

BOOL_TRUE = {"1","true","t","yes","y","on"}
BOOL_FALSE = {"0","false","f","no","n","off"}

def _clean_units(s: str) -> str:
    """Strip common unit clutter (ppm, K, mag) and extra spaces. Keep numbers & delimiters."""
    # Remove trailing unit words if present (very light touch)
    s = s.strip()
    # If value is something like "615.8±19.5", leave it; if "615.8 K", remove "K"
    # Only remove a single trailing unit token.
    s = re.sub(r"\s*(ppm|K|mag|deg|days|hrs|hours|AU)\s*$", "", s, flags=re.IGNORECASE)
    return s.strip()

def parse_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in BOOL_TRUE:
        return True
    if s in BOOL_FALSE:
        return False
    # Sometimes catalogs have 0/1 without quotes
    if RE_SCI.match(s):
        try:
            return float(s) != 0.0
        except Exception:
            return None
    return None

def parse_number_with_uncertainty(
    x: Any,
    strategy: str = "central"
) -> Tuple[Optional[Number], Dict[str, Optional[Number]]]:
    """
    Returns (central_value, meta) where meta may contain:
      - 'sigma' for symmetric ±
      - 'err_plus' and 'err_minus' for asymmetric +up/-down
      - 'raw' original string
    strategy is currently informational; we always return the central value,
    but you could switch to 'upper'/'lower' if desired.
    """
    if x is None:
        return None, {"raw": None, "sigma": None, "err_plus": None, "err_minus": None}

    if isinstance(x, (int, float)):
        return float(x), {"raw": x, "sigma": None, "err_plus": None, "err_minus": None}

    s = _clean_units(str(x))

    # Symmetric: "v ± s"
    m = RE_PM.match(s)
    if m:
        v = float(m.group(1)); sig = float(m.group(2))
        return v, {"raw": s, "sigma": sig, "err_plus": None, "err_minus": None}

    # Asymmetric: "v +up -down"
    m = RE_ASYM.match(s) or RE_ASYM_TIGHT.match(s)
    if m:
        v = float(m.group(1)); up = float(m.group(2)); down = float(m.group(3))
        return v, {"raw": s, "sigma": None, "err_plus": up, "err_minus": down}

    # Plain number (int/float/sci)
    if RE_SCI.match(s):
        try:
            return float(s), {"raw": s, "sigma": None, "err_plus": None, "err_minus": None}
        except Exception:
            pass

    # As a last resort, strip any non-numeric trailing junk and try again
    s2 = re.sub(r"[^\d\.\-\+\seE±]", "", s)
    if s2 != s and RE_SCI.match(s2):
        try:
            return float(s2), {"raw": s, "sigma": None, "err_plus": None, "err_minus": None}
        except Exception:
            pass

    # Could not parse
    return None, {"raw": s, "sigma": None, "err_plus": None, "err_minus": None}
