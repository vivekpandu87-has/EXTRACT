import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Column categories
# ─────────────────────────────────────────────────────────────────────────────
PIPE_COLS = ["Categories", "Stress_Purchases", "Shopping_Situations", "Product_Combinations"]
DROP_FOR_MODEL = ["Happy_Purchases"]   # pipe-separated but not in PIPE_COLS → drop

INCOME_ORDER    = ["<20k", "20k-50k", "50k-1L", ">1L"]
AGE_ORDER       = ["Under 18", "18-24", "25-34", "35-44", "45+"]
FREQ_ORDER      = ["Rarely", "Monthly", "Weekly", "Daily"]
LAST_BUY_ORDER  = [">1 Month", "This Month", "This Week", "Today"]

MOOD_COLORS = {
    "Happy":   "#4CAF50",
    "Sad":     "#5C6BC0",
    "Bored":   "#FF9800",
    "Anxious": "#F44336",
    "Excited": "#E91E63",
    "Neutral": "#90A4AE",
    "Angry":   "#B71C1C",
    "Calm":    "#26C6DA",
}

INTEREST_COLORS = {
    "Yes":   "#A855F7",
    "No":    "#F44336",
    "Maybe": "#FF9800",
}

PRIMARY   = "#A855F7"
SECONDARY = "#7C3AED"
ACCENT    = "#EC4899"


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df["Monthly_Spend"] = pd.to_numeric(df["Monthly_Spend"], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# One-hot encode pipe-separated multi-select columns
# ─────────────────────────────────────────────────────────────────────────────
def one_hot_encode_multiselect(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    s = df[column].fillna("")
    items: set = set()
    for x in s:
        for it in str(x).split("|"):
            it = it.strip()
            if it:
                items.add(it)
    for it in sorted(items):
        df[f"{column}__{it}"] = s.apply(
            lambda x: 1 if it in [t.strip() for t in str(x).split("|")] else 0
        )
    return df.drop(columns=[column])


# ─────────────────────────────────────────────────────────────────────────────
# Full preprocessing pipeline
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in PIPE_COLS:
        if col in df.columns:
            df = one_hot_encode_multiselect(df, col)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Encode for ML  (returns X, y)
# ─────────────────────────────────────────────────────────────────────────────
def encode_for_model(df: pd.DataFrame, target_col=None):
    df = df.copy()
    y = None
    if target_col and target_col in df.columns:
        y = df[target_col]
        df = df.drop(columns=[target_col])
    df = df.drop(columns=[c for c in DROP_FOR_MODEL if c in df.columns])
    X = pd.get_dummies(df, drop_first=True)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Price Sensitivity Meter (PSM) helpers
# ─────────────────────────────────────────────────────────────────────────────
PSM_MIDPOINTS = {
    "<₹200":       175,
    "₹200-500":    350,
    "₹500-1000":   750,
    "₹1000-2000": 1500,
    "₹2000-3500": 2750,
    ">₹3500":     4000,
}

def psm_midpoint(val):
    """Convert a PSM bucket string to its numeric midpoint."""
    val = str(val).strip()
    return PSM_MIDPOINTS.get(val, np.nan)


def compute_psm_curves(df: pd.DataFrame):
    """
    Returns a dict of {label: pd.Series(index=prices, values=cum_%)} for
    Too Cheap / Cheap / Expensive / Too Expensive,
    plus acceptable range (low, high) and optimal price.
    """
    psm_cols = {
        "Too Cheap":     "PSM_ToCheap",
        "Cheap":         "PSM_Cheap",
        "Expensive":     "PSM_Expensive",
        "Too Expensive": "PSM_TooExpensive",
    }
    prices = sorted(PSM_MIDPOINTS.values())
    curves = {}
    for label, col in psm_cols.items():
        if col not in df.columns:
            continue
        midpoints = df[col].map(PSM_MIDPOINTS)
        total = len(midpoints.dropna())
        if total == 0:
            continue
        if label in ("Too Cheap", "Cheap"):
            cum = [(midpoints <= p).sum() / total * 100 for p in prices]
        else:
            cum = [(midpoints >= p).sum() / total * 100 for p in prices]
        curves[label] = pd.Series(cum, index=prices)
    return curves


def psm_acceptable_range(curves):
    """Intersect Too Cheap ↓ and Too Expensive ↓ for acceptable range."""
    if "Too Cheap" not in curves or "Too Expensive" not in curves:
        return None, None
    prices = list(curves["Too Cheap"].index)
    tc = curves["Too Cheap"].values
    te = curves["Too Expensive"].values
    # acceptable low: where Too Cheap drops below 50 %
    low = None
    for p, v in zip(prices, tc):
        if v <= 50:
            low = p
            break
    # acceptable high: where Too Expensive drops below 50 %
    high = None
    for p, v in zip(reversed(prices), reversed(te)):
        if v <= 50:
            high = p
            break
    return low, high


# ─────────────────────────────────────────────────────────────────────────────
# Segment helper
# ─────────────────────────────────────────────────────────────────────────────
def build_segment_profile(df: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    """Group-level summary for a categorical column used as 'segment'."""
    df2 = df.copy()
    df2["Monthly_Spend"] = pd.to_numeric(df2["Monthly_Spend"], errors="coerce")
    agg = (
        df2.groupby(segment_col)
        .agg(
            Count=("Monthly_Spend", "count"),
            Avg_Spend=("Monthly_Spend", "mean"),
            Median_Spend=("Monthly_Spend", "median"),
        )
        .round(0)
        .reset_index()
    )
    agg["Share_%"] = (agg["Count"] / agg["Count"].sum() * 100).round(1)
    return agg
