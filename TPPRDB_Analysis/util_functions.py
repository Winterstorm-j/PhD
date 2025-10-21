# ...existing code...
"""Utility helpers for the TPPRDB analysis project.

Place this file next to your analysis script (or make the folder a package with an __init__.py)
so you can import it as:
    from util_functions import get_names, _extract_range, preprocess, combine_group_rows
or, if you make the directory a package:
    from TPPRDB_Analysis.util_functions import ...
"""

import ast
import re
import pandas as pd
from typing import Any, List, Optional

__all__ = [
    "get_names",
    "combine_group_rows",
    "preprocess_string_columns",
    "_extract_range",
    "preprocess",
]


def get_names(val: Any, col: str):
    """Parse a string representation of a list/dict and extract values for `col`."""
    if isinstance(val, str):
        try:
            data = ast.literal_eval(val)
            # If it's a list of dicts
            if isinstance(data, list):
                return [d[col] for d in data if isinstance(d, dict) and col in d]
            # If it's a single dict
            if isinstance(data, dict):
                return [data[col]] if col in data else []
            # fallback to raw string
            return val
        except (ValueError, SyntaxError):
            return val
    return val


def combine_group_rows(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Combine rows in a DataFrame by grouping on specified columns and aggregating other columns.
    """
    def combine_series(series):
        unique_vals = series.dropna().unique()
        if len(unique_vals) == 0:
            return pd.NA
        if len(unique_vals) == 1:
            return unique_vals[0]
        return "; ".join([str(val) for val in unique_vals])

    grouped = df.groupby(group_cols, dropna=False)
    combined = grouped.agg(combine_series).reset_index()
    return combined


def preprocess_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Uppercase and strip only object/string columns (in-place) and return the DataFrame."""
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.upper().str.strip().replace({'NAN': pd.NA})
    return df


def _extract_range(x: Any) -> Optional[str]:
    """
    Extract a page/range string from a variety of inputs:
    - If input is a dict-like string, try to literal_eval and get 'range'
    - If it's a list/tuple, join with '-'
    - If it's a plain string, attempt to extract numeric ranges like "12-34" or "12 to 34"
    Returns None when nothing sensible can be extracted.
    """
    if pd.isna(x) or x in ('', 'nan', None):
        return None

    # If x is a string, try to interpret Python literal (e.g. "{'range': ['12','34']}")
    val = x
    if isinstance(x, str):
        s = x.strip()
        try:
            val = ast.literal_eval(s)
        except Exception:
            val = s

    # dict-like: get 'range' key
    if isinstance(val, dict):
        r = val.get('range')
        if r is None:
            return None
        if isinstance(r, (list, tuple)):
            parts = [str(i).strip() for i in r if i not in (None, '')]
            return "-".join(parts) if parts else None
        return str(r).strip() or None

    # list/tuple: join parts
    if isinstance(val, (list, tuple)):
        parts = [str(i).strip() for i in val if i not in (None, '')]
        return "-".join(parts) if parts else None

    # plain string: try to find numeric range
    s = str(val).strip()
    # match "12-34", "12 – 34", "12 to 34"
    m = re.search(r'(\d{1,5})\s*(?:[-–—]|to)\s*(\d{1,5})', s, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    # fallback: return the cleaned string if non-empty
    return s if s else None


def preprocess(text: Any, custom_stopwords: Optional[set] = None) -> str:
    """Minimal text preprocess: lowercase, keep words >=3 chars, remove custom stopwords."""
    if text is None or not isinstance(text, str):
        return ""
    if custom_stopwords is None:
        custom_stopwords = set()
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return ' '.join([w for w in words if w not in custom_stopwords])

def _join_non_na(row):
    vals = (
     row.dropna()
        .astype(str)
        .map(lambda s: s.strip())
    )
    vals = [v for v in vals if v and v.lower() not in ('nan', 'none', 'na')]
    return '; '.join(vals) if vals else pd.NA

