"""
Preprocessing helpers compatible with CLI tests.
Handles both Python lists and pandas DataFrames where appropriate.
"""

import pandas as pd
import numpy as np
import re


# -----------------------------
# CLEANING FUNCTIONS
# -----------------------------
def remove_missing(data):
    """Remove missing values (None or empty strings) from list or DataFrame column."""
    if isinstance(data, list):
        return [x for x in data if x not in (None, "")]
    elif isinstance(data, pd.DataFrame):
        return data.dropna()
    else:
        raise TypeError("Unsupported type for remove_missing")


def fill_missing(data, value=0):
    """Fill missing values in list or DataFrame column."""
    if isinstance(data, list):
        return [x if x not in (None, "") else value for x in data]
    elif isinstance(data, pd.DataFrame):
        return data.fillna(value)
    else:
        raise TypeError("Unsupported type for fill_missing")


def remove_duplicates(data):
    """Remove duplicate values from list or DataFrame."""
    if isinstance(data, list):
        return list(dict.fromkeys(data))
    elif isinstance(data, pd.DataFrame):
        return data.drop_duplicates()
    else:
        raise TypeError("Unsupported type for remove_duplicates")


# -----------------------------
# NUMERIC FUNCTIONS
# -----------------------------
def normalize(data, new_min=0.0, new_max=1.0):
    """Normalize numeric list or DataFrame column to [new_min, new_max]."""
    arr = np.array(data, dtype=float)
    min_val, max_val = arr.min(), arr.max()
    if max_val - min_val == 0:
        return [new_min for _ in arr]
    normalized = (arr - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
    return normalized.tolist()


def standardize(data):
    """Standardize numeric list or DataFrame column (mean=0, std=1)."""
    arr = np.array(data, dtype=float)
    mean = arr.mean()
    std = arr.std() if arr.std() != 0 else 1.0
    standardized = (arr - mean) / std
    return standardized.tolist()


def clip(data, min_value=0.0, max_value=1.0):
    """Clip numeric list or DataFrame column between min_value and max_value."""
    arr = np.array(data, dtype=float)
    clipped = np.clip(arr, min_value, max_value)
    return clipped.tolist()


def convert_to_int(data):
    """Convert list or DataFrame column to integers, ignoring non-convertible values."""
    result = []
    for x in data:
        try:
            result.append(int(x))
        except (ValueError, TypeError):
            continue
    return result


def log_transform(data):
    """Apply natural log to positive numbers, ignore negatives."""
    return [np.log(x) for x in data if x > 0]



# -----------------------------
# TEXT FUNCTIONS
# -----------------------------
def tokenize_text(text):
    """Split text into lowercase words, remove punctuation."""
    return re.findall(r"\b\w+\b", text.lower())


def remove_non_alnum_spaces(text):
    """Remove characters that are not alphanumeric or spaces."""
    return re.sub(r"[^0-9a-zA-Z ]+", "", text)


def remove_stopwords(text, stopwords):
    """Remove given stopwords from text."""
    tokens = tokenize_text(text)
    filtered = [t for t in tokens if t not in stopwords]
    return " ".join(filtered)


# -----------------------------
# STRUCTURE FUNCTIONS
# -----------------------------
def flatten_list(lst):
    """Flatten nested list."""
    result = []
    for el in lst:
        if isinstance(el, list):
            result.extend(flatten_list(el))
        else:
            result.append(el)
    return result


def shuffle_list(lst, seed=None):
    """Shuffle a list with optional seed."""
    import random

    rng = random.Random(seed)
    shuffled = lst[:]
    rng.shuffle(shuffled)
    return shuffled
