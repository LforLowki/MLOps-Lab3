"""
Data preprocessing functions for Lab0.
"""

import math
import random
import re


def remove_missing(values):
    """Removes missing values (None, '', NaN) from the list."""
    return [
        v
        for v in values
        if v not in (None, "", float("nan"))
        and not (isinstance(v, float) and math.isnan(v))
    ]


def fill_missing(values, fill_value=0):
    """Fills missing values with the given value."""
    return [
        fill_value if v in (None, "") or (isinstance(v, float) and math.isnan(v)) else v
        for v in values
    ]


def remove_duplicates(values):
    """Removes duplicate values."""
    return list(dict.fromkeys(values))


def normalize(values, new_min=0.0, new_max=1.0):
    """Normalizes numerical values using min-max scaling."""
    if not values:
        return []
    old_min, old_max = min(values), max(values)
    if old_min == old_max:
        return [new_min for _ in values]
    return [
        ((v - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        for v in values
    ]


def standardize(values):
    """Standardizes numerical values using z-score."""
    if not values:
        return []
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    if std == 0:
        return [0 for _ in values]
    return [(v - mean) / std for v in values]


def clip(values, min_value=0, max_value=1):
    """Clips numerical values to a specified range."""
    return [max(min(v, max_value), min_value) for v in values]


def convert_to_int(values):
    """Converts string values to integers (non-numerical excluded)."""
    result = []
    for v in values:
        try:
            result.append(int(v))
        except (ValueError, TypeError):
            continue
    return result


def log_transform(values):
    """Applies logarithmic transformation to positive values."""
    return [math.log(v) for v in values if isinstance(v, (int, float)) and v > 0]


def tokenize_text(text):
    """Tokenizes text into lowercase alphanumeric words."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def remove_non_alnum_spaces(text):
    """Removes non-alphanumeric characters (except spaces)."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)


def remove_stopwords(text, stopwords):
    """Removes stopwords from text."""
    words = tokenize_text(text)
    return " ".join([w for w in words if w not in stopwords])


def flatten_list(list_of_lists):
    """Flattens a list of lists."""
    return [item for sublist in list_of_lists for item in sublist]


def shuffle_list(values, seed=None):
    """Randomly shuffles a list with an optional seed."""
    random.seed(seed)
    result = values[:]
    random.shuffle(result)
    return result
