"""
Simple preprocessing helpers required for the CLI tests.
These are intentionally minimal — only what the tests expect.
"""

import pandas as pd
import numpy as np
import re
import random


# ----------------------
# DataFrame Helpers
# ----------------------
def remove_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna()


def fill_missing(df: pd.DataFrame, value=0) -> pd.DataFrame:
    return df.fillna(value)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def normalize(df: pd.DataFrame, column=None, new_min=0.0, new_max=1.0) -> pd.DataFrame:
    if column:
        min_val = df[column].min()
        max_val = df[column].max()
        df[column] = (df[column] - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
    else:
        df = (df - df.min()) / (df.max() - df.min()) * (new_max - new_min) + new_min
    return df


def standardize(df: pd.DataFrame, column=None) -> pd.DataFrame:
    if column:
        mean = df[column].mean()
        std = df[column].std() or 1.0
        df[column] = (df[column] - mean) / std
    else:
        df = (df - df.mean()) / df.std().replace(0, 1.0)
    return df


def clip(df: pd.DataFrame, column=None, min_value=0.0, max_value=1.0) -> pd.DataFrame:
    if column:
        df[column] = df[column].clip(min_value, max_value)
    else:
        df = df.clip(min_value, max_value)
    return df


def convert_to_int(df: pd.DataFrame, column=None) -> pd.DataFrame:
    if column:
        df[column] = df[column].astype(int)
    else:
        df = df.astype(int)
    return df


def log_transform(df: pd.DataFrame, column=None) -> pd.DataFrame:
    if column:
        df[column] = np.log1p(df[column])
    else:
        df = np.log1p(df)
    return df


def encode_categorical(df: pd.DataFrame, column: str) -> pd.DataFrame:
    encoded = pd.get_dummies(df[column], prefix=column)
    df = df.drop(columns=[column])
    df = pd.concat([df, encoded], axis=1)
    return df


# ----------------------
# Text Helpers
# ----------------------
def tokenize_text(text: str):
    return text.lower().split()


def remove_non_alnum_spaces(text: str):
    return re.sub(r"[^0-9a-zA-Z\s]+", "", text)


def remove_stopwords(text: str, stopwords: list):
    tokens = tokenize_text(text)
    return [t for t in tokens if t not in stopwords]


# ----------------------
# List Helpers
# ----------------------
def flatten_list(lst):
    result = []
    for el in lst:
        if isinstance(el, (list, tuple)):
            result.extend(flatten_list(el))
        else:
            result.append(el)
    return result


def shuffle_list(lst, seed=None):
    rng = random.Random(seed)
    lst_copy = lst.copy()
    rng.shuffle(lst_copy)
    return lst_copy
