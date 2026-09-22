#!/usr/bin/env python3
"""Feature engineering functions for Ames Housing dataset.

Each function takes a DataFrame, modifies it in place, and returns it.
Run these in order: add_numeric_features → encode_ordinal_features → encode_nominal_features.

Usage in notebook:
    from src.features import *

    df = add_numeric_features(df)
    df = encode_ordinal_features(df)
    df = encode_nominal_features(df, nominal_cols)
"""

import pandas as pd
import numpy as np


# ── 1. New numeric features ──────────────────────────────────────────────────

def add_numeric_features(df):
    """Add derived numeric features to the DataFrame.

    Creates:
        TotalSF         — TotalBsmtSF + 1stFlrSF + 2ndFlrSF
        Age             — 2025 - YearBuilt
        RemodAge        — 2025 - YearRemodAdd
        TotalBath       — FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath
        TotalPorchSF    — OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch
        IsNew           — (YearBuilt == YrSold).astype(int)
        HasPool         — (PoolArea > 0).astype(int)
        HasMasonryVeneer — (MasVnrType != 'None').astype(int)

    Modifies df in place, returns df.
    """
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Age'] = 2025 - df['YearBuilt']
    df['RemodAge'] = 2025 - df['YearRemodAdd']
    df['TotalBath'] = (df['FullBath'] +
                       0.5 * df['HalfBath'] +
                       df['BsmtFullBath'] +
                       0.5 * df['BsmtHalfBath'])
    df['TotalPorchSF'] = (df['OpenPorchSF'] +
                           df['EnclosedPorch'] +
                           df['3SsnPorch'] +
                           df['ScreenPorch'])
    df['IsNew'] = (df['YearBuilt'] == df['YrSold']).astype(int)
    df['HasPool'] = (df['PoolArea'] > 0).astype(int)
    df['HasMasonryVeneer'] = (df['MasVnrType'] != 'None').astype(int)
    return df


# ── 2. Ordinal encoding ──────────────────────────────────────────────────────

ORDINAL_MAPPINGS = {
    'ExterQual':    {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'ExterCond':    {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'BsmtQual':     {'NoBasement': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtCond':     {'NoBasement': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'HeatingQC':    {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'KitchenQual':  {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
    'GarageQual':   {'NoGarage': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageCond':   {'NoGarage': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
}


def encode_ordinal_features(df, mappings=None):
    """Encode ordinal categorical columns as integers using predefined mappings.

    Columns encoded: ExterQual, ExterCond, BsmtQual, BsmtCond,
                     HeatingQC, KitchenQual, GarageQual, GarageCond

    Modifies df in place, returns df.
    """
    if mappings is None:
        mappings = ORDINAL_MAPPINGS

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    return df


# ── 3. Nominal encoding (one-hot) ───────────────────────────────────────────

DEFAULT_NOMINAL_COLS = [
    'MSZoning', 'Neighborhood', 'BldgType', 'HouseStyle',
    'RoofStyle', 'Foundation', 'CentralAir', 'PavedDrive',
    'SaleType', 'SaleCondition', 'Electrical', 'Functional',
    'Exterior1st', 'Exterior2nd', 'MasVnrType',
]


def encode_nominal_features(df, nominal_cols=None, drop_first=False):
    """One-hot encode nominal categorical columns.

    Args:
        df: DataFrame
        nominal_cols: list of column names to encode.
                      Defaults to DEFAULT_NOMINAL_COLS (filtered to columns that exist).
        drop_first: whether to drop the first dummy column (avoid dummy variable trap).
                    Default False — keep all dummies for tree models.

    Modifies df in place, returns df.
    """
    if nominal_cols is None:
        nominal_cols = DEFAULT_NOMINAL_COLS

    # Filter to columns that exist in the DataFrame
    nominal_cols = [c for c in nominal_cols if c in df.columns]

    if not nominal_cols:
        return df

    df = pd.get_dummies(df, columns=nominal_cols, drop_first=drop_first, dtype=int)

    return df


# ── Utility ──────────────────────────────────────────────────────────────────

def get_all_features(df):
    """Return list of all feature column names (everything except targets)."""
    target_cols = {'SalePrice', 'LogSalePrice'}
    return [c for c in df.columns if c not in target_cols]
