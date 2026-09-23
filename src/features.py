import pandas as pd
import numpy as np


# ── 1. New numeric features ──────────────────────────────────────────────────

def add_numeric_features(df):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    df['Age'] = 2026 - df['YearBuilt']
    
    df['RemodAge'] = 2026 - df['YearRemodAdd']
    
    df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
    
    df['TotalPorchSF'] = (df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch'])
    
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
    if nominal_cols is None:
        nominal_cols = DEFAULT_NOMINAL_COLS

    # Filter to columns that exist in the DataFrame
    nominal_cols = [c for c in nominal_cols if c in df.columns]

    if not nominal_cols:
        return df

    df = pd.get_dummies(df, columns=nominal_cols, drop_first=drop_first, dtype=int)

    return df



def handle_outliers(df, grlivarea_threshold=3500, price_threshold=300000, quality_threshold=4, id_col='Id'):
    if id_col not in df.columns:
        return df
    
    outliers = df[(df['GrLivArea'] > grlivarea_threshold) & (df['SalePrice'] < price_threshold)]
    
    to_delete = outliers[outliers['OverallQual'] <= quality_threshold]
    
    if len(to_delete) > 0:
        df = df[~df[id_col].isin(to_delete[id_col])]
        print(f"Removed {len(to_delete)} outlier houses (IDs: {to_delete[id_col].tolist()})")
    
    return df


# ── Utility ──────────────────────────────────────────────────────────────────

def get_all_features(df):
    target_cols = {'SalePrice', 'LogSalePrice'}
    return [c for c in df.columns if c not in target_cols]
