import pandas as pd
import numpy as np
from preprocess import CleanData as cl



class Fetures:
  def add_numeric_features(df):
      df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
      df['Age'] = 2025 - df['YearBuilt']
      df['RemodAge'] = 2025 - df['YearRemodAdd']
      df['TotalBath'] = (df['FullBath'] + 0.5*df['HalfBath'] +
                        df['BsmtFullBath'] + 0.5*df['BsmtHalfBath'])
      df['TotalPorchSF'] = (df['OpenPorchSF'] + df['EnclosedPorch'] +
                            df['3SsnPorch'] + df['ScreenPorch'])
      df['IsNew'] = (df['YearBuilt'] == df['YrSold']).astype(int)
      df['HasPool'] = (df['PoolArea'] > 0).astype(int)
      df['HasMasonryVeneer'] = (df['MasVnrType'] != 'None').astype(int)
      return df


  def encode_ordinal_features(df):
      mappings = {
          'ExterQual': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
          'ExterCond': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
          'BsmtQual': {'NoBasement': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
          'BsmtCond': {'NoBasement': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
          'HeatingQC': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
          'KitchenQual': {'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},
          'GarageQual': {'NoGarage': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
          'GarageCond': {'NoGarage': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
      }
      for col, mapping in mappings.items():
          if col in df.columns:
              df[col] = df[col].map(mapping)
      return df


  def encode_nominal_features(df, nominal_cols=None):
      if nominal_cols is None:
          nominal_cols = [
              'MSZoning', 'Neighborhood', 'BldgType', 'HouseStyle',
              'RoofStyle', 'Foundation', 'CentralAir', 'PavedDrive',
              'SaleType', 'SaleCondition', 'Electrical', 'Functional',
              'Exterior1st', 'Exterior2nd', 'MasVnrType',
          ]
      nominal_cols = [c for c in nominal_cols if c in df.columns]
      df = pd.get_dummies(df, columns=nominal_cols, drop_first=False)
      return df
x = cl().clean()
print(Fetures.add_numeric_features(x))