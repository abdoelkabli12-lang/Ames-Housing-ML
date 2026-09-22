import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CleanData:
  def __init__(self):
    self.data = pd.read_csv('C:/Users/ycode/Documents/Ames-Housing-ML/data/House_Prices.csv')
    self.clean()
    
  def clean(self):
    df = self.data.drop(['Id',
        'PoolQC',
        'MiscFeature',
        'Alley',
        'Fence',
        'FireplaceQu',
        'Utilities',
        'Condition2',
        'RoofMatl',
        'LandSlope',
        'LotConfig',
        'MiscVal',
        'LowQualFinSF',
        'BsmtFinSF2',
        'KitchenAbvGr',
        ], axis=1)
    
    # Duplicates
    dup = df.duplicated().sum()
    if dup > 0:
        df = df.drop_duplicates()
        print(f'Dropped {dup} duplicates')
    else:
        print('No duplicates')
    
    # Garage group
    df['HasGarage'] = df['GarageType'].notna().astype(int)
    df['GarageType'] = df['GarageType'].fillna('NoGarage')
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    df['GarageFinish'] = df['GarageFinish'].fillna('NoGarage')
    df['GarageQual'] = df['GarageQual'].fillna('NoGarage')
    df['GarageCond'] = df['GarageCond'].fillna('NoGarage')
    
    # Basement group
    df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['BsmtQual'] = df['BsmtQual'].fillna('NoBasement')
    df['BsmtCond'] = df['BsmtCond'].fillna('NoBasement')
    df['BsmtExposure'] = df['BsmtExposure'].fillna('NoBasement')
    df['BsmtFinType1'] = df['BsmtFinType1'].fillna('NoBasement')
    df['BsmtFinType2'] = df['BsmtFinType2'].fillna('NoBasement')
    df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(0)
    df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(0)
    df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)
    df['BsmtFullBath'] = df['BsmtFullBath'].fillna(0)
    
    # Masonry veneer
    df['MasVnrType'] = df['MasVnrType'].fillna('None')
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    # Fireplace (FireplaceQu already dropped)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    
    # LotFrontage — by neighborhood
    df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    
    # Electrical, Functional — mode
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    df['Functional'] = df['Functional'].fillna(df['Functional'].mode()[0])
    
    self.clean_df = df
    return df
  

    

x = CleanData()
x.clean()




