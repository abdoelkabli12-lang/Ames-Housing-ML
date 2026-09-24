import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

import joblib


from preprocess import CleanData
from features import add_numeric_features, encode_ordinal_features, encode_nominal_features
from analysis import analyze_target, render_df


cleaner = CleanData()
dat = analyze_target(cleaner.clean_df)
df = render_df(dat)


df = add_numeric_features(df)
df = encode_ordinal_features(df)


nominal_cols = [
    'MSZoning', 'Neighborhood', 'BldgType', 'HouseStyle',
    'RoofStyle', 'Foundation', 'CentralAir', 'PavedDrive',
    'SaleType', 'SaleCondition', 'Electrical', 'Functional',
    'Exterior1st', 'Exterior2nd', 'MasVnrType'
]
nominal_cols = [c for c in nominal_cols if c in df.columns]
df = encode_nominal_features(df, nominal_cols)

exclude_cols = {'SalePrice', 'LogSalePrice'}
X = df.select_dtypes(include=[np.number]).drop(columns=exclude_cols, errors='ignore')

y_raw = df['SalePrice']
y_log = df['LogSalePrice']


X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, y_raw, test_size=0.2, random_state=42
)

y_train_log = y_log.loc[y_train_raw.index]
y_test_log = y_log.loc[y_test_raw.index]


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, list(X.columns)),
    ],
    remainder='drop'
)

lr_pipeline = Pipeline(steps = [('preprocessor', preprocessor),
                       ('regression', LinearRegression())])

print('Training Linear Regression on LogSalePrice...')
lr_pipeline.fit(X_train, y_train_log)

y_pred_lr_log = lr_pipeline.predict(X_test)

y_pred_lr = np.expm1(y_pred_lr_log)

lr_mae = mean_absolute_error(y_test_raw, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_lr))
lr_r2 = r2_score(y_test_raw, y_pred_lr)

print('\n=== Linear Regression — Test Set Results ===')
print(f'MAE:  ${lr_mae:>10,.0f}')
print(f'RMSE: ${lr_rmse:>10,.0f}')
print(f'R²:   {lr_r2:>10.4f}')




lr_results = {'Model': 'Linear Regression', 'MAE': lr_mae, 'RMSE': lr_rmse, 'R²': lr_r2}



rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

# Train on raw SalePrice
print('Training Random Forest on SalePrice...')
rf_pipeline.fit(X_train, y_train_raw)

# Predict
y_pred_rf = rf_pipeline.predict(X_test)

# Evaluate
rf_mae = mean_absolute_error(y_test_raw, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_rf))
rf_r2 = r2_score(y_test_raw, y_pred_rf)

print('\n=== Random Forest — Test Set Results ===')
print(f'MAE:  ${rf_mae:>10,.0f}')
print(f'RMSE: ${rf_rmse:>10,.0f}')
print(f'R²:   {rf_r2:>10.4f}')

# Store for comparison
rf_results = {'Model': 'Random Forest', 'MAE': rf_mae, 'RMSE': rf_rmse, 'R²': rf_r2}





# Build the full pipeline: preprocessor + XGBoost
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ))
])

# Train on raw SalePrice
print('Training XGBoost on SalePrice...')
xgb_pipeline.fit(X_train, y_train_raw)

# Predict
y_pred_xgb = xgb_pipeline.predict(X_test)

# Evaluate
xgb_mae = mean_absolute_error(y_test_raw, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_xgb))
xgb_r2 = r2_score(y_test_raw, y_pred_xgb)

print('\n=== XGBoost — Test Set Results ===')
print(f'MAE:  ${xgb_mae:>10,.0f}')
print(f'RMSE: ${xgb_rmse:>10,.0f}')
print(f'R²:   {xgb_r2:>10.4f}')

# Store for comparison
xgb_results = {'Model': 'XGBoost', 'MAE': xgb_mae, 'RMSE': xgb_rmse, 'R²': xgb_r2}