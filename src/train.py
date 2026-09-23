import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# XGBoost
from xgboost import XGBRegressor

# Joblib for saving models
import joblib


# Load cleaned data
# Load cleaned data
import sys
sys.path.insert(0, r'C:\Users\ycode\Documents\Ames-Housing-ML')

from src.preprocess import CleanData
from src.features import add_numeric_features, encode_ordinal_features, encode_nominal_features
from src.analysis import analyze_target

cleaner = CleanData()
dat = analyze_target(cleaner.clean_df)
df = dat

# Apply feature engineering
df = add_numeric_features(df)
df = encode_ordinal_features(df)

# One-hot encode nominal features
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

# Targets
y_raw = df['SalePrice']
y_log = df['LogSalePrice']


X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, y_raw, test_size=0.2, random_state=42
)

# Split log target using the same indices
y_train_log = y_log.loc[y_train_raw.index]
y_test_log = y_log.loc[y_test_raw.index]



numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# ColumnTransformer applies the numeric transformer to all feature columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, list(X.columns)),
    ],
    remainder='drop'
)

print('Pipeline built. Steps:')
print(preprocessor)

# Test the pipeline on the training data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)



# Build the full pipeline: preprocessor + Linear Regression
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train on LogSalePrice
print('Training Linear Regression on LogSalePrice...')
lr_pipeline.fit(X_train, y_train_log)

# Predict on test set (output is in log space)
y_pred_lr_log = lr_pipeline.predict(X_test)

# Convert predictions back to dollars
y_pred_lr = np.expm1(y_pred_lr_log)

# Evaluate on raw SalePrice
lr_mae = mean_absolute_error(y_test_raw, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_lr))
lr_r2 = r2_score(y_test_raw, y_pred_lr)

print('\n=== Linear Regression — Test Set Results ===')
print(f'MAE:  ${lr_mae:>10,.0f}')
print(f'RMSE: ${lr_rmse:>10,.0f}')
print(f'R²:   {lr_r2:>10.4f}')

# Store for comparison
lr_results = {'Model': 'Linear Regression', 'MAE': lr_mae, 'RMSE': lr_rmse, 'R²': lr_r2}

# Show first 10 predictions vs actual
print('\n=== First 10 Predictions vs Actual ===')
pred_df = pd.DataFrame({
    'Actual ($)': y_test_raw.values[:10],
    'Predicted ($)': y_pred_lr[:10],
    'Error ($)': y_pred_lr[:10] - y_test_raw.values[:10],
})
print(pred_df.to_string())


# Build the full pipeline: preprocessor + Random Forest
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

# Feature importance
rf_model = rf_pipeline.named_steps['regressor']
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
print('\n=== Random Forest — Top 15 Feature Importances ===')
print(rf_importances.sort_values(ascending=False).head(15).to_string())




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

# Feature importance
xgb_model = xgb_pipeline.named_steps['regressor']
xgb_importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
print('\n=== XGBoost — Top 15 Feature Importances ===')
print(xgb_importances.sort_values(ascending=False).head(15).to_string())



# Build comparison table
comparison = pd.DataFrame([lr_results, rf_results, xgb_results])
comparison = comparison.set_index('Model')[['MAE', 'RMSE', 'R²']]

print('=== Model Comparison — Test Set ===')
print(comparison.to_string())

# Find the best model by R²
best_model_name = comparison['R²'].idxmax()
best_mae = comparison.loc[best_model_name, 'MAE']
best_rmse = comparison.loc[best_model_name, 'RMSE']
best_r2 = comparison.loc[best_model_name, 'R²']

print(f'\n=== Best Model: {best_model_name} ===')
print(f'MAE:  ${best_mae:>10,.0f}')
print(f'RMSE: ${best_rmse:>10,.0f}')
print(f'R²:   {best_r2:>10.4f}')

# Visual comparison: bar charts
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

colors = ['#3b82f6', '#4ade80', '#f59e0b']
comparison['MAE'].plot(kind='bar', ax=axes[0], color=colors)
axes[0].set_title('MAE (lower is better)')
axes[0].set_ylabel('MAE ($)')
axes[0].tick_params(axis='x', rotation=0)

comparison['RMSE'].plot(kind='bar', ax=axes[1], color=colors)
axes[1].set_title('RMSE (lower is better)')
axes[1].set_ylabel('RMSE ($)')
axes[1].tick_params(axis='x', rotation=0)

comparison['R²'].plot(kind='bar', ax=axes[2], color=colors)
axes[2].set_title('R² (higher is better)')
axes[2].set_ylabel('R²')
axes[2].set_ylim(0, 1)
axes[2].tick_params(axis='x', rotation=0)

for ax in axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.show()

# Actual vs predicted scatter plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

models_preds = [
    ('Linear Regression', y_pred_lr),
    ('Random Forest', y_pred_rf),
    ('XGBoost', y_pred_xgb),
]

for ax, (name, preds) in zip(axes, models_preds):
    ax.scatter(y_test_raw, preds, alpha=0.3, s=10, color='steelblue')
    ax.plot([y_test_raw.min(), y_test_raw.max()],
            [y_test_raw.min(), y_test_raw.max()],
            'r--', alpha=0.7, linewidth=2, label='Perfect')
    ax.set_title(f'{name}\nR² = {comparison.loc[name, "R²"]:.3f}')
    ax.set_xlabel('Actual ($)')
    ax.set_ylabel('Predicted ($)')
    ax.legend()

plt.tight_layout()
plt.show()