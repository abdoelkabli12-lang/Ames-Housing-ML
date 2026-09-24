import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os



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
                       ('regressor', LinearRegression())])

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

print('Training Random Forest on SalePrice...')
rf_pipeline.fit(X_train, y_train_raw)

y_pred_rf = rf_pipeline.predict(X_test)

rf_mae = mean_absolute_error(y_test_raw, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_rf))
rf_r2 = r2_score(y_test_raw, y_pred_rf)

print('\n=== Random Forest — Test Set Results ===')
print(f'MAE:  ${rf_mae:>10,.0f}')
print(f'RMSE: ${rf_rmse:>10,.0f}')
print(f'R²:   {rf_r2:>10.4f}')

rf_results = {'Model': 'Random Forest', 'MAE': rf_mae, 'RMSE': rf_rmse, 'R²': rf_r2}





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

print('Training XGBoost on SalePrice...')
xgb_pipeline.fit(X_train, y_train_raw)

y_pred_xgb = xgb_pipeline.predict(X_test)

xgb_mae = mean_absolute_error(y_test_raw, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_xgb))
xgb_r2 = r2_score(y_test_raw, y_pred_xgb)

print('\n=== XGBoost — Test Set Results ===')
print(f'MAE:  ${xgb_mae:>10,.0f}')
print(f'RMSE: ${xgb_rmse:>10,.0f}')
print(f'R²:   {xgb_r2:>10.4f}')

xgb_results = {'Model': 'XGBoost', 'MAE': xgb_mae, 'RMSE': xgb_rmse, 'R²': xgb_r2}


comparison = pd.DataFrame([lr_results, rf_results, xgb_results])
comparison = comparison.set_index('Model')[['MAE', 'RMSE', 'R²']]

print('=== Model Comparison — Test Set ===')
print(comparison.to_string())

best_model_name = comparison['R²'].idxmax()
best_mae = comparison.loc[best_model_name, 'MAE']
best_rmse = comparison.loc[best_model_name, 'RMSE']
best_r2 = comparison.loc[best_model_name, 'R²']

print(f'\n=== Best Model: {best_model_name} ===')
print(f'MAE:  ${best_mae:>10,.0f}')
print(f'RMSE: ${best_rmse:>10,.0f}')
print(f'R²:   {best_r2:>10.4f}')

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

lr_pipeline_cv = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_r2_scores = cross_val_score(lr_pipeline_cv, X_train, y_train_log, cv=kf, scoring='r2')
cv_mae_scores = cross_val_score(lr_pipeline_cv, X_train, y_train_log, cv=kf, scoring='neg_mean_absolute_error')

print(f'CV R² scores per fold: {cv_r2_scores}')
print(f'CV MAE scores per fold (log): {cv_mae_scores}')


print(f'\nMean CV R²: {cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}')
print(f'Mean CV MAE (log): {cv_mae_scores.mean():.4f} ± {cv_mae_scores.std():.4f}')




xgb_pipeline_cv = Pipeline(steps=[
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
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_r2_scores_xgb = cross_val_score(xgb_pipeline_cv, X_train, y_train_raw, cv=kf, scoring='r2')
cv_mae_scores_xgb = cross_val_score(xgb_pipeline_cv, X_train, y_train_raw, cv=kf, scoring='neg_mean_absolute_error')

print(f'CV R² scores per fold: {cv_r2_scores_xgb}')
print(f'CV MAE scores per fold: {cv_mae_scores_xgb}')


print(f'\nMean CV R²: {cv_r2_scores_xgb.mean():.4f} ± {cv_r2_scores_xgb.std():.4f}')
print(f'Mean CV MAE (log): {cv_mae_scores_xgb.mean():.4f} ± {cv_mae_scores_xgb.std():.4f}')


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


kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_r2_scores_xgb = cross_val_score(xgb_pipeline_cv, X_train, y_train_raw, cv=kf, scoring='r2')
cv_mae_scores_xgb = cross_val_score(xgb_pipeline_cv, X_train, y_train_raw, cv=kf, scoring='neg_mean_absolute_error')

print(f'CV R² scores per fold: {cv_r2_scores_xgb}')
print(f'CV MAE scores per fold: {cv_mae_scores_xgb}')


print(f'\nMean CV R²: {cv_r2_scores_xgb.mean():.4f} ± {cv_r2_scores_xgb.std():.4f}')
print(f'Mean CV MAE (log): {cv_mae_scores_xgb.mean():.4f} ± {cv_mae_scores_xgb.std():.4f}')

if best_model_name == 'Linear Regression':
    best_pipeline = lr_pipeline
elif best_model_name == 'Random Forest':
    best_pipeline = rf_pipeline
else:
    best_pipeline = xgb_pipeline
    
    
os.makedirs('models', exist_ok= True)

pipeline_path = 'models/best_pipeline.pkl'
joblib.dump(best_pipeline, pipeline_path)
print(f'Saved pipeline → {pipeline_path}')


model_path = 'models/best_model.pkl'
joblib.dump(best_pipeline.named_steps['regressor'], model_path)
print(f'Saved model → {model_path}')

feature_names_path = 'models/feature_names.json'
import json
with open(feature_names_path, 'w') as f:
    json.dump(list(X.columns), f, indent=2)
print(f'Saved feature names → {feature_names_path}')

target_info = {
    'model_name': best_model_name,
    'target': 'LogSalePrice' if best_model_name == 'Linear Regression' else 'SalePrice',
    'test_mae': float(best_mae),
    'test_rmse': float(best_rmse),
    'test_r2': float(best_r2),
}

target_info_path = 'models/target_info.json'
with open(target_info_path, 'w') as f:
    json.dump(target_info, f, indent=2)
print(f'Saved target info → {target_info_path}')

print(f'\nBest model: {best_model_name}')
print(f'Test MAE:  ${best_mae:,.0f}')
print(f'Test RMSE: ${best_rmse:,.0f}')
print(f'Test R²:   {best_r2:.4f}')

if best_model_name == 'Linear Regression':
    best_preds = y_pred_lr
elif best_model_name == 'Random Forest':
    best_preds = y_pred_rf
else:
    best_preds = y_pred_xgb

errors = best_preds - y_test_raw.values
abs_errors = np.abs(errors)
pct_errors = abs_errors / y_test_raw.values * 100

print(f'=== Error Analysis — Best Model: {best_model_name} ===')
print(f'Mean Absolute Error:     ${abs_errors.mean():>10,.0f}')
print(f'Median Absolute Error:   ${np.median(abs_errors):>10,.0f}')
print(f'Max Absolute Error:      ${abs_errors.max():>10,.0f}')
print(f'Mean Percentage Error:   {pct_errors.mean():>10.1f}%')
print(f'Median Percentage Error: {np.median(pct_errors):>10.1f}%')

error_df = pd.DataFrame({
    'Actual': y_test_raw.values,
    'Predicted': best_preds,
    'Error': errors,
    'AbsError': abs_errors,
    'PctError': pct_errors,
})

bins = [0, 100000, 150000, 200000, 250000, 300000, 500000, 1000000]
labels = ['<100K', '100-150K', '150-200K', '200-250K', '250-300K', '300-500K', '500K+']
error_df['PriceBin'] = pd.cut(error_df['Actual'], bins=bins, labels=labels)

print('\n=== Error by Price Range ===')
error_by_bin = error_df.groupby('PriceBin').agg(
    Count=('Actual', 'count'),
    MAE=('AbsError', 'mean'),
    MedianPctError=('PctError', 'median'),
).round(1)
print(error_by_bin.to_string())

print('\n=== 10 Worst Predictions (largest absolute error) ===')
worst = error_df.nlargest(10, 'AbsError')[['Actual', 'Predicted', 'Error', 'PctError']]
print(worst.to_string())

print('\n=== 10 Best Predictions (lowest absolute error) ===')
best_preds_df = error_df.nsmallest(10, 'AbsError')[['Actual', 'Predicted', 'Error', 'PctError']]
print(best_preds_df.to_string())