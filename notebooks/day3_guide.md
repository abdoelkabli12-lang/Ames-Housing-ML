# Day 3 — Model Training: Pipeline, 3 Models, Evaluation

Go through these steps in order. Each one builds on the previous — skipping ahead means you'll have to backtrack.

---

## Step 1 — Load and prepare the data

**What you're doing:** Load the cleaned DataFrame from `CleanData` (`src/preprocess.py`), then apply all feature engineering functions from `src/features.py`. The output is one DataFrame with every column you need: cleaned, imputed, with derived numeric features, ordinal-encoded categoricals, and one-hot-encoded nominals.

**The functions to call, in order:**

- `CleanData()` — gives you `clean_df`, the base cleaned DataFrame
- `add_numeric_features(df)` — adds TotalSF, Age, RemodAge, TotalBath, TotalPorchSF, IsNew, HasPool, HasMasonryVeneer
- `encode_ordinal_features(df)` — converts ExterQual, ExterCond, BsmtQual, BsmtCond, HeatingQC, KitchenQual, GarageQual, GarageCond from text labels to integers
- `encode_nominal_features(df, list_of_column_names)` — one-hot encodes the 15 nominal columns (MSZoning, Neighborhood, etc.), creating one new column per category value

After these calls your DataFrame has around 120-185 columns (depending on neighborhood dummies). Check the shape to confirm.

**Why LogSalePrice matters:** SalePrice is right-skewed — a few expensive houses stretch the distribution to the right. Linear Regression assumes the target is normally distributed. Taking `np.log1p(SalePrice)` compresses the tail and makes the distribution much closer to normal (skewness drops from ~2.5 to ~-0.15). Random Forest and XGBoost don't have this assumption, so they use raw SalePrice. You'll end up with two target columns: `SalePrice` and `LogSalePrice`.

> ✅ Check: after loading, print `df.shape` and `len(df.columns)`. If you see columns you don't recognize, check that the feature engineering functions ran in the right order.

---

## Step 2 — Separate features from targets

**What you're doing:** Split the DataFrame into X (the matrix of features that goes into every model) and y (the target you're predicting).

**How to do it:** Take all numeric columns from the DataFrame, then remove SalePrice and LogSalePrice. What remains is X. Then extract SalePrice and LogSalePrice as separate Series (y_raw and y_log).

**Why take all numeric columns:** After one-hot encoding, every column in the DataFrame is numeric. There are no object/string columns left. So selecting numeric columns gives you everything: the original numeric features, the derived features, the ordinal-encoded integers, and all the one-hot dummy columns. This is simpler than maintaining a hardcoded list of feature names and less error-prone.

**What you have after this step:** X (DataFrame with ~120-185 columns and 2919 rows), y_raw (Series, 2919 dollar values), y_log (Series, 2919 log-transformed values).

> 🔍 Check: print `X.isna().sum().sum()` to see if there are any missing values in the feature matrix. The pipeline's imputer will handle them, but you should know the count.

---

## Step 3 — Train/test split

**What you're doing:** Split X, y_raw, and y_log into training and test sets. 80% training, 20% test. Use a fixed random_state so the split is reproducible.

**Important:** split y_log using the same indices as the y_raw split. If you call train_test_split three separate times (once for X, once for y_raw, once for y_log), each call uses a different random shuffle and the indices won't align. Instead: split X and y_raw together, then index y_log by the resulting train/test indices.

**After this step:** X_train, X_test, y_train_raw, y_test_raw, y_train_log, y_test_log. All aligned by index.

> ⚠️ Verify: assert `X_train.index.equals(y_train_raw.index)` — if this fails, your indices are misaligned and everything downstream will be wrong.

---

## Step 4 — Build the preprocessing pipeline

**What you're doing:** Build a ColumnTransformer that applies two operations to every feature column: (1) impute missing values with the median, (2) scale to zero mean and unit variance.

**Why a pipeline:** Without a pipeline, you'd fit the imputer on all data, transform X_train and X_test separately, then fit the scaler on all data again, then transform separately. Each step is a chance to leak information from the test set into the training process. A Pipeline bundles preprocessing + model into one object that you fit on X_train only and use to transform X_test. The test data never influences the imputer's median or the scaler's mean/variance.

**Why StandardScaler:** Linear Regression benefits from scaled features — the coefficients become comparable and the optimization is more stable. Random Forest and XGBoost don't need scaling (trees split on individual thresholds, not distances), but including the scaler in the shared pipeline is harmless and keeps things consistent. The imputer is needed for all three models because the feature matrix may have NaNs after one-hot encoding.

**Structure of the ColumnTransformer:** One transformer named 'num' that applies to all feature columns. It's a Pipeline itself: `[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]`. The ColumnTransformer has `remainder='drop'` so any column not listed is discarded (there shouldn't be any, since you're listing all columns).

**When to fit the preprocessor:** Fit the preprocessor on X_train only, then use it to transform both X_train and X_test. The test data is transformed using the statistics computed from the training data only.

> 🧪 Test the pipeline: after fitting, call `preprocessor.transform(X_train)` and check the shape and NaN count. The output should have the same number of rows as X_train and zero NaNs.

---

## Step 5 — Train Model 1: Linear Regression

**What you're doing:** Wrap the preprocessor and a LinearRegression in a Pipeline, fit it on X_train and y_train_log, predict on X_test, convert predictions from log space back to dollars, and evaluate.

**Why LogSalePrice for Linear Regression:** Linear Regression assumes the residuals are normally distributed, which implies the target should be roughly normal. Raw SalePrice is heavily right-skewed. The log transform makes the distribution symmetric enough that the linear model's assumptions hold better. The model learns to predict log dollars, and you convert predictions back with `np.expm1()` before comparing to actual dollars.

**Evaluation:** Compute MAE, RMSE, and R² by comparing the dollar predictions to y_test_raw (not y_test_log). The predictions are in dollars after np.expm1(), and the actual values are in dollars, so the comparison is meaningful.

**What to look for:** R² around 0.7-0.85 is typical for Linear Regression on this dataset with good features. If R² is below 0.6, the features may not be capturing enough signal, or the log transform may not have fully addressed the skew. Check the first 10 predictions vs actual to get a feel for the error magnitude.

> 🔢 Don't forget: the predictions from LinearRegression are in log space. Convert with `np.expm1(y_pred_log)` to get dollars before computing MAE/RMSE/R².

---

## Step 6 — Train Model 2: Random Forest

**What you're doing:** Wrap the preprocessor and a RandomForestRegressor in a Pipeline, fit on X_train and y_train_raw (raw dollars), predict on X_test, evaluate.

**Why raw SalePrice for Random Forest:** Decision trees split on individual feature values and don't assume any distribution for the target. The skew in SalePrice doesn't affect them. Using raw dollars also means you don't need to convert predictions back — what the model predicts is already in dollars.

**Suggested starting hyperparameters:** n_estimators=200 (number of trees), max_depth=15 (how deep each tree can grow), min_samples_split=5 (minimum samples to split a node), min_samples_leaf=2 (minimum samples in a leaf), random_state=42 (reproducibility), n_jobs=-1 (use all CPU cores). These are starting points — the right values depend on your data and are tuned in Day 4.

**Feature importance:** After training, access the model via `pipeline.named_steps['regressor']` and read its `feature_importances_` attribute. Create a Series indexed by X.columns and sort descending. Print the top 15. This tells you which features the model found most useful — useful for understanding the model and for Day 4 tuning.

**What to look for:** R² typically 0.85-0.92 for Random Forest on this dataset. If it's much higher (above 0.95), check for data leakage — the model may be seeing information it shouldn't. If it's lower than Linear Regression, the hyperparameters may need tuning.

---

## Step 7 — Train Model 3: XGBoost

**What you're doing:** Same pattern as Random Forest: Pipeline(preprocessor + XGBRegressor), fit on X_train and y_train_raw, predict, evaluate, extract feature importance.

**Why XGBoost:** Gradient boosting builds trees sequentially, each one correcting the errors of the previous ones. It often outperforms Random Forest on tabular data because it focuses on the hard cases. It's the model that usually wins on the Ames Housing dataset.

**Suggested starting hyperparameters:** n_estimators=300 (number of boosting rounds), max_depth=6 (tree depth — boosting uses shallower trees than Random Forest), learning_rate=0.05 (step size for each tree — lower means more trees needed but often better results), subsample=0.8 (fraction of samples per tree — adds randomness, reduces overfitting), colsample_bytree=0.8 (fraction of features per tree — same idea), random_state=42, n_jobs=-1, verbosity=0 (suppress training output).

**Feature importance:** Same as Random Forest — access via `pipeline.named_steps['regressor'].feature_importances_`, build a Series with X.columns as the index, sort descending, print top 15.

**What to look for:** R² typically 0.88-0.94 for XGBoost on this dataset. Compare the top features with Random Forest's top features — if they're similar, that's a sign the important features are robust. If they're very different, investigate why.

---

## Step 8 — Compare all models

**What you're doing:** Put the three models' results side by side and visualize the comparison.

**Build a comparison table:** Create a DataFrame with one row per model and columns MAE, RMSE, R². Print it. The model with the highest R² is the best on the test set.

**Bar charts:** Plot MAE, RMSE, and R² as three separate bar charts, one bar per model. This makes the comparison visual and immediately shows which model wins on each metric.

**Scatter plots (actual vs predicted):** For each model, plot the actual test values on the x-axis and the predicted values on the y-axis. Add a red diagonal line (y = x) representing perfect predictions. Points on the line are perfect predictions; points away from the line have error. This visualization shows whether the model systematically over-predicts or under-predicts, and whether errors are concentrated in certain price ranges.

> ⚖️ Fair comparison reminder: all three models were trained on the same X_train, X_test split with the same random_state. If they weren't, the comparison is meaningless.

---

## Step 9 — Cross-validation on Linear Regression

**What you're doing:** Run 5-fold cross-validation on the training set to check if the Linear Regression result is stable across different data splits.

**Why cross-validation:** A single train/test split can be lucky or unlucky — the test set might be easier or harder than average. Cross-validation splits the training data into 5 folds, trains on 4 and validates on 1, rotating through all 5. The average R² across folds is a more reliable estimate of how the model will perform on unseen data.

**How to do it:** Use `cross_val_score` with the full pipeline (preprocessor + LinearRegression), X_train, y_train_log, `cv=KFold(n_splits=5, shuffle=True, random_state=42)`, `scoring='r2'`. This returns 5 R² scores (one per fold). Also compute MAE scores with `scoring='neg_mean_absolute_error'` (negated because sklearn returns negative values for error metrics).

**How to interpret:** Compare the mean CV R² with the test set R². If they're within 0.02 of each other, the test result is consistent and the model is stable. If the test R² is much higher than the CV R², the test set may have been unusually easy — the CV score is the more honest estimate. If the CV R² is much higher, the test set may be unusually hard.

**Also run CV on raw SalePrice:** Use the same pipeline but with y_train_raw instead of y_train_log. This shows how Linear Regression performs without the log transform, for comparison. It will almost certainly be worse than the log version, confirming why the log transform is needed.

> ⏱️ Cross-validation is computationally more expensive (5 fits instead of 1). For Random Forest and XGBoost with many trees, this can take noticeably longer. Day 4 will add CV for all three models — for Day 3, just doing LR is enough to understand the concept.

---

## Step 10 — Save the best model

**What you're doing:** Save the winning model and all the information needed to use it later (in the Streamlit app or for further analysis).

**What to save and why:**

- **The full pipeline (best_pipeline.pkl):** preprocessor + model together. This is what the Streamlit app loads — it takes raw input, preprocesses it the same way, and predicts. Without the preprocessor, the app has no way to scale or impute new data.
- **The model alone (best_model.pkl):** the regressor object without preprocessing. Useful for inspection, feature importance analysis, or reloading in a different context.
- **The feature names (feature_names.json):** the list of column names in X, in order. The Streamlit app needs this to know which columns to expect in new data and in what order. If the app receives data with different columns or a different order, the prediction will be wrong.
- **The target info (target_info.json):** which model won, which target was used (LogSalePrice or SalePrice), and the test set metrics. This documents what was saved and how it was evaluated, so you don't have to remember later.

**Where to save:** Create a `models/` folder in the project root. All four files go there. The folder structure after saving should look like: `models/best_pipeline.pkl`, `models/best_model.pkl`, `models/feature_names.json`, `models/target_info.json`.

> 💾 Test the saved pipeline: after saving, load it back with `joblib.load()` and run a prediction on a single row from X_test. Compare the loaded prediction with the original. If they match, the save was successful.

---

## Step 11 — Error analysis (optional but recommended)

**What you're doing:** Look at where the best model makes the biggest errors. This helps you understand the model's weaknesses and prepares you for Day 4 tuning.

**Group errors by price range:** Bin the test houses by actual price (e.g., <$100K, $100-150K, $150-200K, etc.) and compute the mean absolute error and median percentage error for each bin. If errors are much larger for expensive houses, the model struggles with the tail. If they're larger for cheap houses, the model may be biased toward the middle of the distribution.

**Look at the worst predictions:** Find the 10 test houses with the largest absolute error. Print their actual price, predicted price, and error. Look for patterns — are they all from the same neighborhood? All new construction? All very large houses? This can reveal systematic issues.

**Look at the best predictions:** Find the 10 test houses with the smallest absolute error. These are the houses the model predicts almost perfectly. Understanding what they have in common helps you understand where the model is strongest.

---

## Step 12 — What to deliver at the end of Day 3

By the end of Day 3 you should have:

- A fully run notebook (`03_modeling.ipynb`) that executes without errors from top to bottom
- A comparison table showing MAE, RMSE, R² for all 3 models on the test set
- A clear answer to which model is best and why
- Cross-validation results for Linear Regression showing the model is stable
- Four files in `models/`: `best_pipeline.pkl`, `best_model.pkl`, `feature_names.json`, `target_info.json`
- Feature importance tables for Random Forest and XGBoost (top 15 features each)

This is also the point where you can start thinking about the Streamlit app (Day 5). The app will load `best_pipeline.pkl` and `feature_names.json`, receive user input, preprocess it the same way, and predict. Knowing what the pipeline expects (which columns, in what order) makes building the app much easier.

---

## After Day 3 — what comes next

Day 4 is about making the models better and more reliable:

- Cross-validation for all 3 models (not just Linear Regression)
- GridSearchCV on at least one model to tune hyperparameters systematically
- Deeper error analysis — which features correlate with large errors?
- Final model selection with justification

Day 5 is the Streamlit app and Docker packaging. The app loads the saved pipeline and predicts on new data — it does NOT retrain. That's why saving the full pipeline (not just the model) in Step 10 is critical.

---

## Resources

- [Sklearn Pipeline and ColumnTransformer documentation](https://scikit-learn.org/stable/modules/compose.html)
- [Sklearn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html)
- [XGBoost sklearn API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn)
- [joblib persistence (save/load models)](https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html)

Your own modules to use:

- `src/preprocess.py` — CleanData class (loads and cleans the raw CSV)
- `src/features.py` — add_numeric_features, encode_ordinal_features, encode_nominal_features
- `src/analysis.py` — EDA functions (reference for what the data looks like after cleaning)

---

## Common pitfalls

- **Training on the test set by accident:** the split must happen BEFORE the pipeline is fit. If you fit the preprocessor on all data and then split, information leaks and your metrics are artificially good. Always split first, then fit the preprocessor only on X_train.
- **Using the wrong target for the wrong model:** Linear Regression should use LogSalePrice (the skewed distribution breaks its assumptions). Random Forest and XGBoost use raw SalePrice (trees don't care about distribution). Mixing these up gives misleading comparisons.
- **Comparing models trained on different data:** if you use different random_state values or different train_test_split calls for each model, you're not comparing apples to apples. Use the same split for all 3 models.
- **Forgetting to convert log predictions back to dollars:** LinearRegression predicts LogSalePrice, but you report MAE/RMSE in dollars. Convert back with np.expm1() before evaluating.
- **Ignoring missing values in X after feature engineering:** one-hot encoding creates many columns and some may have NaNs if the source data had them. The imputer in the pipeline handles this, but you should check the count before training to know what you're dealing with.
- **Saving only the model and not the preprocessor:** the Streamlit app needs the full pipeline (preprocessing + model) because new input data arrives unpreprocessed. If you save only the model, the app has no way to scale/impute new data the same way.
- **Overfitting to the test set:** if you tune hyperparameters by trial and error using the test set as feedback, you're effectively training on the test set. Use cross-validation (Day 4) for tuning, keep the test set untouched.

---

## Checklist

- [ ] Load CleanData and run all feature engineering functions
- [ ] Add LogSalePrice column (np.log1p of SalePrice)
- [ ] Separate X (all numeric minus targets), y_raw, y_log
- [ ] Split into train/test (80/20, fixed random_state)
- [ ] Build ColumnTransformer (imputer + scaler) and fit on X_train only
- [ ] Train Linear Regression on LogSalePrice, convert predictions back to dollars
- [ ] Train Random Forest on raw SalePrice, extract feature importance
- [ ] Train XGBoost on raw SalePrice, extract feature importance
- [ ] Build comparison table and visualizations (bars + scatter plots)
- [ ] Run 5-fold cross-validation on Linear Regression, compare CV vs test
- [ ] Save best pipeline, model, feature names, and target info to models/
- [ ] Verify saved pipeline by loading and predicting on a test row
