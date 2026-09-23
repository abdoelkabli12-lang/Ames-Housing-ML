import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from preprocess import CleanData as cl


# ── Section 3: Target Analysis ──────────────────────────────────────────────

def analyze_target(df, target='SalePrice'):
    if target not in df.columns:
        print(f"Column '{target}' not found in DataFrame.")
        return df

    print(f"=== {target} — Basic Stats ===")
    print(df[target].describe())
    print(f"\nZero values: {(df[target] == 0).sum()}")
    print(f"Negative values: {(df[target] < 0).sum()}")

    # Raw distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target], kde=True, bins=50)
    plt.title(f'{target} Distribution — Raw')
    plt.xlabel(target)
    plt.show()

    # Skewness
    skew_raw = df[target].skew()
    print(f"\nSkewness (raw {target}): {skew_raw:.3f}")

    # Log transform
    log_col = f'Log{target}'
    df[log_col] = np.log1p(df[target])

    # Log distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df[log_col], kde=True, bins=50, color='orange')
    plt.title(f'{target} Distribution — Log Transformed')
    plt.xlabel(f'log(1 + {target})')
    plt.show()

    # Skewness after log
    skew_log = df[log_col].skew()
    print(f"Skewness (log {target}): {skew_log:.3f}")

    print(f"\n→ Raw skewness: {skew_raw:.3f}  →  Log skewness: {skew_log:.3f}")
    if abs(skew_log) < 0.5:
        print("→ Log transform successfully normalized the distribution.")
    else:
        print(f"→ Log transform reduced skewness but not fully normalized (|skew| = {abs(skew_log):.3f}).")

    return df


# ── Section 4: Correlation Analysis ─────────────────────────────────────────

def analyze_correlations(df, target='SalePrice', top_n=20, plot=True):
    if target not in df.columns:
        print(f"Column '{target}' not found.")
        return None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target not in numeric_cols:
        print(f"Target '{target}' is not numeric — cannot compute Pearson correlation.")
        return None

    corr = df[numeric_cols].corr()[target].sort_values(ascending=False)

    print(f"=== Correlations with {target} ===\n")
    print(f"Top {top_n} positive correlations:")
    print(corr.head(top_n).to_string())
    print(f"\nTop {top_n} negative correlations:")
    print(corr.tail(top_n).to_string())

    # Weak correlation count
    weak = (corr.abs() < 0.1) & (corr.index != target)
    print(f"\nFeatures with |correlation| < 0.1: {weak.sum()} (likely not individually predictive)")

    # Bar chart
    if plot:
        plot_cols = corr.head(top_n + 1).drop(target, errors='ignore')
        plt.figure(figsize=(10, 8))
        plot_cols.plot(kind='barh', color='steelblue')
        plt.title(f'Top {top_n} Features — Correlation with {target}')
        plt.xlabel('Correlation')
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        plt.tight_layout()
        plt.show()

    return corr

# ── Section 5: Categorical — Low Cardinality ────────────────────────────────

def analyze_categoricals_low_card(df, max_unique=15, target='SalePrice'):
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    low_card = [c for c in cat_cols if df[c].nunique() <= max_unique]

    print(f"=== Categorical Columns — Low Cardinality (<= {max_unique} unique) ===")
    print(f"Found {len(low_card)} columns: {low_card}\n")

    for col in low_card:
        print(f"{'='*60}")
        print(f"  {col}  ({df[col].nunique()} unique values)")
        print(f"{'='*60}")
        summary = df.groupby(col)[target].agg(['mean', 'median', 'count', 'std'])
        summary = summary.sort_values('mean', ascending=False)
        print(summary.to_string())
        print()



# ── Section 6: Categorical — High Cardinality ───────────────────────────────

def analyze_categoricals_high_card(df, target='SalePrice'):

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    high_card = [c for c in cat_cols if df[c].nunique() > 15]

    print(f"=== Categorical Columns — High Cardinality (> 15 unique) ===")
    print(f"Found {len(high_card)} columns: {high_card}\n")

    for col in high_card:
        print(f"{'='*60}")
        print(f"  {col}  ({df[col].nunique()} unique values)")
        print(f"{'='*60}")
        print(f"Top 10 most frequent values:")
        print(df[col].value_counts().head(10).to_string())
        print()

    # Neighborhood deep dive
    if 'Neighborhood' in df.columns:
        print(f"{'='*60}")
        print("  Neighborhood — Mean SalePrice (sorted)")
        print(f"{'='*60}")
        nbhd = df.groupby('Neighborhood')[target].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False)
        print(nbhd.to_string())
        print()

        # Visualize
        plt.figure(figsize=(12, 8))
        nbhd_sorted = df.groupby('Neighborhood')[target].mean().sort_values()
        nbhd_sorted.plot(kind='barh', color='steelblue')
        plt.title('Mean SalePrice by Neighborhood')
        plt.xlabel('Mean Sale Price ($)')
        plt.tight_layout()
        plt.show()
# ── Section 7: Outlier Detection — Target ────────────────────────────────────

def detect_target_outliers(df, target='SalePrice', log_target='LogSalePrice'):

    results = {}

    # Z-score on log target
    if log_target in df.columns:
        z_scores = np.abs(stats.zscore(df[log_target]))
        outliers_z = df[z_scores > 3]
        results['zscore'] = outliers_z
        print(f"=== Outliers by Z-score on {log_target} (> 3σ) ===")
        print(f"Count: {len(outliers_z)}")
        if len(outliers_z) > 0:
            show_cols = [c for c in ['SalePrice', 'GrLivArea', 'OverallQual',
                                       'YearBuilt', 'Neighborhood', 'HouseStyle']
                         if c in df.columns]
            print(outliers_z[show_cols].to_string())
        print()


    # IQR on raw target
    if target in df.columns:
        Q1 = df[target].quantile(0.25)
        Q3 = df[target].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers_iqr = df[(df[target] < lower) | (df[target] > upper)]
        results['iqr'] = outliers_iqr
        print(f"=== Outliers by IQR on {target} ===")
        print(f"Q1: ${Q1:,.0f}  |  Q3: ${Q3:,.0f}  |  IQR: ${IQR:,.0f}")
        print(f"Lower bound: ${lower:,.0f}  |  Upper bound: ${upper:,.0f}")
        print(f"Count: {len(outliers_iqr)}")
        if len(outliers_iqr) > 0:
            show_cols = [c for c in ['SalePrice', 'GrLivArea', 'OverallQual']
                         if c in df.columns]
            print(outliers_iqr[show_cols].head(20).to_string())
        print()

    return results



# ── Section 8: Outlier Detection — Feature Scatter Plots ────────────────────

def plot_feature_scatter_outliers(df, target='SalePrice', features=None):
    if features is None:
        features = [
            'GrLivArea', 'LotArea', 'TotalBsmtSF',
            '1stFlrSF', 'GarageArea', 'MasVnrArea',
            'LotFrontage', '2ndFlrSF', 'BsmtFinSF1'
        ]

    available = [f for f in features if f in df.columns]

    for feat in available:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=feat, y=target, alpha=0.3, s=10)
        plt.title(f'{feat} vs {target}')
        plt.xlabel(feat)
        plt.ylabel(target)
        plt.axhline(y=df[target].median(), color='gray', linestyle='--', alpha=0.3, label=f'Median {target}')
        plt.legend()
        plt.show()

    # Special: GrLivArea — highlight potential outliers
    if 'GrLivArea' in df.columns:
        extreme = df[(df['GrLivArea'] > 3500) & (df[target] < 300000)]
        print(f"\n=== GrLivArea Outlier Candidates ===")
        print(f"GrLivArea > 3500 sqft AND {target} < $300K")
        print(f"Count: {len(extreme)}")
        if len(extreme) > 0:
            id_col = 'Id' if 'Id' in df.columns else None
            show_cols = [c for c in ['SalePrice', 'GrLivArea', 'OverallQual',
                                       'YearBuilt', 'Neighborhood', 'HouseStyle', id_col]
                         if c in df.columns]
            print(extreme[show_cols].to_string())


# ── Section 9: Additional Feature Scatter Plots ─────────────────────────────

def plot_additional_feature_scatter(df, target='SalePrice', features=None):
    if features is None:
        features = [
            'LotFrontage', '1stFlrSF', '2ndFlrSF',
            'BsmtFinSF1', 'BsmtUnfSF', 'WoodDeckSF',
            'OpenPorchSF', 'PoolArea'
        ]

    available = [f for f in features if f in df.columns]

    for feat in available:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=feat, y=target, alpha=0.3, s=10)
        plt.title(f'{feat} vs {target}')
        plt.xlabel(feat)
        plt.ylabel(target)
        plt.axhline(y=df[target].median(), color='gray', linestyle='--', alpha=0.3)
        plt.show()


# ── Section 10: Outlier Decision Summary ────────────────────────────────────

def create_outlier_decision_table():
    columns = ['Feature', 'Issue', 'Count', 'Decision', 'Action']
    return pd.DataFrame(columns=columns)


# ── Section 12: Final Data Check ────────────────────────────────────────────

def final_data_check(df):
    print("=== Final Data Check ===")
    print(f"Shape: {df.shape}")
    print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(f"\nMissing values: {len(missing)} columns with missing data")
    if len(missing) > 0:
        print(missing.to_string())
    else:
        print("No missing values — data is clean.")

    print(f"\nColumn types:")
    print(f"  Numeric: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"  Categorical: {len(df.select_dtypes(include=['object']).columns)}")

    print(f"\nAll columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:3d}. {col}")

    return df.shape, len(missing), df.columns.tolist()

x = cl().clean()




























































































































