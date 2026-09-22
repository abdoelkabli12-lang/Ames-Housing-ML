from preprocess import CleanData as cl
from analysis import analyze_target, analyze_categoricals_low_card, analyze_categoricals_high_card, analyze_correlations, plot_additional_feature_scatter, detect_target_outliers, create_outlier_decision_table, plot_feature_scatter_outliers, final_data_check




x = cl().clean()


print(analyze_target(x))
print(analyze_correlations(x))
print(analyze_categoricals_low_card(x))
print(analyze_categoricals_high_card(x))
print(detect_target_outliers(x))
print(plot_feature_scatter_outliers(x))
print(plot_additional_feature_scatter(x))
print(final_data_check(x))

