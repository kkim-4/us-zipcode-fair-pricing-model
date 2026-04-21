import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import shap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

print(" Loading and aligning datasets...")
df = pd.read_csv('XGBOOST_FAIR_VALUE_READY.csv', dtype={'ZIP': str})
realtor_target = pd.read_csv('master_realtor_yearly.csv', dtype={'ZIP': str})

df['ZIP'] = df['ZIP'].str.zfill(5)
realtor_target['ZIP'] = realtor_target['ZIP'].str.zfill(5)

df = df.merge(realtor_target, on=['ZIP', 'YEAR'], how='inner')

lower_bound = df['median_listing_price'].quantile(0.01)
upper_bound = df['median_listing_price'].quantile(0.99)
df = df[(df['median_listing_price'] > lower_bound) & (df['median_listing_price'] < upper_bound)]


features = [
    'ZVHI', 
    'median_income', 
    'crime_capita_rolling', 
    'school_count', 
    'schools_per_capita', 
    'yearly_temp_avg', 
    'temp_stability_score',
    'price_momentum',
    'population_estimate',
    'state_id'
]

df_clean = df.dropna(subset=features + ['median_listing_price']).copy()

df_clean['state_id'] = df_clean['state_id'].astype('category')

train_df = df_clean[df_clean['YEAR'] < 2025]
test_df = df_clean[df_clean['YEAR'] >= 2025]

X_train = train_df[features]
y_train = np.log1p(train_df['median_listing_price'])

X_test = test_df[features]
y_test = np.log1p(test_df['median_listing_price'])


X_train_naive = train_df[['ZVHI']]
X_test_naive = test_df[['ZVHI']]

print(f" Temporal Split: Training ({len(train_df)} rows) | Testing ({len(test_df)} rows)")


print("\n Training Linear Regression (Log-Space)...")
lr_model = LinearRegression().fit(X_train.drop(columns=['state_id']), y_train)

print(" Training Naïve Baseline XGBoost (ZVHI Only)...")
naive_xgb_model = xgb.XGBRegressor(
    n_estimators=1000, 
    learning_rate=0.05, 
    max_depth=6, 
    subsample=0.8,
    colsample_bytree=1.0 
).fit(X_train_naive, y_train)

print(" Training Champion XGBoost (Full Infrastructure Suite + Geography)...")
champion_xgb_model = xgb.XGBRegressor(
    n_estimators=1000, 
    learning_rate=0.05, 
    max_depth=6, 
    subsample=0.8,
    colsample_bytree=0.8,
    enable_categorical=True, 
    max_cat_to_onehot=55
).fit(X_train, y_train)

def evaluate(model, X, y_log, label):
    y_pred_log = model.predict(X)
    
    y_pred_log = np.clip(y_pred_log, 9.2, 18.4)
    
    y_true_usd = np.expm1(y_log)
    y_pred_usd = np.expm1(y_pred_log)
    
    mask = np.isfinite(y_pred_usd)
    if not np.all(mask):
        print(f" Warning: {label} produced non-finite values. Filtering for metrics...")
        y_true_usd = y_true_usd[mask]
        y_pred_usd = y_pred_usd[mask]
        y_log = y_log[mask]
        y_pred_log = y_pred_log[mask]

    mae = mean_absolute_error(y_true_usd, y_pred_usd)
    rmse = np.sqrt(mean_squared_error(y_true_usd, y_pred_usd))
    mape = mean_absolute_percentage_error(y_true_usd, y_pred_usd) * 100
    r2 = r2_score(y_log, y_pred_log)
    
    print(f"\n[{label}]")
    print(f"MAE (Avg Miss):    ${mae:,.2f}")
    print(f"RMSE (Penalty):    ${rmse:,.2f}")
    print(f"MAPE (Avg % Miss): {mape:.2f}%")
    print(f"R2 (Explained):    {r2:.4f}")

evaluate(lr_model, X_test.drop(columns=['state_id']), y_test, "Linear Regression (Full Features)")
evaluate(naive_xgb_model, X_test_naive, y_test, "Naïve XGBoost (ZVHI Only)")
evaluate(champion_xgb_model, X_test, y_test, "Champion XGBoost (Full Features)")

print("\n Generating SHAP insights for Model (Sampling 500 rows)...")
explainer = shap.TreeExplainer(champion_xgb_model)
X_sample = X_test.sample(min(500, len(X_test)), random_state=42)
shap_values = explainer(X_sample)

plt.figure(figsize=(10, 6))
shap.plots.beeswarm(shap_values, show=False)
plt.title("XGBoost SHAP: Infrastructure & Climate vs. Listing Price")
plt.tight_layout()
plt.show()

print("\n Conducting 2026 National Valuation Audit...")

df_clean['predicted_price'] = np.expm1(np.clip(champion_xgb_model.predict(df_clean[features]), 9.2, 18.4))
df_clean['fair_value_delta'] = (df_clean['median_listing_price'] - df_clean['predicted_price']) / df_clean['predicted_price']

delta_mean = df_clean['fair_value_delta'].mean()
delta_std = df_clean['fair_value_delta'].std()
df_clean['delta_zscore'] = (df_clean['fair_value_delta'] - delta_mean) / delta_std

def classify_statistically(z):
    if z < -1.0: return 'Undervalued'     
    elif z > 1.0: return 'Overvalued'    
    else: return 'Fairly Priced'         

df_clean['valuation_class'] = df_clean['delta_zscore'].apply(classify_statistically)

audit_2026 = df_clean[df_clean['YEAR'] == 2026].copy()
audit_2026.to_csv('USA_ZIP_VALUATION_AUDIT_2026_ZSCORE.csv', index=False)

print("\n Evaluating Classifier: 1-Year Forward Returns (2025 -> 2026)...")

backtest_2025 = df_clean[df_clean['YEAR'] == 2025][['ZIP', 'valuation_class', 'ZVHI']].copy()
backtest_2025.rename(columns={'ZVHI': 'ZVHI_2025'}, inplace=True)

reality_2026 = df_clean[df_clean['YEAR'] == 2026][['ZIP', 'ZVHI']].copy()
reality_2026.rename(columns={'ZVHI': 'ZVHI_2026'}, inplace=True)

eval_df = backtest_2025.merge(reality_2026, on='ZIP', how='inner')
eval_df['actual_1yr_return'] = ((eval_df['ZVHI_2026'] - eval_df['ZVHI_2025']) / eval_df['ZVHI_2025']) * 100

performance = eval_df.groupby('valuation_class')['actual_1yr_return'].mean()

print("\n" + "="*50)
print("   CLASSIFIER BACKTEST: ACTUAL 1-YEAR APPRECIATION")
print("="*50)
for status in ['Undervalued', 'Fairly Priced', 'Overvalued']:
    if status in performance:
        print(f"{status.ljust(15)}: {performance[status]:+.2f}% Average Growth")
print("-" * 50)

if 'Undervalued' in performance and 'Overvalued' in performance:
    alpha = performance['Undervalued'] - performance['Overvalued']
    print(f"Model Spread (Alpha): {alpha:+.2f}%")
print("="*50)

print("\n Generating 2026 Distribution Visualizations...")
counts = audit_2026['valuation_class'].value_counts()
pcts = audit_2026['valuation_class'].value_counts(normalize=True) * 100

summary_df = pd.DataFrame({
    'Market Status': counts.index,
    'ZIP Count': counts.values,
    'Market Share %': pcts.values
})

print("\n" + "="*45)
print("   NATIONAL MARKET VALUATION SUMMARY (2026)")
print("="*45)
print(summary_df.to_string(index=False))
print("-" * 45)
print(f"Total Unique ZIPs Audited: {len(audit_2026):,}")
print("="*45)

color_map = {
    'Fairly Priced': '#94a3b8', 
    'Undervalued':   '#10b981', 
    'Overvalued':    '#ef4444'  
}
mapped_colors = [color_map.get(status, '#334155') for status in summary_df['Market Status']]


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'DejaVu Sans']

fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')


bars = ax.bar(summary_df['Market Status'], summary_df['ZIP Count'], 
              color=mapped_colors, edgecolor='none', width=0.7, alpha=0.9)

ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#cbd5e1')
ax.xaxis.grid(False)


for bar, pct in zip(bars, summary_df['Market Share %']):
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2, 
        yval + (max(summary_df['ZIP Count']) * 0.01), 
        f"{int(yval):,}\n({pct:.1f}%)", 
        ha='center', va='bottom', 
        fontsize=11, fontweight='600', color='#1e293b'
    )

ax.set_title('National Housing Market Valuation Summary', 
             fontsize=18, fontweight='bold', color='#0f172a', pad=30, loc='left')

ax.text(0, 1.04, 'Distribution of ZIP codes based on price-to-income metrics (Q1 2026)', 
        transform=ax.transAxes, fontsize=11, color='#64748b')

ax.set_ylabel('Number of ZIP Codes', fontsize=12, fontweight='500', color='#475569', labelpad=15)


ax.tick_params(axis='both', which='major', labelsize=11, labelcolor='#475569', length=0)
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))


for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_color('#cbd5e1')


plt.tight_layout()
plt.savefig('valuation_bar_chart_v2.png', dpi=300, bbox_inches='tight')
plt.show()