import pandas as pd
import numpy as np

def classify_market_gaps(df, actual_col, predicted_col, std_multiplier=1.0):
   
    df['Residual'] = df[actual_col] - df[predicted_col]
    
    mean_residual = df['Residual'].mean()
    std_residual = df['Residual'].std()
    
    lower_bound = mean_residual - (std_multiplier * std_residual)
    upper_bound = mean_residual + (std_multiplier * std_residual)
    
    print(f"--- Classification Bounds ---")
    print(f"Mean Residual: ${mean_residual:,.2f}")
    print(f"Lower Bound (Undervalued): < ${lower_bound:,.2f}")
    print(f"Upper Bound (Overvalued):  > ${upper_bound:,.2f}")
    

    conditions = [
        (df['Residual'] < lower_bound),
        (df['Residual'] > upper_bound)
    ]
    
    
    choices = ['Undervalued', 'Overvalued']
    
    df['Valuation_Status'] = np.select(conditions, choices, default='Fairly Valued')
    
    return df

test_data = pd.DataFrame({
    'ZIP_Code': ['30301', '30302', '30303', '30304', '30305', '30306', '30307'],
    'Actual_Price': [450000, 500000, 250000, 750000, 420000, 600000, 900000],
    'Predicted_Price': [550000, 490000, 350000, 755000, 300000, 610000, 700000]
})

final_mapped_df = classify_market_gaps(
    df=test_data, 
    actual_col='Actual_Price', 
    predicted_col='Predicted_Price', 
    std_multiplier=1.0
)

print("\n--- Final Output Ready for Geopandas ---")
print(final_mapped_df[['ZIP_Code', 'Residual', 'Valuation_Status']])