import pandas as pd
import numpy as np

def classify_market_gaps(df, actual_col, predicted_col, std_multiplier=1.0):
    """
    Classifies real estate markets based on the residuals of an XGBoost Regression.
    
    Parameters:
    - df: The dataframe containing your ZIP codes, actual prices, and predicted prices.
    - actual_col: String name of the column with ground-truth prices.
    - predicted_col: String name of the column with XGBoost predictions.
    - std_multiplier: How strict the threshold should be. 
                      1.0 = ~68% of data is "Fair", 15% Over, 15% Under.
                      
    Returns:
    - DataFrame with new 'Residual' and 'Valuation_Status' columns.
    """
    # 1. Calculate the Residual (Actual Price - What the Model Thinks it Should Be)
    # Negative Residual = The home is cheaper than its fundamentals suggest (Undervalued)
    df['Residual'] = df[actual_col] - df[predicted_col]
    
    # 2. Calculate the statistical bounds
    mean_residual = df['Residual'].mean()
    std_residual = df['Residual'].std()
    
    lower_bound = mean_residual - (std_multiplier * std_residual)
    upper_bound = mean_residual + (std_multiplier * std_residual)
    
    print(f"--- Classification Bounds ---")
    print(f"Mean Residual: ${mean_residual:,.2f}")
    print(f"Lower Bound (Undervalued): < ${lower_bound:,.2f}")
    print(f"Upper Bound (Overvalued):  > ${upper_bound:,.2f}")
    
    # 3. Apply the categorical tags
    conditions = [
        (df['Residual'] < lower_bound),
        (df['Residual'] > upper_bound)
    ]
    
    # The tags MUST match the exact spelling in your Choropleth color dictionary
    choices = ['Undervalued', 'Overvalued']
    
    df['Valuation_Status'] = np.select(conditions, choices, default='Fairly Valued')
    
    return df

# ==========================================
# Example Usage with your XGBoost Pipeline
# ==========================================

# Assuming 'X_test' is your feature dataframe and 'y_test' are the actual prices
# xgb_model = ... (your trained XGBoost regressor)

# 1. Get the predictions from your trained model
# predicted_prices = xgb_model.predict(X_test)

# 2. Create a working dataframe with your ZIP codes, Actuals, and Predictions
# (Replace with your actual dataframe variables)
test_data = pd.DataFrame({
    'ZIP_Code': ['30301', '30302', '30303', '30304', '30305', '30306', '30307'],
    'Actual_Price': [450000, 500000, 250000, 750000, 420000, 600000, 900000],
    'Predicted_Price': [550000, 490000, 350000, 755000, 300000, 610000, 700000]
})

# 3. Run the classifier!
# Setting std_multiplier=1.0 is the standard academic baseline.
final_mapped_df = classify_market_gaps(
    df=test_data, 
    actual_col='Actual_Price', 
    predicted_col='Predicted_Price', 
    std_multiplier=1.0
)

# Show the results
print("\n--- Final Output Ready for Geopandas ---")
print(final_mapped_df[['ZIP_Code', 'Residual', 'Valuation_Status']])