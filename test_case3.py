import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import PowerTransformer
import xgboost as xgb
from catboost import CatBoostRegressor

# Load NDVI data
ndvi_df = pd.read_csv('NDVI_Pune.csv')  # Replace with the path to your NDVI satellite data
print("NDVI Data Preview:")
print(ndvi_df.head())

# Generate a synthetic 'Year' column for alignment (if applicable)
# Adjust this logic to suit your dataset structure
ndvi_df['Year'] = range(2000, 2000 + len(ndvi_df))  # Replace with actual years if available

# Load yield data
# Replace with your yield data file path
yield_data = {
    'Year': [2024,2025],  # Replace with actual years
    'Yield': [1431,1440]  # Replace with actual yield values
}
yield_df = pd.DataFrame(yield_data)

print("Yield Data Preview:")
print(yield_df)

# Merge NDVI data with yield data on 'Year'
df = pd.merge(ndvi_df, yield_df, on='Year')

# Extract features (X) and target variable (y)
X = df.drop(columns=['Yield'])
y = df['Yield']

# Apply Yeo-Johnson transformation to continuous features
power_transformer = PowerTransformer(method='yeo-johnson')
X_transformed = power_transformer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Define regression models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(random_state=42),
    "CatBoost": CatBoostRegressor(random_state=42, verbose=0)
}

# Function to evaluate models
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "EVS": evs
    }

# Evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    results[name] = metrics
    print(f"\n{name} Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Plot actual vs predicted values
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.6, label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit')
    plt.title(f'{name}: Predicted vs Actual')
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.legend()
    plt.show()

# Find the best model based on RMSE
best_model_name = min(results, key=lambda x: results[x]["RMSE"])
print(f"\nBest Model: {best_model_name}")
print("Performance Metrics:")
for metric, value in results[best_model_name].items():
    print(f"  {metric}: {value:.4f}")

# Example prediction with NDVI data
example_input = pd.DataFrame([[0.4, 0.5]], columns=X.columns)  # Replace with real feature values
example_transformed = power_transformer.transform(example_input)
best_model = models[best_model_name]
predicted_yield = best_model.predict(example_transformed)
print(f"\nPredicted Yield: {predicted_yield[0]:.2f}")
