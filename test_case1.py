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

# Load data
df = pd.read_csv('yield.csv')
X = df.copy()
y = X.pop('Value')

# One-Hot Encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Apply Yeo-Johnson transformation to continuous features
power_transformer = PowerTransformer(method='yeo-johnson')
continuous_features = X.select_dtypes(include=[np.number]).columns
X[continuous_features] = power_transformer.fit_transform(X[continuous_features])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define regression models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
      "Random Forest (Optimized)": RandomForestRegressor(
        n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
    ),
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
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else float('inf')
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "EVS": evs,
        "MAPE": mape
    }

# Evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_test_pred)
    results[name] = metrics
    print(f"\n{name} - Test Set Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Plot actual vs predicted values
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_test_pred, alpha=0.6, label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit')
    plt.title(f'{name}: Predicted vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.show()

# Find the best model based on RMSE
best_model = min(results, key=lambda x: results[x]["RMSE"])
print(f"\nBest Model: {best_model}")
print("Performance Metrics:")
for metric, value in results[best_model].items():
    print(f"  {metric}: {value:.4f}")

# Example prediction
example_input = {
    "average_rain_fall_mm_per_year": 1200,
    "avg_temp": 25,
    "pH": 6.5,
    "Moisture": 75,
    "Soil EC": 0.3,
    "Phosphorus": 20,
    "Potassium": 100,
    "Urea": 60,
    # Include dummy values for one-hot encoded columns
    "Categorical_Feature_1": 0,  # Replace with actual column names
    "Categorical_Feature_2": 1   # Replace with actual column names
}
example_df = pd.DataFrame([example_input])
example_df = example_df.reindex(columns=X_train.columns, fill_value=0)
example_df[continuous_features] = power_transformer.transform(example_df[continuous_features])

# Predict using the best model
best_model_instance = models[best_model]
example_prediction = best_model_instance.predict(example_df)
print(f"\nPredicted Yield (hg/ha) using {best_model}: {example_prediction[0]}")
