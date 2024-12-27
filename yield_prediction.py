import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
import xgboost as xgb
import catboost as cb

# Load data
df = pd.read_csv('yield.csv')
X = df.copy()
y = X.pop('Value')

# Use One-Hot Encoding for categorical columns
X = pd.get_dummies(X, drop_first=True)  # drop_first=True to avoid dummy variable trap

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regression": DecisionTreeRegressor(random_state=42),
    "Random Forest Regression": RandomForestRegressor(random_state=42),
    "XGBoost Regression": xgb.XGBRegressor(random_state=42)
}

# Function to calculate evaluation metrics
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "EVS": evs,
        "MAPE": mape
    }

# Function to plot results for actual vs predicted values
def plot_results(y_test, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Data points', alpha=0.7, s=10)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal fit')
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.legend()
    plt.show()

# Evaluate models on both training and test sets
for model_name, model in models.items():
    print(f"\n{model_name} - Evaluation Metrics:")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predictions for training set
    y_train_pred = model.predict(X_train)
    train_metrics = evaluate_model(y_train, y_train_pred)
    
    # Predictions for test set
    y_test_pred = model.predict(X_test)
    test_metrics = evaluate_model(y_test, y_test_pred)
    
    # Print training metrics
    print("Training Set:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Print test metrics
    print("Test Set:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Plot results for test set
    plot_results(y_test, y_test_pred, model_name)