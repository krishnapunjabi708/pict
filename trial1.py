import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load datasets
soil_data = pd.read_csv("Plant_Parameters.csv")
yield_data = pd.read_csv("yield_df.csv")

# Merge soil data with yield data based on crop type
data = yield_data.merge(soil_data, left_on="Item", right_on="Plant Type", how="left")

# Drop rows with missing values
data = data.dropna()

# Select features and target variable
features = ["average_rain_fall_mm_per_year", "avg_temp", "pH", "Soil EC", "Phosphorus", "Potassium", "Urea", "Moisture"]
X = data[features]
y = data["hg/ha_yield"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Example prediction
example_input = {
    "average_rain_fall_mm_per_year": 1200,
    "avg_temp": 25,
    "pH": 6.5,
    "Soil EC": 0.3,
    "Phosphorus": 20,
    "Potassium": 100,
    "Urea": 60,
    "Moisture": 75
}
example_df = pd.DataFrame([example_input])
example_prediction = model.predict(example_df)
print(f"Predicted Yield (hg/ha): {example_prediction[0]}")
