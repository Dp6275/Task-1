import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
file_path = 'pro1/train.csv'
data = pd.read_csv(file_path)

# Display the first few rows to understand the structure
print("Dataset Preview:")
print(data.head())

# Target variable: SalePrice
# Select relevant features: LotArea, OverallQual, YearBuilt
selected_features = ['LotArea', 'OverallQual', 'YearBuilt']
X = data[selected_features]
y = data['SalePrice']

# Handle missing values if any
X.fillna(X.mean(), inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output results
print(f"\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")
print("\nModel Coefficients:")
for feature, coef in zip(selected_features, model.coef_):
    print(f"{feature}: {coef}")
print(f"Intercept: {model.intercept_}")
