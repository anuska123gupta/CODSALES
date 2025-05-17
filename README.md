\'\'\'python
# CODSALES
Repository for CODSOFT Data Science Internship tasks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Load the Dataset ---
try:
    data = pd.read_csv('Advertising.csv')
except FileNotFoundError:
    print("Error: 'Advertising.csv' not found. Please make sure the file is in the correct directory.")
    exit()

# --- 2. Data Cleaning ---
# Check for and remove an unnecessary index column
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'], axis=1)
    print("Removed unnecessary index column.")

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())
# Handle missing values if any (e.g., imputation):
# data = data.fillna(data.mean())

# Check for duplicate rows
print("\nNumber of duplicate rows:", data.duplicated().sum())
# Remove duplicate rows if any:
# data = data.drop_duplicates()
# print("Duplicate rows removed.")

# Check data types
print("\nData types:")
print(data.dtypes)

# --- 3. Exploratory Data Analysis (EDA) ---
print("\n--- Exploratory Data Analysis ---")

# Visualize the relationships between advertising spend and sales
sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=5, aspect=0.8)
plt.suptitle("Relationship between Advertising Spend and Sales", y=1.02)
plt.show()
plt.savefig('movie_ratings_distribution.png')

# Create a heatmap to visualize the correlation matrix
correlation_matrix = data[['TV', 'Radio', 'Newspaper', 'Sales']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# --- 4. Feature Selection ---
print("\n--- Feature Selection ---")
X = data[['TV', 'Radio', 'Newspaper']] # Features (independent variables)
y = data['Sales'] # Target variable (dependent variable)

# --- 5. Split the Data ---
print("\n--- Data Splitting ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of y_test: {y_test.shape}")

# --- 6. Choose and Train a Model ---
print("\n--- Model Training ---")
# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# --- 7. Make Predictions ---
print("\n--- Making Predictions ---")
# Make predictions on the test data
y_pred = model.predict(X_test)

# --- 8. Evaluate the Model ---
print("\n--- Model Evaluation ---")
# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Visualize the predictions against the actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual Sales vs. Predicted Sales")
plt.show()

# --- 9. Interpret the Results ---
print("\n--- Model Interpretation ---")
# Get the coefficients of the model
print("Coefficients (TV, Radio, Newspaper):", model.coef_)
print("Intercept:", model.intercept_)

# The coefficients indicate the change in Sales for a one-unit increase
# in the corresponding advertising medium, holding other mediums constant.
# The intercept is the predicted Sales value when all advertising spends are zero.

# --- 10. Making New Predictions ---
print("\n--- Making New Predictions ---")
# Example of predicting sales for a new advertising budget
new_data = pd.DataFrame({'TV': [200], 'Radio': [30], 'Newspaper': [50]})
predicted_sales_new = model.predict(new_data)
print(f"Predicted Sales for new budget (TV=200, Radio=30, Newspaper=50): {predicted_sales_new[0]:.2f}")

new_data_multiple = pd.DataFrame({'TV': [100, 150, 250], 'Radio': [20, 40, 10], 'Newspaper': [30, 0, 60]})
predicted_sales_multiple = model.predict(new_data_multiple)
print("\nPredicted Sales for multiple new budgets:")
print(predicted_sales_multiple)



Result:


Missing values:
TV           0
Radio        0
Newspaper    0
Sales        0
dtype: int64

Number of duplicate rows: 0

Data types:
TV           float64
Radio        float64
Newspaper    float64
Sales        float64
dtype: object

--- Exploratory Data Analysis ---

--- Feature Selection ---

--- Data Splitting ---
Shape of X_train: (160, 3)
Shape of X_test: (40, 3)
Shape of y_train: (160,)
Shape of y_test: (40,)

--- Model Training ---

--- Making Predictions ---

--- Model Evaluation ---
Mean Squared Error: 2.91
R-squared: 0.91

--- Model Interpretation ---
Coefficients (TV, Radio, Newspaper): [0.05450927 0.10094536 0.00433665]
Intercept: 4.714126402214127

--- Making New Predictions ---
Predicted Sales for new budget (TV=200, Radio=30, Newspaper=50): 18.86

Predicted Sales for multiple new budgets:
[12.31406014 16.92833152 19.61109654]
![s](https://github.com/user-attachments/assets/15da327d-a94d-4b2a-8f9f-79bebd85eef3)
![a](https://github.com/user-attachments/assets/22c99cad-1798-4264-92c1-f5e8f5981a1c)
![l](https://github.com/user-attachments/assets/a4f54cf0-3db1-4ecd-95bc-d8e3ebd4d983)


\'\'\'
