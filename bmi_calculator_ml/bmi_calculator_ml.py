import numpy as np
from sklearn.linear_model import LinearRegression

def calculate_bmi(weight, height):
    """
    Calculate BMI using the formula: weight (kg) / height (m)^2
    """
    return weight / (height ** 2)

# Sample data for machine learning model
# Features: [Height in meters, Weight in kg]
# Target: BMI
sample_data = np.array([
    [1.5, 50], [1.6, 60], [1.7, 70], [1.8, 80], [1.9, 90]
])

height_weight = sample_data[:, 0:2]
bmi_values = np.array([calculate_bmi(row[1], row[0]) for row in sample_data])

# Train a simple linear regression model to predict BMI
model = LinearRegression()
model.fit(height_weight, bmi_values)

# Input for BMI prediction
input_height = float(input("Enter your height in meters: "))
input_weight = float(input("Enter your weight in kg: "))

predicted_bmi = model.predict(np.array([[input_height, input_weight]]))[0]

# BMI classification
if predicted_bmi < 18.5:
    category = "Underweight"
elif 18.5 <= predicted_bmi < 24.9:
    category = "Normal weight"
elif 25 <= predicted_bmi < 29.9:
    category = "Overweight"
else:
    category = "Obesity"

print(f"Your predicted BMI is: {predicted_bmi:.2f}")
print(f"BMI Category: {category}")
