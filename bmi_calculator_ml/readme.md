BMI Calculator Using Python and Machine Learning

This project is a Body Mass Index (BMI) calculator implemented in Python. It uses the Linear Regression model from scikit-learn to predict BMI based on user inputs of height and weight. Additionally, it categorizes the predicted BMI into various health ranges.

Features

Accepts user input for height (in meters) and weight (in kilograms).

Uses a pre-trained linear regression model to predict BMI.

Classifies BMI into categories:

Underweight

Normal weight

Overweight

Obesity

Requirements

Ensure you have the following installed:

Python 3.7+

NumPy

scikit-learn

You can install the required libraries using pip:

pip install numpy scikit-learn

Usage

Clone the repository:

git clone https://github.com/yourusername/bmi_calculator_ml.git

Navigate to the project directory:

cd bmi_calculator_ml

Run the script:

python bmi_calculator_ml.py

Enter your height in meters and weight in kilograms when prompted. The script will output:

Your predicted BMI.

Your BMI category.

How It Works

Data Preparation: The model is trained on sample data containing height and weight pairs and their corresponding BMI values.

Model Training: A Linear Regression model is used to predict BMI based on height and weight.

BMI Calculation: Uses the formula:

BMI = weight / (height ** 2)

BMI Categorization: The predicted BMI is categorized into health ranges based on standard BMI classification.

Example

Input:

Enter your height in meters: 1.75
Enter your weight in kg: 70

Output:

Your predicted BMI is: 22.86
BMI Category: Normal weight

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contribution

Feel free to fork this repository, create a new branch, and submit a pull request for any features or improvements. Contributions are welcome!
