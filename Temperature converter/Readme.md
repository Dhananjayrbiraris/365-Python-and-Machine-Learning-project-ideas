# Temperature Converter Application

This repository contains a Python-based Temperature Converter application implemented in a Kaggle notebook. The application uses `ipywidgets` to create an interactive user interface for temperature conversion between Celsius and Fahrenheit.

## Features

- **Two-way Conversion**: Convert temperatures from Celsius to Fahrenheit and vice versa.
- **Interactive UI**: Simple and intuitive interface built with `ipywidgets`.
- **Real-time Results**: Instant temperature conversion upon input and selection.

## Requirements

To run this application, ensure you have the following installed:

- Python (>=3.6)
- Jupyter Notebook or Kaggle Notebook environment
- `ipywidgets` library

Install the `ipywidgets` library if it's not already installed:
```bash
pip install ipywidgets
```

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/temperature-converter.git
   cd temperature-converter
   ```

2. **Run the Notebook**:
   Open the Kaggle notebook or any Jupyter notebook environment and copy-paste the code into a new cell.

3. **Interact with the UI**:
   - Enter the temperature value in the input box.
   - Select the conversion type from the dropdown menu (Celsius to Fahrenheit or Fahrenheit to Celsius).
   - Click the **Convert** button to see the result.

4. **View Results**:
   The converted temperature will be displayed below the Convert button.

## Code Overview

The application consists of:

1. **Conversion Functions**:
   - `celsius_to_fahrenheit(celsius)`
   - `fahrenheit_to_celsius(fahrenheit)`

2. **UI Elements**:
   - Input box for temperature value.
   - Dropdown menu for selecting the conversion type.
   - Button to trigger the conversion.
   - Label to display the result.

3. **Event Handling**:
   - The `convert_temperature` function handles user interactions and updates the result in real-time.

## Example Screenshot

![Temperature Converter](path-to-screenshot.png)

## Contribution

Feel free to fork this repository and contribute by submitting pull requests. Suggestions and feedback are welcome!

---

Happy Coding!

