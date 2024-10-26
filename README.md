# Vehicle Crash Analysis Model

## Overview
This Python script implements a linear regression model to analyze and predict vehicle crashes based on manufacturer data. The model processes historical crash data by manufacturer, trains a prediction model, and provides tools for estimating future crashes for specific manufacturers.

## Features
- Data preprocessing and cleaning
- Manufacturer name normalization
- Linear regression model training
- Crash prediction capabilities
- Statistical analysis and model evaluation
- Visualization of real vs predicted crashes

## Requirements
The following Python packages are required:
```
pandas
numpy
scikit-learn
matplotlib
```

## Installation
1. Ensure Python 3.x is installed on your system
2. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

## Usage

### Basic Usage
```python
# Import the analyzer
from crash_analyzer import analyze_vehicle_crashes

# Prepare your data
vehicle_crashes = [
    ['TOYOTA', 25973],
    ['HONDA', 21014],
    ['FORD', 18517],
    ['NISSAN', 9406],
    ['TOYT', 8841],
]

# Generate analysis
output = analyze_vehicle_crashes(vehicle_crashes)

# Get predictions for specific manufacturers
estimate = output['estimator']('TOYOTA')
```

### Data Format
Input data should be a list of lists, where each inner list contains:
- Manufacturer name (string)
- Number of incidents (integer)

### Output Structure
The function returns a dictionary containing:
- `predictor`: Trained LinearRegression model
- `converter`: LabelEncoder for manufacturer names
- `scores`: Dictionary of accuracy metrics (MSE, RMSE, R²)
- `estimator`: Function for making new predictions
- `stats`: DataFrame comparing real vs predicted crashes

## Functions

### analyze_vehicle_crashes(input_data)
Main analysis function that processes crash data and builds the prediction model.

#### Parameters:
- `input_data`: List of [manufacturer, incidents] lists

#### Returns:
Dictionary containing model components and analysis results

### show_comparison(stats_df)
Visualization function that creates a bar chart comparing real vs predicted crashes.

#### Parameters:
- `stats_df`: DataFrame containing comparison statistics

## Model Evaluation
The model's performance is evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score (Coefficient of determination)

## Visualization
The script includes a visualization function that generates a bar chart comparing:
- Actual crash numbers
- Predicted crash numbers
- Top 10 manufacturers by crash volume

## Limitations
- Requires consistent manufacturer naming
- Limited to historical data patterns
- Assumes linear relationship between variables
- Prediction accuracy depends on training data quality

## Example Output
```python
# Accuracy Metrics
R² Score: 0.8563
Root Mean Square Error: 1234.56
Mean Square Error: 1524123.45

# Manufacturer Comparison
manufacturer  real_crashes  predicted_crashes
TOYOTA       25973         24789.32
HONDA        21014         20456.78
FORD         18517         17998.45
```


## License
This project is licensed under the MIT License - see the LICENSE file for details.
