import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

def create_crash_prediction_model(data):
    # Create DataFrame from the data
    df = pd.DataFrame(data, columns=['make', 'crashes'])
    df['make'] = df['make'].str.strip().str.upper()
    
    
    label_encoder = LabelEncoder()
    df['make_encoded'] = label_encoder.fit_transform(df['make'])
    
   
    X = df[['make_encoded']]
    y = df['crashes']
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    mdl = LinearRegression()
    mdl.fit(X_train, y_train)
    
    
    y_pred = mdl.predict(X_test)
    
    # Calculate metrics
    mse_val = mse(y_test, y_pred)
    rmse = np.sqrt(mse_val)
    r2 = r2_score(y_test, y_pred)
    
    
    def predict_crashes(make):
        make = make.strip().upper()
        try:
            make_encoded = label_encoder.transform([make])
            prediction = mdl.predict([[make_encoded[0]]])
            return round(prediction[0], 2)
        except ValueError:
            return "Make not found in training data"
    
   
    performance_df = pd.DataFrame({
        'make': label_encoder.classes_,
        'actual_crashes': df.groupby('make')['crashes'].first().values,
        'predicted_crashes': mdl.predict(
            label_encoder.transform(label_encoder.classes_).reshape(-1, 1)
        )
    })
    
    
    performance_df = performance_df.sort_values('actual_crashes', ascending=False)
    
    return {
        'model': mdl,
        'label_encoder': label_encoder,
        'metrics': {
            'mse': mse_val,
            'rmse': rmse,
            'r2': r2
        },
        'predict_function': predict_crashes,
        'performance_df': performance_df
    }


data = [
    ['TOYOTA', 25973],
    ['HONDA', 21014],
    ['FORD', 18517],
    ['NISSAN', 9406],
    ['TOYT', 8841],
   
]


result = create_crash_prediction_model(data)


print("\nModel Metrics:")
print(f"R² Score: {result['metrics']['r2']:.4f}")
print(f"Root Mean Square Error: {result['metrics']['rmse']:.2f}")
print(f"Mean Square Error: {result['metrics']['mse']:.2f}")


print("\nTop Makes - Actual vs Predicted Crashes:")
print(result['performance_df'].head())


makes_to_predict = ["TOYOTA", "HONDA", "FORD"]
print("\nPredictions for specific makes:")
for make in makes_to_predict:
    prediction = result['predict_function'](make)
    print(f"{make}: {prediction} predicted crashes")


import matplotlib.pyplot as plt

def plot_actual_vs_predicted(performance_df):
    plt.figure(figsize=(12, 6))
    top_n = 10  
    
    makes = performance_df['make'].head(top_n)
    actual = performance_df['actual_crashes'].head(top_n)
    predicted = performance_df['predicted_crashes'].head(top_n)
    
    x = np.arange(len(makes))
    width = 0.35
    
    plt.bar(x - width/2, actual, width, label='Actual Crashes')
    plt.bar(x + width/2, predicted, width, label='Predicted Crashes')
    
    plt.xlabel('Make')
    plt.ylabel('Number of Crashes')
    plt.title('Actual vs Predicted Crashes by Vehicle Make')
    plt.xticks(x, makes, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_actual_vs_predicted(result['performance_df'])