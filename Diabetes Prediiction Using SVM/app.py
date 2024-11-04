from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

# Load the scaler and classifier (make sure to use your saved models)
scaler = pickle.load(open('models\scaler.pkl', 'rb'))
classifier = pickle.load(open('models\classifier.pkl', 'rb'))

# Create a Flask app instance
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Create a simple HTML form to input data

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    input_data = [
        float(request.form['Pregnancies']),
        float(request.form['Glucose']),
        float(request.form['BloodPressure']),
        float(request.form['SkinThickness']),
        float(request.form['Insulin']),
        float(request.form['BMI']),
        float(request.form['DiabetesPedigreeFunction']),
        float(request.form['Age'])
    ]
    
    # Convert input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    # Standardize the input data
    std_data = scaler.transform(input_data_as_numpy_array)

    # Predict the result
    prediction = classifier.predict(std_data)

    # Return the result as a response
    if prediction[0] == 0:
        result = "The Patient is not diabetic"
    else:
        result = "The patient is diabetic"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
