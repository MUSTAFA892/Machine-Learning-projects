from flask import Flask, request, render_template
import numpy as np
import pickle

classifier = pickle.load(open('models/heart.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [
        float(request.form['Age']),
        float(request.form['Sex']),
        float(request.form['CP']),
        float(request.form['Trestbps']),
        float(request.form['Chol']),
        float(request.form['FBS']),
        float(request.form['RestECG']),
        float(request.form['Thalach']),
        float(request.form['Exang']),
        float(request.form['Oldpeak']),
        float(request.form['Slope']),
        float(request.form['CA']),
        float(request.form['Thal'])
    ]
    
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    prediction = classifier.predict(input_data_as_numpy_array)

    if prediction[0] == 0:
        result = "The patient is not at risk of heart disease."
    else:
        result = "The patient is at risk of heart disease."

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
