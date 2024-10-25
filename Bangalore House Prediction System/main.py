import pandas as pd
from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

data = pd.read_csv("Cleaned_Datasets/Banglore_Cleaned_Dataset.csv")
pipe = pickle.load(open("Models/RidgeModel_Price.pkl", "rb"))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    sqft = request.form.get('sqft')
    bath = request.form.get('bath')

    if not (location and bhk and sqft and bath):
        return jsonify({'error': 'All fields are required'})

    input_data = pd.DataFrame([[location, float(sqft), int(bath), int(bhk)]], 
                              columns=['location', 'total_sqft', 'bath', 'bhk'])
    try:
        prediction = pipe.predict(input_data)[0] * 1e6
        return jsonify({'prediction': f"₹{round(prediction, 1):,}"})
    except ValueError as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
