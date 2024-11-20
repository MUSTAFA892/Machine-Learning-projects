from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Initialize Flask app
app = Flask(__name__)

# Load books data
books = pd.read_csv("Datasets/Books.csv")  # Ensure the file is in the same directory

# Create a mapping of book titles to their poster URLs
book_posters = dict(zip(books['Book-Title'], books['Image-URL-M']))

# Load your preprocessed data and model
book_pivot = pd.read_pickle("book_pivot.pkl")  # Replace with your dataset
model = NearestNeighbors(n_neighbors=6, algorithm='brute')
model.fit(book_pivot)

# Recommendation function
def recommend_book(book_name):
    try:
        book_id = np.where(book_pivot.index == book_name)[0][0]
        distances, suggestions = model.kneighbors(
            book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6
        )
        # Return a list of dictionaries containing book titles and poster URLs
        return [{"title": book_pivot.index[i], "poster": book_posters.get(book_pivot.index[i], "#")} for i in suggestions[0]]
    except IndexError:
        return [{"title": "Book not found", "poster": "#"}]


# Routes
@app.route('/')
def index():
    return render_template('index.html')  # Render the input form

@app.route('/recommend', methods=['POST'])
def recommend():
    book_name = request.form['book_name']  # Get the book name from the form
    recommendations = recommend_book(book_name)  # Get recommendations
    return render_template(
        'recommendations.html', book_name=book_name, recommendations=recommendations
    )


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
