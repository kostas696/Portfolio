from flask import Flask, render_template, request
import pickle
from preprocessing import preprocess_text_spacy  # Import the preprocess_text_spacy function

app = Flask(__name__)

# Load the best model
with open('best_fine_tuned_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        # Preprocess the input text using SpaCy method
        preprocessed_text = preprocess_text_spacy(text)  # Use the imported function
        # Vectorize the preprocessed text using TF-IDF vectorizer
        X_text_tfidf = tfidf_vectorizer.transform([preprocessed_text])
        # Predict category
        predicted_category = model.predict(X_text_tfidf)[0]
        return render_template('result.html', text=text, category=predicted_category)

if __name__ == '__main__':
    app.run(debug=True)