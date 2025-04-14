from flask import Flask, request, jsonify
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Ensure NLTK data is available
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
sentimentModel = joblib.load('sentiment_model.pkl')
tfidfVectorizer = joblib.load('tfidf_vectorizer.pkl')

# Setup preprocessing tools
stopWords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning function
def cleanText(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopWords]
    return " ".join(tokens)

# Define API route
@app.route('/predict', methods=['POST'])
def predictSentiment():
    data = request.get_json()

    if 'review' not in data:
        return jsonify({'error': 'Missing review text'}), 400

    cleanedReview = cleanText(data['review'])
    vectorizedReview = tfidfVectorizer.transform([cleanedReview])
    prediction = sentimentModel.predict(vectorizedReview)[0]
    sentiment = 'positive' if prediction == 1 else 'negative'

    return jsonify({'sentiment': sentiment})

# Run app
if __name__ == '__main__':
    app.run(debug=True)
