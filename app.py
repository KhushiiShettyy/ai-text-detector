from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import re
import textstat
import os

app = Flask(__name__)

# Load trained model and feature columns
MODEL_PATH = os.path.join("model", "saved_model.pkl")
FEATURES_PATH = os.path.join("model", "feature_columns.pkl")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)


def sentence_split(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 0]


def extract_features(text):
    text = str(text)
    words = text.split()
    sentences = sentence_split(text)

    num_words = len(words) if len(words) > 0 else 1
    num_sentences = len(sentences) if len(sentences) > 0 else 1

    sentence_lengths = [len(s.split()) for s in sentences] if sentences else [0]
    avg_sentence_len = np.mean(sentence_lengths)
    std_sentence_len = np.std(sentence_lengths)

    unique_words = set(w.lower() for w in words)
    vocab_richness = len(unique_words) / num_words

    avg_word_len = np.mean([len(w) for w in words]) if words else 0

    try:
        flesch = textstat.flesch_reading_ease(text)
    except Exception:
        flesch = 0
    try:
        fk_grade = textstat.flesch_kincaid_grade(text)
    except Exception:
        fk_grade = 0

    punct_count = sum(1 for c in text if c in ".,;:!?")
    punct_density = punct_count / num_words

    return {
        "avg_sentence_len": avg_sentence_len,
        "std_sentence_len": std_sentence_len,
        "vocab_richness": vocab_richness,
        "avg_word_len": avg_word_len,
        "flesch_reading_ease": flesch,
        "flesch_kincaid_grade": fk_grade,
        "punct_density": punct_density,
        "num_words": num_words,
        "num_sentences": num_sentences,
    }


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()

    if len(text) < 20:
        return jsonify({'error': 'Please enter at least a few sentences for accurate analysis.'}), 400

    features = extract_features(text)
    feature_vector = [[features[col] for col in feature_columns]]

    prediction = model.predict(feature_vector)[0]
    probabilities = model.predict_proba(feature_vector)[0]

    confidence = round(float(max(probabilities)) * 100, 2)
    verdict = "AI-Generated" if prediction == 1 else "Human-Written"

    return jsonify({
        'verdict': verdict,
        'confidence': confidence,
        'reasoning': {
            'avg_sentence_length': round(features['avg_sentence_len'], 2),
            'burstiness': round(features['std_sentence_len'], 2),
            'vocabulary_richness': round(features['vocab_richness'], 3),
            'readability_score': round(features['flesch_reading_ease'], 2),
        }
    })


if __name__ == '__main__':
    app.run(debug=True)