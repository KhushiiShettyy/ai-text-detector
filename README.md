# 🧠 AI Text Detector — Linguistic Forensics

A full-stack machine learning web app that classifies text as **human-written** or **AI-generated**, using explainable linguistic features rather than a black-box model. Built end-to-end: data pipeline, feature engineering, model training, Flask API, and a custom UI.

**Live demo:** _add your deployed link here once live_
**Screenshots:** _add 1-2 screenshots here_

---

## Why explainable features, not just "AI says so"

Instead of feeding raw text into a black-box classifier, this project extracts **9 measurable linguistic signals** — sentence rhythm, vocabulary richness, readability, punctuation density — and trains a Random Forest on those. Every prediction comes with the actual evidence behind it, shown to the user, not hidden in a model nobody can inspect.

## How it works

1. **Text submission** — the user pastes a passage into the UI
2. **Feature extraction** — 9 features are computed: average sentence length, burstiness (sentence length variance), vocabulary richness, average word length, Flesch reading ease, Flesch-Kincaid grade, punctuation density, word count, sentence count
3. **Random Forest classification** — 200 decision trees, trained on a combined dataset of student essays and diverse social/news/academic text (human vs. GPT-4o / Gemini 2.0 generated), each cast a vote
4. **Verdict + confidence** — the majority vote becomes the verdict; the margin becomes the confidence score, returned via a Flask API and rendered in the UI

## Tech stack

- **Backend:** Python, Flask
- **ML:** scikit-learn (Random Forest), textstat (readability scoring)
- **Frontend:** HTML, CSS, vanilla JavaScript
- **Data:** Kaggle — student essay corpus + diverse human/AI text (academic, news, social)

## Model performance

- **Test accuracy:** ~94% on held-out data
- Trained on a combined dataset of ~10,000 labeled passages across formal essays, news, academic, and social text

## Known limitations

This is a genuinely useful finding, not a footnote: **the model measures writing style, not authorship.** Polished, well-structured human writing (Wikipedia articles, professional journalism, careful non-native English) can score close to AI-generated text, because both tend to be structurally clean and grammatically consistent.

This is a documented, unsolved problem across commercial AI-detection tools (including products like Turnitin and GPTZero) — not a bug unique to this implementation. Feature-based detectors like this one pick up on *surface patterns* (sentence rhythm, vocabulary distribution) rather than deeper semantic understanding, which is the fundamental tradeoff of this lightweight approach versus a fine-tuned transformer model.

**Practical implication:** results should be treated as one signal, not proof. The UI includes an explicit disclaimer, and the "evidence" panel shows the underlying metrics so users can judge for themselves rather than trusting a single number.

## Running locally

\`\`\`bash
# clone the repo
git clone https://github.com/KhushiiShettyy/ai-text-detector.git
cd ai-text-detector

# create and activate a virtual environment
python -m venv venv
venv\\Scripts\\activate   # Windows
# source venv/bin/activate   # Mac/Linux

# install dependencies
pip install flask scikit-learn pandas numpy nltk transformers torch textstat joblib

# run the app
python app.py
\`\`\`

Then open `http://127.0.0.1:5000` in your browser.

**Note:** the trained model files (`model/saved_model.pkl`, `model/feature_columns.pkl`) are included in the repo, so you can run the app immediately without retraining. To retrain from scratch, download the datasets referenced in `model/train.py` and run `python model/train.py`.

## Project structure

\`\`\`
ai-text-detector/
├── app.py                    # Flask backend + prediction endpoint
├── model/
│   ├── train.py               # Feature extraction + Random Forest training
│   ├── saved_model.pkl        # Trained model
│   └── feature_columns.pkl    # Feature column order (for consistent inference)
├── static/
│   ├── style.css
│   └── script.js
├── templates/
│   └── index.html
└── data/                      # Datasets (not tracked in git — see .gitignore)
\`\`\`

## What I'd improve next

- Fine-tune a transformer model (e.g. DistilBERT) alongside the feature-based approach for a hybrid verdict
- Expand training data with more diverse, high-quality human writing across genres to reduce the polished-writing bias described above
- Add per-feature confidence weighting so the UI can show *which* signals drove a given verdict most strongly

---

Built as a portfolio project to explore explainable ML for text classification.