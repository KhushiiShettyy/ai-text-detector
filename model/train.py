"""
train.py (v2)
Trains a classical ML model (Random Forest) to detect AI-generated text
using explainable linguistic features. Combines TWO datasets:
1. dataset.csv    -> columns: text, generated (0=human, 1=ai)   [student essays]
2. dataset2.csv   -> columns: text_id, label, source_model, domain, text_content, ...
                     label values: "human" / "ai"                [diverse: social, news, academic]

Combining both gives the model exposure to a wider range of writing styles,
reducing the "formal writing = AI" bias.
"""

import pandas as pd
import numpy as np
import textstat
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------- 1. Load Dataset 1 (essays) ----------
print("Loading dataset 1 (essays)...")
df1 = pd.read_csv("../data/dataset.csv")
df1 = df1.dropna(subset=["text", "generated"])
df1 = df1.rename(columns={"generated": "label"})
df1["label"] = df1["label"].astype(int)

if len(df1) > 8000:
    df1 = df1.sample(n=8000, random_state=42).reset_index(drop=True)

df1 = df1[["text", "label"]]

# ---------- 2. Load Dataset 2 (diverse: social/news/academic) ----------
print("Loading dataset 2 (diverse sources)...")
df2 = pd.read_csv("../data/dataset2.csv")
df2 = df2.dropna(subset=["text_content", "label"])

df2 = df2.rename(columns={"text_content": "text"})
df2["label"] = df2["label"].map({"human": 0, "ai": 1})
df2 = df2.dropna(subset=["label"])
df2["label"] = df2["label"].astype(int)

df2 = df2[["text", "label"]]

# ---------- 3. Combine Both Datasets ----------
df = pd.concat([df1, df2], ignore_index=True)
df = df.dropna(subset=["text", "label"])
df = df[df["text"].str.strip().str.len() > 0]

print(f"\nCombined dataset size: {len(df)} rows")
print(df["label"].value_counts())


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


print("\nExtracting features... (this may take a few minutes)")
feature_rows = []
for i, text in enumerate(df["text"]):
    feature_rows.append(extract_features(text))
    if i % 2000 == 0:
        print(f"  Processed {i}/{len(df)} rows...")

features_df = pd.DataFrame(feature_rows)
X = features_df
y = df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}\n")
print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))

joblib.dump(model, "saved_model.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")
print("\nModel saved to model/saved_model.pkl (overwritten with improved version)")