import pandas as pd
import re
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("news.csv")

# Drop missing values
df = df[['text', 'label']].dropna()

# -------------------------------
# 🔥 BALANCE DATASET
# -------------------------------
real = df[df['label'] == 'REAL']
fake = df[df['label'] == 'FAKE']

min_len = min(len(real), len(fake))

real = real.sample(min_len, random_state=42)
fake = fake.sample(min_len, random_state=42)

df = pd.concat([real, fake])

print("Balanced Data:")
print(df['label'].value_counts())

# -------------------------------
# CLEAN TEXT
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['text'] = df['text'].apply(clean_text)

# -------------------------------
# SPLIT DATA
# -------------------------------
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# TF-IDF (IMPROVED)
# -------------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# MODEL (UPGRADED)
# -------------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# -------------------------------
# EVALUATION
# -------------------------------
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nDetailed Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# SAVE MODEL
# -------------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved!")