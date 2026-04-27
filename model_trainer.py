import pandas as pd
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from processor import MovieProcessor

# --- 1. SETUP DIRECTORY ---
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(f"✅ Created directory: {MODEL_DIR}")

# --- 2. DATA LOADING ---
try:
    print("📂 Loading dataset...")
    # Ensure imdb.csv is in the same directory
    df = pd.read_csv('data/imdb.csv') 
except FileNotFoundError:
    print("❌ Error: imdb.csv not found. Please place the dataset in the project root.")
    exit()

# --- 3. PREPROCESSING ---
proc = MovieProcessor()
print("🧹 Cleaning text data (Applying Master's level NLP filters)...")
start_time = time.time()
df['review'] = df['review'].apply(proc.clean_text)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
print(f"⏱️ Preprocessing complete in {time.time() - start_time:.2f} seconds.")

# --- 4. VECTORIZATION (TF-IDF) ---
print("🔢 Vectorizing text (N-gram Range: 1-2)...")
# ngram_range=(1,2) captures both single words and pairs (e.g., "not good")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(df['review'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. MODEL SUITE ---
# Using SGDClassifier as a high-speed alternative to standard SVM
models = {
    'lr': LogisticRegression(max_iter=1000),
    'nb': MultinomialNB(),
    'xgb': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'rf': RandomForestClassifier(n_estimators=100, n_jobs=-1),
    'svm': SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000, tol=1e-3)
}

# --- 6. TRAINING & SAVING ---
print(f"🚀 Starting training suite for {len(models)} models...")

for name, model in models.items():
    m_start = time.time()
    print(f"Training {name.upper()}...", end=" ", flush=True)
    
    model.fit(X_train, y_train)
    
    # Save to the /models folder
    save_path = os.path.join(MODEL_DIR, f'{name}_model.pkl')
    joblib.dump(model, save_path)
    
    duration = time.time() - m_start
    print(f"Done! ({duration:.2f}s) 💾 Saved to {save_path}")

# Save the vectorizer to the same folder
joblib.dump(tfidf, os.path.join(MODEL_DIR, 'vectorizer.pkl'))

print("\n" + "="*40)
print("🎯 TRAINING SUCCESSFUL")
print(f"All models and the vectorizer are stored in: /{MODEL_DIR}")
print("You can now run: streamlit run app.py")
print("="*40)