import pandas as pd
import joblib
import os
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
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
    df = pd.read_csv('data/imdb.csv') 
except FileNotFoundError:
    print("❌ Error: data/imdb.csv not found.")
    exit()

# --- 3. PREPROCESSING ---
proc = MovieProcessor()
print("🧹 Cleaning text data (Applying NLP filters)...")
start_time = time.time()
df['review'] = df['review'].apply(proc.clean_text)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
print(f"⏱️ Preprocessing complete in {time.time() - start_time:.2f}s")

# --- 4. VECTORIZATION ---
print("🔢 Vectorizing text (N-gram Range: 1-2)...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(df['review'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. ADVANCED HYPERPARAMETER TUNING (Requirement: Technical Depth) ---
print("🚀 Tuning XGBoost via GridSearchCV (this may take a minute)...")
xgb_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.1, 0.01]
}
# Removed deprecated use_label_encoder to prevent warnings
grid_xgb = GridSearchCV(XGBClassifier(eval_metric='logloss'), 
                        xgb_params, cv=3, n_jobs=-1, scoring='f1')
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
print(f"✅ Best Params: {grid_xgb.best_params_}")

# --- 6. MODEL SUITE ---
models = {
    'lr': LogisticRegression(max_iter=1000),
    'nb': MultinomialNB(),
    'xgb': best_xgb, # Using the optimized model
    'rf': RandomForestClassifier(n_estimators=100, n_jobs=-1),
    'svm': SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000, tol=1e-3)
}

# --- 7. TRAINING, EVALUATION & SAVING ---
print(f"🧪 Training suite and generating Master's Level metrics...")
performance_data = []

for name, model in models.items():
    m_start = time.time()
    model.fit(X_train, y_train)
    
    # Generate Evaluation Metrics (Requirement: Precision, Recall, F1)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    performance_data.append({
        'Model': name.upper(),
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(report['weighted avg']['precision'], 4),
        'Recall': round(report['weighted avg']['recall'], 4),
        'F1-Score': round(report['weighted avg']['f1-score'], 4)
    })
    
    # Save the model
    joblib.dump(model, os.path.join(MODEL_DIR, f'{name}_model.pkl'))
    print(f"Done training {name.upper()} ({time.time() - m_start:.2f}s)")

# --- 8. SAVE METRICS & VECTORIZER ---
# Save the metrics to CSV for the App's "Research" Tab
metrics_df = pd.DataFrame(performance_data)
metrics_df.to_csv(os.path.join(MODEL_DIR, 'evaluation_metrics.csv'), index=False)
joblib.dump(tfidf, os.path.join(MODEL_DIR, 'vectorizer.pkl'))

print("\n" + "="*40)
print("🎯 TRAINING & EVALUATION SUCCESSFUL")
print(f"Metrics saved to: {MODEL_DIR}/evaluation_metrics.csv")
print("You can now run 'streamlit run app.py' to view Tab 3.")
print("="*40)
