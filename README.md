# 🎬 Multi-Architectural Narrative Intelligence for Movie   Review Sentiment Analysis

### Multi-Architectural Sentiment Auditing & Market Forecasting  

A high-fidelity NLP diagnostic tool that utilizes a **"Jury of Models"** to interrogate movie reviews, visualize emotional trajectories, and forecast commercial success through ensemble consensus.

---
TEAM INFORMATION:
SWAATHI SHELVAPULLE,
ANKITHA AMMU,
BALA ARAVIND


## 🚀 Key Features  

### 🧠 Multi-Architecture Voting  
Side-by-side comparison of:  
- Logistic Regression  
- Naive Bayes  
- Support Vector Machine (SVM)  
- Random Forest  
- XGBoost  

---

### 📈 Sentiment Narrative Arc  
- Sentence-level sentiment tracking  
- Detects **Sentiment Reversals**  
- Visualizes emotional progression across a review  

---

### 🔍 Lexical Weighting  
- TF-IDF-based forensic analysis  
- Highlights **word-level impact on predictions**  
- Identifies influential sentiment drivers  

---

### 💼 Business Intelligence Ledger  
- Aggregates model outputs into a unified decision  
- Classifies films as:  
  - ✅ Certified Hit  
  - ❌ Box Office Flop  
- Built on **ensemble consensus logic**

---

## 🛠️ Setup and Execution Workflow  

### 1️⃣ Installation  


# Create Environment
```
python -m venv venv
```
# Activate (Mac/Linux)
```
source venv/bin/activate
```
# Activate (Windows)
```
venv\Scripts\activate
```
# Install Dependencies
```
pip install -r requirements.txt
```

### 📊 Dataset Requirement

This project utilizes the **IMDB Dataset of 50K Movie Reviews**. Due to GitHub's file size limitations, the raw CSV is not included in this repository. 

1. Download the dataset from Kaggle: [IMDB Dataset (50K Reviews)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
2. Rename the file to `data/imdb.csv` and place it in the project root directory.
3. Run `python3 processor.py` to begin the pipeline.


### Execution Order (CRITICAL)

The project must be run in the following sequence:

| Step | Action           | Command                     | Purpose                                                            |
|------|------------------|-----------------------------|--------------------------------------------------------------------|
| 1    | Data Processing  | `python3 processor.py`      | Cleans and preprocesses the dataset before training.               |
| 2    | Build Brain      | `python3 model_trainer.py`  | Trains models using processed data and generates `.pkl` files.     |
| 3    | Launch UI        | `streamlit run app.py`      | Opens the web interface (Light Mode recommended).                  |
| 4    | Analyze          | Use Tab 1                   | Paste a comment → click **Execute & Log** to analyze and store.    |
| 5    | Forecast         | Use Tab 2                   | Click **Predict Movie Verdict** to generate box office prediction. |

---

## Technical Requirements

- Python Version: 3.9+
- Core Libraries:
  - streamlit
  - pandas
  - numpy
  - scikit-learn
  - joblib
  - xgboost

---

## Project structure
```
├── app.py               # Streamlit dashboard (UI + visualization)
├── data/imdb.csv        # 50k kaggle IMDB Movie Dataset
├── processor.py         # NLP pipeline & preprocessing logic
├── model_trainer.py     # Training engine for ensemble models
├── models/              # Exported trained models (.pkl files)
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

## Notes

- `processor.py` MUST be run before model training.
- Ensure `.pkl` files are generated before launching the UI.
- Do not skip steps, or predictions may fail.
- Light Mode UI recommended for better visibility.
