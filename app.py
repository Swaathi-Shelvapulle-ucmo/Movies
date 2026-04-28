import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import os
from processor import MovieProcessor

# --- 1. UI CONFIGURATION ---
st.set_page_config(page_title="Movie Review NLP", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #000000; }
    div[data-testid="stMetric"] { background-color: #f8f9fa; border: 1px solid #e9ecef; padding: 20px; border-radius: 12px; }
    .verdict-card { padding: 40px; border-radius: 20px; text-align: center; font-size: 32px; font-weight: 800; border: 3px solid #000000; margin: 25px 0; }
    .topic-tag { background-color: #e1e4e8; padding: 4px 10px; border-radius: 12px; margin-right: 5px; font-size: 13px; color: #24292e; border: 1px solid #d1d5da; font-weight: 600; }
    .explanation-box { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #000000; margin-bottom: 20px; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ASSETS ---
if 'history' not in st.session_state: st.session_state['history'] = []
proc = MovieProcessor()

@st.cache_resource
def load_all_assets():
    base_path = "models"
    try:
        return {
            'LR': joblib.load(os.path.join(base_path, 'lr_model.pkl')),
            'XGB': joblib.load(os.path.join(base_path, 'xgb_model.pkl')),
            'NB': joblib.load(os.path.join(base_path, 'nb_model.pkl')),
            'RF': joblib.load(os.path.join(base_path, 'rf_model.pkl')),
            'SVM': joblib.load(os.path.join(base_path, 'svm_model.pkl')), 
            'TFIDF': joblib.load(os.path.join(base_path, 'vectorizer.pkl')),
            'METRICS': pd.read_csv(os.path.join(base_path, 'evaluation_metrics.csv')) if os.path.exists(os.path.join(base_path, 'evaluation_metrics.csv')) else None
        }
    except Exception:
        st.error("Critical Error: Models or Metrics not found. Please run model_trainer.py first.")
        return None

models = load_all_assets()

# --- 3. HEADER ---
st.title("🎬 Movie Review Sentiment Analysis")
st.markdown("*Master's Project: Multi-Architectural Narrative Intelligence*")

# --- 4. MODEL ENCYCLOPEDIA ---
with st.expander("📖 Model Encyclopedia: How do these algorithms differ?"):
    st.markdown("""
    This project utilizes a **Heterogeneous Ensemble** approach. Each model "sees" the text differently:
    * **Logistic Regression (LR):** A linear baseline assigning mathematical weights to specific words.
    * **Naive Bayes (NB):** Calculates sentiment based on word frequency probabilities.
    * **SVM (SGD):** Finds the optimal hyperplane to separate linguistic boundaries.
    * **Random Forest (RF):** Uses an ensemble of 100 Decision Trees to reduce variance.
    * **XGBoost (XGB):** A Gradient Boosting framework that builds trees sequentially to correct errors.
    """)

tab1, tab2, tab3 = st.tabs(["🔍 Review NLP", "📈 Business Intelligence", "🧪 Model Research"])

# =========================================================================
# TAB 1: Review NLP
# =========================================================================
with tab1:
    user_input = st.text_area("Input for Movie Review:", height=150, placeholder="Paste a movie review here...")
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1: 
        execute = st.button("Run Diagnostic", width="stretch")
    with col_btn2: 
        if st.button("Clear Session"): 
            st.session_state['history'] = []; st.rerun()

    if execute and user_input and models:
        cleaned = proc.clean_text(user_input)
        vec = models['TFIDF'].transform([cleaned])
        feature_names = models['TFIDF'].get_feature_names_out()
        
        # Inferences
        scores = {name: float(models[name].predict_proba(vec)[0][1]) for name in ['LR', 'NB', 'SVM', 'RF', 'XGB']}
        mean_score = np.mean(list(scores.values()))
        detected_topics = proc.get_topics(user_input)
        
        # Log History
        log_entry = {"Comment": user_input[:50] + "...", "Mean": round(mean_score, 4)}
        log_entry.update({k: round(v, 4) for k, v in scores.items()})
        st.session_state['history'].append(log_entry)

        # 1. ENSEMBLE VOTING
        st.subheader("🧠 Multi-Architecture Voting Results")
        st.markdown('<div class="explanation-box">Each model casts a "vote" based on its internal logic. The <b>Mean Score</b> acts as the final ensemble decision.</div>', unsafe_allow_html=True)
        m_cols = st.columns(5)
        for i, (name, val) in enumerate(scores.items()):
            m_cols[i].metric(name, "POS" if val > 0.5 else "NEG", f"{val:.1%}")

        # 2. PROBABILITY COMPARISON (RE-ADDED)
        st.markdown("---")
        st.subheader("📊 Architectural Probability Comparison")
        st.markdown('<div class="explanation-box">This graph compares the <b>confidence level</b> of each model. A higher probability indicates the model is more certain the review is Positive.</div>', unsafe_allow_html=True)
        df_bar = pd.DataFrame(list(scores.items()), columns=['Model', 'Probability'])
        fig_bar = px.bar(df_bar, x='Model', y='Probability', color='Probability', 
                         color_continuous_scale='RdYlGn', range_y=[0,1], text_auto='.2f', height=400)
        fig_bar.add_hline(y=0.5, line_dash="dash", line_color="black")
        st.plotly_chart(fig_bar, width="stretch")

        # 3. SENTIMENT NARRATIVE ARC
        st.markdown("---")
        st.subheader("📈 Sentiment Narrative Arc")
        st.markdown('<div class="explanation-box"><b>Theory:</b> This arc breaks the review into sentences to see how the mood shifts. It reveals "Sentiment Reversals" where a reviewer changes their mind mid-text.</div>', unsafe_allow_html=True)
        sentences = proc.split_sentences(user_input)
        if len(sentences) > 1:
            arc_data = [float(models['LR'].predict_proba(models['TFIDF'].transform([proc.clean_text(s)]))[0][1]) for s in sentences]
            fig_arc = px.line(x=range(1, len(arc_data)+1), y=arc_data, markers=True, range_y=[0,1], 
                             labels={'x':'Sentence Sequence', 'y':'Positivity Score'}, height=350)
            fig_arc.add_hline(y=0.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig_arc, width="stretch")
        else:
            st.warning("Narrative Arc requires at least 2 sentences.")

        # 4. TOPICS & IMPACT
        st.markdown("---")
        col_lex1, col_lex2 = st.columns([1, 2])
        with col_lex1:
            st.subheader("🏷️ Topic Clusters")
            st.markdown('<div class="explanation-box">Identifies thematic areas based on keyword extraction.</div>', unsafe_allow_html=True)
            st.markdown(" ".join([f'<span class="topic-tag">{t}</span>' for t in detected_topics]), unsafe_allow_html=True)
        with col_lex2:
            st.subheader("⚖️ Lexical Weighting (Impact)")
            st.markdown('<div class="explanation-box">Shows which specific words influenced the model verdict.</div>', unsafe_allow_html=True)
            indices = vec.nonzero()[1]
            if len(indices) > 0:
                weights = models['LR'].coef_[0]
                impact_df = pd.DataFrame([{"Word": feature_names[i], "Impact": weights[i]} for i in indices]).sort_values(by="Impact", ascending=False)
                st.dataframe(impact_df.style.background_gradient(cmap='RdYlGn', subset=['Impact']), width="stretch", hide_index=True)

# =========================================================================
# TAB 2: BUSINESS INTELLIGENCE
# =========================================================================
with tab2:
    st.header("📈 Business Intelligence Ledger")
    st.markdown('<div class="explanation-box">Aggregated session data used to predict overall commercial viability.</div>', unsafe_allow_html=True)
    
    if not st.session_state['history']:
        st.info("No data processed yet.")
    else:
        history_df = pd.DataFrame(st.session_state['history'])
        avg_market = history_df['Mean'].mean()
        
        if avg_market >= 0.65:
            st.markdown('<div class="verdict-card" style="background-color: #d4edda; color: #155724;">OVERALL MARKET PREDICTION: CERTIFIED HIT 🔥</div>', unsafe_allow_html=True)
        elif avg_market <= 0.40:
            st.markdown('<div class="verdict-card" style="background-color: #f8d7da; color: #721c24;">OVERALL MARKET PREDICTION: BOX OFFICE FLOP 🧊</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="verdict-card" style="background-color: #fff3cd; color: #856404;">OVERALL MARKET PREDICTION: POLARIZED ⚖️</div>', unsafe_allow_html=True)

        st.subheader("Aggregated Data Breakdown")
        st.dataframe(history_df.style.background_gradient(cmap='RdYlGn', subset=['Mean']), width="stretch", hide_index=True)

# =========================================================================
# TAB 3: MODEL RESEARCH
# =========================================================================
with tab3:
    st.header("🧪 Technical Research & Evaluation")
    if models['METRICS'] is not None:
        st.subheader("📊 Comparative Performance Metrics")
        st.table(models['METRICS'])
    else:
        st.info("Run model_trainer.py to generate evaluation_metrics.csv.")

    st.markdown("---")
    st.subheader("⚙️ System Architecture")
    st.markdown("""
    **Pipeline Workflow:**
    1.  **Ingestion:** UI-driven text input.
    2.  **Preprocessing:** Sanitization via `processor.py`.
    3.  **Vectorization:** TF-IDF with N-gram (1,2) range.
    4.  **Ensemble Inference:** Concurrent prediction across five architectures.
    5.  **Analytics:** Plotly-driven narrative and probability visualization.
    """)
