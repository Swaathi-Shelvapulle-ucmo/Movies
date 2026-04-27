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
            'TFIDF': joblib.load(os.path.join(base_path, 'vectorizer.pkl'))
        }
    except Exception:
        st.error("Critical Error: Models not found in /models folder.")
        return None

models = load_all_assets()

# --- 3. HEADER ---
st.title("🎬 Movie Review Sentiment Analysis")
st.markdown("*Multi-Architectural Narrative Arcs & Business Intelligence.*")

tab1, tab2 = st.tabs(["🔍 Review NLP", "📈 Business Intelligence Ledger"])

# =========================================================================
# TAB 1: Review NLP
# =========================================================================
with tab1:
    user_input = st.text_area("Input for Movie Review:", height=150, placeholder="Paste a movie review here...")
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1: 
        execute = st.button("Run Diagnostic", use_container_width=True)
    with col_btn2: 
        if st.button("Clear Session"): 
            st.session_state['history'] = []; st.rerun()

    if execute and user_input and models:
        # A. PIPELINE
        cleaned = proc.clean_text(user_input)
        vec = models['TFIDF'].transform([cleaned])
        feature_names = models['TFIDF'].get_feature_names_out()
        
        # B. COMPARISON DATA
        scores = {name: float(models[name].predict_proba(vec)[0][1]) for name in ['LR', 'NB', 'SVM', 'RF', 'XGB']}
        mean_score = np.mean(list(scores.values()))
        detected_topics = proc.get_topics(user_input)
        
        # LOGGING
        log_entry = {"Comment": user_input[:50] + "...", "Mean": round(mean_score, 4)}
        log_entry.update({k: round(v, 4) for k, v in scores.items()})
        st.session_state['history'].append(log_entry)

        # --- 1. MULTI-ARCHITECTURE VOTING ---
        st.subheader("🧠 Multi-Architecture Voting Results")
        m_cols = st.columns(5)
        for i, (name, val) in enumerate(scores.items()):
            m_cols[i].metric(name, "POS" if val > 0.5 else "NEG", f"{val:.1%}")

        # --- 2. SENTIMENT NARRATIVE ARC (EMOTION ARC) ---
        st.markdown("---")
        st.subheader("📈 Sentiment Narrative Arc")
        st.info("💡 **Analysis Theory:** This arc tracks the emotional flow by analyzing every sentence independently. It reveals if the user started positive and ended negative (or vice versa), identifying 'Sentiment Reversals' that single-score models miss.")
        
        sentences = proc.split_sentences(user_input)
        if len(sentences) > 1:
            arc_data = [float(models['LR'].predict_proba(models['TFIDF'].transform([proc.clean_text(s)]))[0][1]) for s in sentences]
            fig_arc = px.line(x=range(1, len(arc_data)+1), y=arc_data, markers=True, range_y=[0,1], 
                             labels={'x':'Sentence Sequence', 'y':'Positivity Score'}, height=350)
            fig_arc.add_hline(y=0.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig_arc, use_container_width=True)
        else:
            st.warning("Narrative Arc requires at least 2 sentences to visualize emotional trajectory.")

        # --- 3. COMPARATIVE PROBABILITY DISTRIBUTION GRAPH ---
        st.markdown("---")
        st.subheader("📊 Architectural Probability Comparison")
        df_bar = pd.DataFrame(list(scores.items()), columns=['Model', 'Probability'])
        fig_bar = px.bar(df_bar, x='Model', y='Probability', color='Probability', 
                         color_continuous_scale='RdYlGn', range_y=[0,1], text_auto='.2f', height=400)
        fig_bar.add_hline(y=0.5, line_dash="dash", line_color="black")
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- 4. TOPICS & IMPACT ---
        st.markdown("---")
        col_lex1, col_lex2 = st.columns([1, 2])
        with col_lex1:
            st.subheader("🏷️ Topic Clusters")
            st.markdown(" ".join([f'<span class="topic-tag">{t}</span>' for t in detected_topics]), unsafe_allow_html=True)
        with col_lex2:
            st.subheader("⚖️ Lexical Weighting (Impact)")
            indices = vec.nonzero()[1]
            if len(indices) > 0:
                weights = models['LR'].coef_[0]
                impact_df = pd.DataFrame([{"Word": feature_names[i], "Impact": weights[i]} for i in indices]).sort_values(by="Impact", ascending=False)
                st.dataframe(impact_df.style.background_gradient(cmap='RdYlGn', subset=['Impact']), use_container_width=True, hide_index=True)

# =========================================================================
# TAB 2: BUSINESS INTELLIGENCE LEDGER
# =========================================================================
with tab2:
    st.header("📈 Business Intelligence Ledger")
    st.markdown("### Market Prediction Summary")
    st.write("Aggregated session data used to predict commercial viability and overall audience reception.")
    
    if not st.session_state['history']:
        st.info("No data processed. Run a diagnostic to populate the intelligence ledger.")
    else:
        history_df = pd.DataFrame(st.session_state['history'])
        avg_market = history_df['Mean'].mean()
        
        # Intelligence Predictor
        if avg_market >= 0.65:
            st.markdown('<div class="verdict-card" style="background-color: #d4edda; color: #155724;">OVERALL MARKET PREDICTION: CERTIFIED HIT 🔥</div>', unsafe_allow_html=True)
        elif avg_market <= 0.40:
            st.markdown('<div class="verdict-card" style="background-color: #f8d7da; color: #721c24;">OVERALL MARKET PREDICTION: BOX OFFICE FLOP 🧊</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="verdict-card" style="background-color: #fff3cd; color: #856404;">OVERALL MARKET PREDICTION: POLARIZED / MIXED ⚖️</div>', unsafe_allow_html=True)

        st.subheader("Architecture-Level Breakdown")
        model_cols = ['LR', 'NB', 'SVM', 'RF', 'XGB', 'Mean']
        st.dataframe(history_df.style.background_gradient(cmap='RdYlGn', subset=model_cols), use_container_width=True, hide_index=True)