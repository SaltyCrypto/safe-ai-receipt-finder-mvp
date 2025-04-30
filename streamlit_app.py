# streamlit_app_pro_v6.py
# 🚀 Safe AI Receipt Finder – PRO Creative Analyzer (v6.2 Guided Workflow)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

st.set_page_config(
    page_title="Creative Analyzer PRO",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
def init_state():
    defaults = {
        "df": None,
        "step": "Upload",
        "api_key": None,
        "valid_key": False,
        "x": None,
        "y": None,
        "cluster": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_state()

# --- Sidebar Workflow Navigator ---
st.sidebar.title("🧠 Creative Workflow")
steps = ["Upload", "Scoring", "Explorer", "Clustering", "Optimization", "Export"]
step_index = steps.index(st.session_state["step"])
st.sidebar.markdown(f"### ▶️ Step {step_index + 1}: {steps[step_index]}")
st.progress(step_index / (len(steps) - 1))

# --- API Key Input ---
st.sidebar.subheader("🔐 API Access")
api_input = st.sidebar.text_input("Enter OpenAI API Key", type="password")
if api_input:
    try:
        import openai
        openai.Model.list(api_key=api_input)
        st.session_state.api_key = api_input
        st.session_state.valid_key = True
        st.sidebar.success("✅ API Key Valid")
    except:
        st.sidebar.error("❌ Invalid API Key")

# --- Step: Upload ---
if st.session_state["step"] == "Upload":
    st.title("📂 Upload Your Creative Dataset")
    uploaded_file = st.file_uploader("Upload a CSV with a `creative_text` column", type=["csv"])
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("✅ File loaded successfully!")
        st.markdown(f"**{len(st.session_state.df)} hooks uploaded.**")
        with st.expander("📋 Preview uploaded data (first 10 rows)"):
            st.dataframe(st.session_state.df.head(10))
        if st.button("Next: Score Creatives"):
            st.session_state["step"] = "Scoring"

# --- Step: Scoring ---
elif st.session_state["step"] == "Scoring":
    st.title("📊 Creative Scoring Engine")
    if st.session_state.df is not None:
        if st.button("Score Creatives"):
            with st.spinner("Scoring..."):
                time.sleep(1)
                st.session_state.df['score'] = np.random.uniform(3, 9, len(st.session_state.df))
                st.toast("Scoring complete")
        if 'score' in st.session_state.df:
            st.dataframe(st.session_state.df[['creative_text', 'score']])
            avg_score = st.session_state.df['score'].mean()
            st.plotly_chart(go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_score,
                title={"text": "Average Creative Score"},
                gauge={"axis": {"range": [0, 10]}}
            )), use_container_width=True)
            if st.button("Next: Explore Embeddings"):
                st.session_state["step"] = "Explorer"

# --- Step: Explorer ---
elif st.session_state["step"] == "Explorer":
    st.title("🧭 Embedding Explorer")
    if st.session_state.df is not None:
        if st.session_state["x"] is None:
            st.session_state.df['x'] = np.random.randn(len(st.session_state.df))
            st.session_state.df['y'] = np.random.randn(len(st.session_state.df))
        fig = px.scatter(st.session_state.df, x='x', y='y', color='score', hover_data=['creative_text'])
        st.plotly_chart(fig, use_container_width=True)
        if st.button("Next: Cluster Creatives"):
            st.session_state["step"] = "Clustering"

# --- Step: Clustering ---
elif st.session_state["step"] == "Clustering":
    st.title("📚 Topic Clustering Dashboard")
    if st.session_state.df is not None:
        k = st.slider("Number of clusters", 2, 10, 4)
        st.session_state.df['cluster'] = np.random.randint(0, k, size=len(st.session_state.df))
        fig = px.scatter(st.session_state.df, x='x', y='y', color='cluster', hover_data=['creative_text'])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(st.session_state.df[['creative_text', 'cluster']])
        if st.button("Next: Optimize Creatives"):
            st.session_state["step"] = "Optimization"

# --- Step: Optimization ---
elif st.session_state["step"] == "Optimization":
    st.title("🪄 Optimization Magic Tab")
    st.subheader("🏆 A/B Creative Match")
    col1, col2 = st.columns(2)
    with col1:
        a = st.text_area("Creative A", "Affordable life insurance in under 60 seconds")
    with col2:
        b = st.text_area("Creative B", "Protect your family starting at $5/month")
    if st.button("Simulate Winner"):
        winner = a if np.random.rand() > 0.5 else b
        st.success(f"🏁 Simulated Winner: {winner[:50]}...")

    st.subheader("💬 Rewrite & Explanation")
    raw = st.text_area("Paste creative for rewrite:", "Get $300 off your first year")
    if st.button("Rewrite & Explain"):
        st.chat_message("system").write("Analyzing...")
        st.chat_message("assistant").write("✨ Rewritten: Claim $300 savings today – no paperwork!")
        st.info("Why: This version increases urgency and removes friction words.")

    if st.button("Next: Export Results"):
        st.session_state["step"] = "Export"

# --- Step: Export ---
elif st.session_state["step"] == "Export":
    st.title("📤 Export & Share")
    if st.session_state.df is not None:
        st.download_button("Download Enhanced CSV", st.session_state.df.to_csv(index=False), "enhanced_creatives.csv", "text/csv")
    email = st.text_input("Send summary to email")
    if st.button("Send"):
        st.toast("Report sent! (Simulated)")
    st.success("🎉 You've completed the full creative analysis workflow!")
