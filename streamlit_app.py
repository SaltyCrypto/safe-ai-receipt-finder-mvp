# streamlit_app.py
# 🚀 Safe AI Receipt Finder – PRO Creative Analyzer (v6.3)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from openai import OpenAI

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
        "client": None,
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
st.sidebar.progress(step_index / (len(steps) - 1))

# Global top progress bar
with st.container():
    st.markdown("#### Progress")
    st.progress(step_index / (len(steps) - 1))

# --- API Key Input ---
st.sidebar.markdown("---")
if st.session_state.valid_key:
    if st.sidebar.button("🔁 Test OpenAI Ping"):
        try:
            models = st.session_state.client.models.list()
            st.sidebar.success(f"✅ Connected. {len(models.data)} models available.")
        except Exception as e:
            st.sidebar.error(f"❌ Ping failed: {e}")

st.sidebar.subheader("🔐 API Access")
api_input = st.sidebar.text_input("Enter OpenAI API Key", type="password").strip()
if api_input:
    try:
        client = OpenAI(api_key=api_input)
        st.session_state.api_key = api_input
        st.session_state.valid_key = True
        st.session_state.client = client
        st.sidebar.success("✅ API Key Valid")
    except Exception as e:
        st.sidebar.error(f"❌ API Error: {str(e)}")

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
            st.success("✅ Creatives scored successfully!")

        if 'score' in st.session_state.df:
            cols = ['creative_text', 'score'] if 'score' in st.session_state.df.columns else ['creative_text']
            st.dataframe(st.session_state.df[cols])
            avg_score = st.session_state.df['score'].mean()
            st.plotly_chart(go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_score,
                title={"text": "Average Creative Score"},
                gauge={"axis": {"range": [0, 10]}}
            )), use_container_width=True)
            st.button("Next: Explore Embeddings", disabled='score' not in st.session_state.df, on_click=lambda: st.session_state.update({"step": "Explorer"}))
        else:
            st.warning("No scores found yet. Please click 'Score Creatives' first.")

# --- Step: Explorer ---
elif st.session_state["step"] == "Explorer":
    st.title("🧭 Embedding Explorer")
    if st.session_state.df is not None:
        if st.session_state["x"] is None:
            st.session_state.df['x'] = np.random.randn(len(st.session_state.df))
            st.session_state.df['y'] = np.random.randn(len(st.session_state.df))

        required_cols = {'x', 'y', 'creative_text'}
        if required_cols.issubset(st.session_state.df.columns) and not st.session_state.df.empty:
            df = st.session_state.df.dropna(subset=['x', 'y'])
            if len(df) == 0:
                st.warning("No valid rows to plot after removing missing coordinates.")
            else:
                try:
                    if 'score' in df.columns:
                        fig = px.scatter(df, x='x', y='y', color='score', hover_data=['creative_text'])
                    else:
                        st.warning("No scores found. You can continue exploring embeddings without them or go back to score.")
                        fig = px.scatter(df, x='x', y='y', hover_data=['creative_text'])
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Plotting failed: {e}")
        else:
            st.error("❌ Required data is missing or corrupted. Please check your uploaded file.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Back to Scoring"):
                st.session_state["step"] = "Scoring"
        with col2:
            if st.button("Next: Cluster Creatives"):
                st.session_state["step"] = "Clustering"

# --- Step: Clustering ---
elif st.session_state["step"] == "Clustering":
    st.title("📚 Topic Clustering Dashboard")
    if st.session_state.df is not None:
        required_cols = {'x', 'y', 'creative_text'}
        if required_cols.issubset(st.session_state.df.columns) and not st.session_state.df.empty:
            k = st.slider("Number of clusters", 2, 10, 4)
            st.session_state.df['cluster'] = np.random.randint(0, k, size=len(st.session_state.df))
            fig = px.scatter(st.session_state.df, x='x', y='y', color='cluster', hover_data=['creative_text'])
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(st.session_state.df[['creative_text', 'cluster']])
            st.info("Each color group represents a cluster of similar creative themes.")
        else:
            st.error("❌ Required data is missing or corrupted. Please go back to scoring and embedding steps.")

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
    if st.button("Rewrite & Explain") and st.session_state.valid_key:
        client = st.session_state.client
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a creative marketing assistant."},
                    {"role": "user", "content": f"Rewrite this creative: {raw}"}
                ]
            )
            rewritten = response.choices[0].message.content.strip()
            st
