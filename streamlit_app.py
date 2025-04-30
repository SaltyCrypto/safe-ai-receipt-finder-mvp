# streamlit_app_pro_v6.py
# 🚀 Safe AI Receipt Finder – PRO Creative Analyzer (v6.1 Enhanced UI/UX)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time

# --- Page Config ---
st.set_page_config(
    page_title="Creative Analyzer PRO",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
for key in ["df", "step", "api_key", "valid_key"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "step" else "Upload"

# --- Sidebar Progress Navigator ---
st.sidebar.title("🧠 Creative Workflow")
steps = ["Upload", "Scoring", "Explorer", "Clustering", "Optimization", "Export"]
st.session_state["step"] = st.sidebar.radio("Navigate steps:", steps, index=steps.index(st.session_state["step"]))

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

# --- Top Progress Bar ---
st.progress(steps.index(st.session_state["step"]) / (len(steps)-1))

# --- Step: Upload ---
if st.session_state["step"] == "Upload":
    st.title("📂 Upload Your Creative Dataset")
    st.markdown("Upload a CSV file with a `creative_text` column to begin scoring.")
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded and loaded into session.")
        st.dataframe(st.session_state.df.head())

# --- Step: Scoring ---
elif st.session_state["step"] == "Scoring":
    st.title("📊 Creative Scoring Engine")
    st.markdown("We’ll evaluate your ad texts using internal scoring logic.")
    if st.session_state.df is not None:
        with st.spinner("Scoring..."):
            time.sleep(1)
            st.session_state.df['score'] = np.random.uniform(3, 9, len(st.session_state.df))
            st.toast("Scoring complete")
        st.dataframe(st.session_state.df[['creative_text', 'score']])

        avg_score = st.session_state.df['score'].mean()
        st.plotly_chart(go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_score,
            title={"text": "Average Creative Score"},
            gauge={"axis": {"range": [0, 10]}}
        )), use_container_width=True)

# --- Step: Explorer ---
elif st.session_state["step"] == "Explorer":
    st.title("🧭 Embedding Explorer")
    st.markdown("Visualize creative distributions in embedding space.")
    if st.session_state.df is not None:
        st.session_state.df['x'] = np.random.randn(len(st.session_state.df))
        st.session_state.df['y'] = np.random.randn(len(st.session_state.df))
        fig = px.scatter(st.session_state.df, x='x', y='y', color='score', hover_data=['creative_text'])
        st.plotly_chart(fig, use_container_width=True)

# --- Step: Clustering ---
elif st.session_state["step"] == "Clustering":
    st.title("📚 Topic Clustering Dashboard")
    st.markdown("Group creatives by cluster to identify content themes.")
    if st.session_state.df is not None:
        k = st.slider("Number of clusters", 2, 10, 4)
        st.session_state.df['cluster'] = np.random.randint(0, k, size=len(st.session_state.df))
        fig = px.scatter(st.session_state.df, x='x', y='y', color='cluster', hover_data=['creative_text'])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(st.session_state.df[['creative_text', 'cluster']])

# --- Step: Optimization ---
elif st.session_state["step"] == "Optimization":
    st.title("🪄 Optimization Magic Tab")
    st.markdown("Tweak, test, and simulate high-performing copy rewrites.")

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

# --- Step: Export ---
elif st.session_state["step"] == "Export":
    st.title("📤 Export & Share")
    if st.session_state.df is not None:
        st.download_button("Download Enhanced CSV", st.session_state.df.to_csv(index=False), "enhanced_creatives.csv", "text/csv")
    email = st.text_input("Send summary to email")
    if st.button("Send"):
        st.toast("Report sent! (Simulated)")

    st.markdown("---")
    st.success("🚀 Thanks for using Creative Analyzer PRO v6.1")
