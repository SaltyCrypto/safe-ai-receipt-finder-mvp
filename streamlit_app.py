import streamlit as st
import pandas as pd
import openai
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.config import load_from_dict
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Page config and style
st.set_page_config(page_title="Creative Intelligence OS", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] {
    font-family: 'Segoe UI', sans-serif;
    font-size: 1.25rem;
    padding: 1rem;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-weight: 700;
    color: #3A3A3A;
}
</style>
    <style>
    [data-testid="stSidebar"] .css-1d391kg {
        font-size: 1.1rem;
        padding-left: 0.5rem;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        font-size: 1.3rem !important;
    }
    </style>
""", unsafe_allow_html=True)

steps = [
    "Upload",
    "Scoring",
    "Keyword Planner",
    "Review + Annotate",
    "GPT Rewrite",
    "Clustering",
    "Export"
]
steps = ["Upload", "Scoring", "Keyword Planner", "Explorer", "Emotional Lens + Scoring", "GPT Rewrite", "Clustering", "Export"]

if "step_idx" not in st.session_state:
st.session_state.step_idx = 0
@@ -57,7 +43,6 @@ def go_back():

st.title(f"🧠 Step {st.session_state.step_idx + 1}: {current_step}")

# Now begin with proper structure
if current_step == "Upload":
file = st.file_uploader("📤 Upload CSV with 'creative_text' or 'Text' column", type="csv")
if file:
@@ -75,21 +60,8 @@ def go_back():
if "df" in st.session_state:
df = st.session_state.df.copy()
df["score"] = df["creative_text"].apply(lambda x: len(str(x)) % 10 + 1)
        emotion_map = {
            "Fear": ["urgent", "risk", "alert", "warning"],
            "Curiosity": ["what", "why", "how", "did you know", "?"],
            "Aspirational": ["grow", "future", "dream", "success"],
            "Authority": ["expert", "top", "proven", "official"]
        }
        def detect_emotion(text):
            t = str(text).lower()
            for emotion, keywords in emotion_map.items():
                if any(k in t for k in keywords):
                    return emotion
            return "Neutral"
        df["emotion_detected"] = df["creative_text"].apply(detect_emotion)
st.session_state.df = df
        st.success("✅ Creatives scored and emotion-tagged.")
        st.success("✅ Creatives scored.")
st.dataframe(df)
else:
st.warning("Upload data first.")
@@ -107,123 +79,50 @@ def detect_emotion(text):
try:
client = GoogleAdsClient.load_from_dict(config_dict)
except Exception as e:
        st.error(f"❌ Google Ads client load failed: {e}")
        st.error(f"Google Ads error: {e}")
st.stop()

customer_id = "2933192176"
    st.markdown("Enter one keyword phrase per line:")
    keyword_input = st.text_area("💡 Seed Keywords", "life insurance\nterm life insurance\nbest policy for parents")

    geo = st.selectbox("🌍 Geo Target", {
    keyword = st.text_input("💡 Seed keyword", "life insurance")
    geo = st.selectbox("🌍 Geo", {
"United States": "geoTargetConstants/2840",
"UK": "geoTargetConstants/2826",
"Canada": "geoTargetConstants/2124"
    }.items(), format_func=lambda x: x[0])
    }.items())
    lang = st.selectbox("🈯 Language", {"English": "1000", "Spanish": "1003"}.items())

    lang = st.selectbox("🈯 Language", {"English": "1000", "Spanish": "1003"}.items(), format_func=lambda x: x[0])

    if st.button("🚀 Fetch Keyword Ideas"):
    if st.button("🔎 Fetch Keyword Ideas"):
try:
            keyword_plan_idea_service = client.get_service("KeywordPlanIdeaService")
            service = client.get_service("KeywordPlanIdeaService")
request = client.get_type("GenerateKeywordIdeasRequest")
request.customer_id = customer_id
request.language = lang[1]
request.geo_target_constants.append(geo[1])
            keywords = [kw.strip() for kw in keyword_input.split("\n") if kw.strip()]
            if not keywords:
                st.warning("⚠️ Please enter at least one keyword phrase.")
                st.stop()
keyword_seed = client.get_type("KeywordSeed")
            keyword_seed.keywords.extend(keywords)
            keyword_seed.keywords.append(keyword)
request.keyword_seed = keyword_seed
            response = keyword_plan_idea_service.generate_keyword_ideas(request=request)
            result = []
            for idea in response:
                metrics = idea.keyword_idea_metrics
                result.append({
                    "Keyword": idea.text,
                    "Searches/mo": metrics.avg_monthly_searches,
                    "Competition": metrics.competition.name,
                    "Low CPC ($)": round(metrics.low_top_of_page_bid_micros / 1_000_000, 2),
                    "High CPC ($)": round(metrics.high_top_of_page_bid_micros / 1_000_000, 2)
                })

            ideas = service.generate_keyword_ideas(request=request)
            result = [{
                "Keyword": idea.text,
                "Searches": idea.keyword_idea_metrics.avg_monthly_searches,
                "Competition": idea.keyword_idea_metrics.competition.name,
                "Low CPC": idea.keyword_idea_metrics.low_top_of_page_bid_micros,
                "High CPC": idea.keyword_idea_metrics.high_top_of_page_bid_micros
            } for idea in ideas]
df = pd.DataFrame(result)
st.session_state.df = df
            st.success("✅ Keyword ideas retrieved")
            st.success("✅ Keywords pulled")
st.dataframe(df)
except Exception as e:
            st.error(f"❌ Keyword Planner error: {e}")

elif current_step == "Review + Annotate":
    if "df" in st.session_state:
        st.markdown("🗂 Use this view to review, filter or tag your generated creatives.")
        st.dataframe(st.session_state.df)

elif current_step == "GPT Rewrite":
    openai.api_key = st.secrets["openai"]["api_key"]
    if "df" in st.session_state:
        df = st.session_state.df.copy()
        styles = {
            "Bold": "Make this copy sound polished and bold.",
            "Snappy": "Rewrite this ad to be short, punchy, and attention-grabbing.",
            "Empathetic": "Rewrite this in an emotionally supportive, human tone.",
            "Rude": "Rewrite this with a blunt, no-nonsense voice.",
            "Inquisitive": "Rewrite this in the form of a question or curiosity-driven hook."
        }
        style_choice = st.selectbox("🎨 Choose rewrite style", list(styles.keys()))
        prompt_template = styles[style_choice]

        if st.button("🔁 Rewrite with GPT"):
            rewritten, reasons = [], []
            for text in df["creative_text"]:
                try:
                    messages = [
                        {"role": "system", "content": "You are a creative ad writer."},
                        {"role": "user", "content": f"{prompt_template}\n\nOriginal: {text}\n\nExplain in one sentence why it might perform better."}
                    ]
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=messages,
                        temperature=0.8
                    )
                    content = response.choices[0].message["content"]
                    parts = content.split("\n\n", 1)
                    rewritten.append(parts[0].strip())
                    reasons.append(parts[1].strip() if len(parts) > 1 else "—")
                except Exception as e:
                    rewritten.append("ERROR")
                    reasons.append(str(e))

            df[f"rewrite_{style_choice}"] = rewritten
            df[f"reason_{style_choice}"] = reasons
            st.session_state.df = df
            st.success("✅ Rewrites completed")
            st.dataframe(df)
    else:
        st.warning("Upload creative content first.")

elif current_step == "Clustering":
    if "df" in st.session_state and "creative_text" in st.session_state.df.columns:
        df = st.session_state.df.copy()
        vectorizer = TfidfVectorizer(max_features=50)
        X = vectorizer.fit_transform(df["creative_text"].astype(str))
        reduced = PCA(n_components=3).fit_transform(X.toarray())
        df["x"], df["y"], df["z"] = reduced[:, 0], reduced[:, 1], reduced[:, 2]
        fig = px.scatter_3d(df.head(15), x="x", y="y", z="z", text="creative_text", title="3D Creative Clustering")
        st.plotly_chart(fig)
        st.session_state.df = df
    else:
        st.warning("Upload creative data to visualize.")

elif current_step == "Export":
    if "df" in st.session_state:
        st.download_button("⬇️ Download Results", data=st.session_state.df.to_csv(index=False), file_name="creative_output.csv")
            st.error(f"Keyword error: {e}")

# Navigation
# Navigation buttons
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
if st.session_state.step_idx > 0:
st.button("⬅️ Back", on_click=go_back)
with col3:
if st.session_state.step_idx < len(steps) - 1:
        st.button("Next ➡️", on_click=go_next)
        st.button("Next ➡️", on_click=go_next)