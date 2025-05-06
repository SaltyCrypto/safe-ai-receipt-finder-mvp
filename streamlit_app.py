import traceback
import streamlit as st
import pandas as pd
import openai
from google.ads.googleads.client import GoogleAdsClient
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# ——————————————————————————————
# Page config
# ——————————————————————————————
st.set_page_config(page_title="Creative Intelligence OS", layout="wide")

# ——————————————————————————————
# Helpers & Cached Clients
# ——————————————————————————————
@st.cache_resource
def get_ads_client():
    cfg = {
        "developer_token":   st.secrets.google_ads.developer_token,
        "client_id":         st.secrets.google_ads.client_id,
        "client_secret":     st.secrets.google_ads.client_secret,
        "refresh_token":     st.secrets.google_ads.refresh_token,
        # use either customer_id or login_customer_id depending on your setup
        "login_customer_id": st.secrets.google_ads.get("login_customer_id", None),
    }
    return GoogleAdsClient.load_from_dict(cfg)

@st.cache_data(ttl=3600)
def fetch_keyword_ideas(customer_id, seed_keywords, language, geo_targets):
    client = get_ads_client()
    svc = client.get_service("KeywordPlanIdeaService")
    req = client.get_type("GenerateKeywordIdeasRequest")
    req.customer_id = customer_id
    req.language = language
    req.geo_target_constants.extend(geo_targets)

    seed = client.get_type("KeywordSeed")
    seed.keywords.extend(seed_keywords)
    req.keyword_seed = seed

    resp = svc.generate_keyword_ideas(request=req)
    rows = []
    for idea in resp:
        m = idea.keyword_idea_metrics
        rows.append({
            "Keyword":      idea.text,
            "Searches/mo":  m.avg_monthly_searches,
            "Competition":  m.competition.name,
            "Low CPC ($)":  round(m.low_top_of_page_bid_micros  / 1e6, 2),
            "High CPC ($)": round(m.high_top_of_page_bid_micros / 1e6, 2),
        })
    return pd.DataFrame(rows)

def detect_emotion(text: str) -> str:
    emotion_map = {
        "Fear":        ["urgent", "risk", "alert", "warning"],
        "Curiosity":   ["what", "why", "how", "did you know", "?"],
        "Aspirational":["grow", "future", "dream", "success"],
        "Authority":   ["expert", "top", "proven", "official"],
    }
    t = str(text).lower()
    for emo, kws in emotion_map.items():
        if any(k in t for k in kws):
            return emo
    return "Neutral"

# ——————————————————————————————
# Steps & Navigation
# ——————————————————————————————
STEPS = [
    "Upload",
    "Scoring",
    "Keyword Planner",
    "Review + Annotate",
    "GPT Rewrite",
    "Clustering",
    "Export",
]

# Initialize session state
if "step_idx" not in st.session_state:
    st.session_state.step_idx = 0
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

def go_next():
    st.session_state.step_idx = min(st.session_state.step_idx + 1, len(STEPS) - 1)

def go_back():
    st.session_state.step_idx = max(st.session_state.step_idx - 1, 0)

current_step = STEPS[st.session_state.step_idx]

# Sidebar navigation
st.sidebar.title("Steps")
for idx, name in enumerate(STEPS):
    if st.sidebar.button(name):
        st.session_state.step_idx = idx

st.title(f"🧠 Step {st.session_state.step_idx + 1}: {current_step}")

# ——————————————————————————————
# Step: Upload
# ——————————————————————————————
if current_step == "Upload":
    try:
        uploaded = st.file_uploader("Upload CSV with 'creative_text' column", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            if "creative_text" not in df.columns:
                st.error("CSV must include a 'creative_text' column.")
            else:
                st.session_state.df = df
                st.success(f"Loaded {len(df)} rows.")
    except Exception:
        st.error("Failed to load CSV:")
        st.code(traceback.format_exc())

# ——————————————————————————————
# Step: Scoring
# ——————————————————————————————
elif current_step == "Scoring":
    try:
        df = st.session_state.df.copy()
        if df.empty:
            st.warning("Upload data first.")
        else:
            df["score"] = df["creative_text"].str.len().mod(10).add(1)
            df["emotion_detected"] = df["creative_text"].apply(detect_emotion)
            st.session_state.df = df
            st.dataframe(df)
    except Exception:
        st.error("Scoring failed:")
        st.code(traceback.format_exc())

# ——————————————————————————————
# Step: Keyword Planner
# ——————————————————————————————
elif current_step == "Keyword Planner":
    try:
        st.markdown("Enter one keyword phrase per line:")
        keyword_input = st.text_area("Seed Keywords", height=120)
        geo = st.selectbox("Geo", {
            "United States": "geoTargetConstants/2840",
            "UK":             "geoTargetConstants/2826",
            "Canada":         "geoTargetConstants/2124",
        }.items(), format_func=lambda x: x[0])
        lang = st.selectbox("Language", {
            "English": "1000",
            "Spanish": "1003",
        }.items(), format_func=lambda x: x[0])

        if st.button("Fetch Keyword Ideas"):
            seeds = [w.strip() for w in keyword_input.splitlines() if w.strip()]
            if not seeds:
                st.warning("Please enter at least one keyword.")
            else:
                with st.spinner("Fetching ideas…"):
                    df_kws = fetch_keyword_ideas(
                        customer_id   = st.secrets.google_ads.customer_id,
                        seed_keywords = seeds,
                        language      = lang[1],
                        geo_targets   = [geo[1]],
                    )
                st.session_state.df = df_kws
                st.success(f"Retrieved {len(df_kws)} keywords.")
                st.dataframe(df_kws)
    except Exception:
        st.error("Keyword Planner error:")
        st.code(traceback.format_exc())

# ——————————————————————————————
# Step: Review + Annotate
# ——————————————————————————————
elif current_step == "Review + Annotate":
    if st.session_state.df.empty:
        st.warning("No data to review.")
    else:
        st.dataframe(st.session_state.df)

# ——————————————————————————————
# Step: GPT Rewrite
# ——————————————————————————————
elif current_step == "GPT Rewrite":
    try:
        df = st.session_state.df.copy()
        if df.empty or "creative_text" not in df.columns:
            st.warning("Upload or generate creatives first.")
        else:
            openai.api_key = st.secrets.openai.api_key
            styles = {
                "Bold":        "Make this copy sound polished and bold.",
                "Snappy":      "Rewrite short, punchy, attention-grabbing.",
                "Empathetic":  "Emotionally supportive, human tone.",
                "Rude":        "Blunt, no-nonsense voice.",
                "Inquisitive": "In the form of a curiosity-driven question.",
            }
            choice = st.selectbox("Rewrite Style", list(styles.keys()))
            if st.button("Rewrite with GPT"):
                rewritten, reasons = [], []
                progress = st.progress(0)
                total = len(df)
                for i, text in enumerate(df["creative_text"], 1):
                    try:
                        msgs = [
                            {"role": "system", "content": "You are a creative ad writer."},
                            {"role": "user",   "content": f"{styles[choice]}\n\nOriginal: {text}\n\nWhy better?"}
                        ]
                        resp = openai.ChatCompletion.create(
                            model="gpt-4", messages=msgs, temperature=0.8
                        )
                        out = resp.choices[0].message["content"]
                        parts = out.split("\n\n", 1)
                        rewritten.append(parts[0].strip())
                        reasons.append(parts[1].strip() if len(parts) > 1 else "—")
                    except Exception as e:
                        rewritten.append("ERROR")
                        reasons.append(str(e))
                    progress.progress(i / total)
                df[f"rewrite_{choice}"] = rewritten
                df[f"reason_{choice}"] = reasons
                st.session_state.df = df
                st.dataframe(df)
    except Exception:
        st.error("GPT Rewrite failed:")
        st.code(traceback.format_exc())

# ——————————————————————————————
# Step: Clustering
# ——————————————————————————————
elif current_step == "Clustering":
    try:
        df = st.session_state.df.copy()
        if df.empty or "creative_text" not in df.columns:
            st.warning("No creative texts to cluster.")
        else:
            vec = TfidfVectorizer(max_features=50)
            X = vec.fit_transform(df["creative_text"].astype(str))
            red = PCA(n_components=3).fit_transform(X.toarray())
            df[["x", "y", "z"]] = red
            fig = px.scatter_3d(
                df.head(50), x="x", y="y", z="z",
                text="creative_text", title="3D Clustering"
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.error("Clustering failed:")
        st.code(traceback.format_exc())

# ——————————————————————————————
# Step: Export
# ——————————————————————————————
elif current_step == "Export":
    try:
        df = st.session_state.df
        if df.empty:
            st.warning("Nothing to export.")
        else:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "Download CSV", data=csv_data,
                file_name="creative_output.csv"
            )
    except Exception:
        st.error("Export failed:")
        st.code(traceback.format_exc())

# ——————————————————————————————
# Navigation Buttons
# ——————————————————————————————
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("← Back"):
        go_back()
with col3:
    if st.button("Next →"):
        go_next()
