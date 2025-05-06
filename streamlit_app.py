import traceback
import streamlit as st
import pandas as pd
from openai import OpenAI
from google.ads.googleads.client import GoogleAdsClient
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# ————————— Page config —————————
st.set_page_config(page_title="Creative Intelligence OS", layout="wide")

# ————————— Cached clients —————————
@st.cache_resource
def get_ads_client():
    cfg = {
        "developer_token": st.secrets.google_ads.developer_token,
        "use_proto_plus":  True,
        "client_id":       st.secrets.google_ads.client_id,
        "client_secret":   st.secrets.google_ads.client_secret,
        "refresh_token":   st.secrets.google_ads.refresh_token,
    }
    login_cid = st.secrets.google_ads.get("login_customer_id")
    if login_cid:
        cfg["login_customer_id"] = login_cid
    return GoogleAdsClient.load_from_dict(cfg)

@st.cache_data(ttl=3600)
def fetch_keyword_ideas(customer_id, seeds, lang, geos):
    client = get_ads_client()
    svc = client.get_service("KeywordPlanIdeaService")
    req = client.get_type("GenerateKeywordIdeasRequest")()
    req.customer_id = customer_id
    req.language = lang
    req.geo_target_constants.extend(geos)
    seed = client.get_type("KeywordSeed")()
    seed.keywords.extend(seeds)
    req.keyword_seed = seed

    resp = svc.generate_keyword_ideas(request=req)
    rows = []
    for idea in resp:
        m = idea.keyword_idea_metrics
        rows.append({
            "Keyword":       idea.text,
            "Searches/mo":   m.avg_monthly_searches,
            "Competition":   m.competition.name,
            "Low CPC ($)":   round(m.low_top_of_page_bid_micros  / 1e6, 2),
            "High CPC ($)":  round(m.high_top_of_page_bid_micros / 1e6, 2),
        })
    return pd.DataFrame(rows)

def detect_emotion(text: str) -> str:
    emotion_map = {
        "Fear":        ["urgent", "risk", "alert", "warning"],
        "Curiosity":   ["what", "why", "how", "?"],
        "Aspirational":["grow", "future", "dream", "success"],
        "Authority":   ["expert", "top", "proven", "official"],
    }
    t = str(text).lower()
    for emo, kws in emotion_map.items():
        if any(k in t for k in kws):
            return emo
    return "Neutral"

# ————————— Sidebar connection tests —————————
st.sidebar.markdown("## 🔌 Connection Tests")
if st.sidebar.button("Test Google Ads"):
    try:
        _ = get_ads_client()
        st.sidebar.success("✅ Google Ads client OK")
    except Exception:
        st.sidebar.error("Google Ads failed")
        st.sidebar.code(traceback.format_exc(), language="python")

if st.sidebar.button("Test OpenAI"):
    try:
        client = OpenAI(api_key=st.secrets.openai.api_key)
        _ = client.models.list()
        st.sidebar.success("✅ OpenAI OK")
    except Exception:
        st.sidebar.error("OpenAI failed")
        st.sidebar.code(traceback.format_exc(), language="python")

# ————————— Steps & state —————————
STEPS = [
    "Upload", "Scoring", "Keyword Planner",
    "Review + Annotate", "GPT Rewrite",
    "Clustering", "Export",
]
if "step" not in st.session_state:
    st.session_state.step = 0
# two distinct dataframes:
if "creatives_df" not in st.session_state:
    st.session_state.creatives_df = pd.DataFrame()
if "kws_df" not in st.session_state:
    st.session_state.kws_df = pd.DataFrame()

def next_step():
    st.session_state.step = min(st.session_state.step + 1, len(STEPS)-1)
def prev_step():
    st.session_state.step = max(st.session_state.step - 1, 0)

current = STEPS[st.session_state.step]
st.sidebar.title("Workflow")
for i, name in enumerate(STEPS):
    prefix = "▶️" if i == st.session_state.step else "  "
    st.sidebar.write(f"{prefix} {name}")
st.title(f"🧠 Step {st.session_state.step+1}: {current}")

# ————————— Step: Upload —————————
if current == "Upload":
    uploaded = st.file_uploader("Upload CSV with 'creative_text' or 'Text' column", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        for col in ["creative_text", "Creative text", "Text"]:
            if col in df.columns:
                df = df.rename(columns={col: "creative_text"})
                st.session_state.creatives_df = df
                st.success(f"Loaded {len(df)} rows from '{col}'")
                break
        else:
            st.error("CSV must include a 'creative_text', 'Creative text' or 'Text' column.")

# ————————— Step: Scoring —————————
elif current == "Scoring":
    df = st.session_state.creatives_df.copy()
    if df.empty:
        st.warning("Upload first.")
    else:
        df["score"] = df["creative_text"].str.len().mod(10).add(1)
        df["emotion"] = df["creative_text"].apply(detect_emotion)
        st.session_state.creatives_df = df
        st.dataframe(df, use_container_width=True)

# ————————— Step: Keyword Planner —————————
elif current == "Keyword Planner":
    st.markdown("Enter one seed keyword per line:")
    seeds = [s.strip() for s in st.text_area("", height=120).splitlines() if s.strip()]
    geo = st.selectbox("Geo", {"US":"geoTargetConstants/2840","UK":"geoTargetConstants/2826","CA":"geoTargetConstants/2124"}.items(), format_func=lambda x:x[0])[1]
    lang = st.selectbox("Language", {"EN":"1000","ES":"1003"}.items(), format_func=lambda x:x[0])[1]
    if st.button("Fetch Keyword Ideas"):
        if not seeds:
            st.warning("Enter at least one seed.")
        else:
            with st.spinner("Fetching…"):
                df_kws = fetch_keyword_ideas(st.secrets.google_ads.customer_id, seeds, lang, [geo])
            st.session_state.kws_df = df_kws
            st.success(f"Fetched {len(df_kws)} keywords.")
            st.dataframe(df_kws, use_container_width=True)

# ————————— Step: Review + Annotate —————————
elif current == "Review + Annotate":
    df = st.session_state.creatives_df
    if df.empty:
        st.warning("No creatives to review.")
    else:
        edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        st.session_state.creatives_df = edited

# ————————— Step: GPT Rewrite —————————
elif current == "GPT Rewrite":
    df = st.session_state.creatives_df.copy()
    if df.empty:
        st.warning("No creatives to rewrite.")
    else:
        client = OpenAI(api_key=st.secrets.openai.api_key)
        styles = {
            "Bold":"Polished & bold",
            "Snappy":"Short & punchy",
            "Empathetic":"Supportive tone",
            "Rude":"Blunt voice",
            "Inquisitive":"Question hook",
        }
        choice = st.selectbox("Style", list(styles))
        if st.button("Rewrite"):
            out_txt, out_reason = [], []
            prog = st.progress(0)
            for i, text in enumerate(df["creative_text"], 1):
                resp = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role":"system","content":"You are a creative ad writer."},
                        {"role":"user","content":f"{styles[choice]}\n\n{text}"}
                    ],
                    temperature=0.8)
                parts = resp.choices[0].message["content"].split("\n\n",1)
                out_txt.append(parts[0])
                out_reason.append(parts[1] if len(parts)>1 else "")
                prog.progress(i/len(df))
            df[f"rewrite_{choice}"] = out_txt
            df[f"reason_{choice}"]  = out_reason
            st.session_state.creatives_df = df
            st.dataframe(df, use_container_width=True)

# ————————— Step: Clustering —————————
elif current == "Clustering":
    df = st.session_state.creatives_df
    if df.empty:
        st.warning("No creatives to cluster.")
    else:
        X = TfidfVectorizer(max_features=50).fit_transform(df["creative_text"].astype(str))
        coords = PCA(n_components=3).fit_transform(X.toarray())
        df[["x","y","z"]] = coords
        fig = px.scatter_3d(df.head(50), x="x", y="y", z="z", text="creative_text", title="3D Clusters")
        st.plotly_chart(fig, use_container_width=True)

# ————————— Step: Export —————————
elif current == "Export":
    choice = st.selectbox("Export", ["Creatives","Keywords"])
    df = (st.session_state.creatives_df if choice=="Creatives" else st.session_state.kws_df)
    if df.empty:
        st.warning(f"No {choice.lower()} to export.")
    else:
        btn = st.download_button(f"Download {choice}", data=df.to_csv(index=False), file_name=f"{choice}.csv")

# ————————— Navigation —————————
col1, _, col3 = st.columns([1,2,1])
with col1:
    if st.button("← Back"): prev_step()
with col3:
    if st.button("Next →"): next_step()
