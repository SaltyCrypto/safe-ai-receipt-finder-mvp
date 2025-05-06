import traceback
import streamlit as st
import pandas as pd
from openai import OpenAI
from google.ads.googleads.client import GoogleAdsClient
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# ————————————— Page config —————————————
st.set_page_config(page_title="Creative Intelligence OS", layout="wide")

# ————————————— Helpers & Clients —————————————
@st.cache_resource
def get_ads_client():
    cfg = {
        "developer_token":   st.secrets.google_ads.developer_token,
        "client_id":         st.secrets.google_ads.client_id,
        "client_secret":     st.secrets.google_ads.client_secret,
        "refresh_token":     st.secrets.google_ads.refresh_token,
        "login_customer_id": st.secrets.google_ads.get("login_customer_id", None),
    }
    return GoogleAdsClient.load_from_dict(cfg)

@st.cache_data(ttl=3600)
def fetch_keyword_ideas(customer_id, seeds, lang, geos):
    client = get_ads_client()
    svc = client.get_service("KeywordPlanIdeaService")
    req = client.get_type("GenerateKeywordIdeasRequest")(
        customer_id=customer_id,
        language=lang,
        geo_target_constants=geos,
        keyword_seed=client.get_type("KeywordSeed")(keywords=seeds),
    )
    resp = svc.generate_keyword_ideas(request=req)
    rows = [{
        "Keyword": idea.text,
        "Searches/mo": idea.keyword_idea_metrics.avg_monthly_searches,
        "Competition": idea.keyword_idea_metrics.competition.name,
        "Low CPC ($)": round(idea.keyword_idea_metrics.low_top_of_page_bid_micros/1e6,2),
        "High CPC ($)": round(idea.keyword_idea_metrics.high_top_of_page_bid_micros/1e6,2),
    } for idea in resp]
    return pd.DataFrame(rows)

def detect_emotion(txt: str) -> str:
    m = {
        "Fear": ["urgent","risk","alert","warning"],
        "Curiosity": ["what","why","how","?"],
        "Aspirational": ["grow","future","dream","success"],
        "Authority": ["expert","top","proven","official"],
    }
    t = txt.lower()
    for emo, kws in m.items():
        if any(k in t for k in kws):
            return emo
    return "Neutral"

# ————————————— Steps & State —————————————
STEPS = [
    "Upload",
    "Scoring",
    "Keyword Planner",
    "Review + Annotate",
    "GPT Rewrite",
    "Clustering",
    "Export",
]

if "step" not in st.session_state:
    st.session_state.step = 0
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

def next_step():
    st.session_state.step = min(st.session_state.step + 1, len(STEPS)-1)
def prev_step():
    st.session_state.step = max(st.session_state.step - 1, 0)

current = STEPS[st.session_state.step]

# ————————————— Navigation UI —————————————
st.sidebar.title("Steps")
for i, name in enumerate(STEPS):
    prefix = "▶️" if i == st.session_state.step else "  "
    st.sidebar.write(f"{prefix} {name}")
st.title(f"🧠 Step {st.session_state.step+1}: {current}")

# ————————————— Step: Upload —————————————
if current == "Upload":
    try:
        f = st.file_uploader("Upload CSV with 'creative_text', 'Creative text' or 'Text' column", type="csv")
        if f:
            df = pd.read_csv(f)
            found = None
            for col in ["creative_text","Creative text","Text"]:
                if col in df:
                    df = df.rename(columns={col:"creative_text"})
                    found = col
                    break
            if not found:
                st.error("CSV must include one of: creative_text, Creative text, Text.")
            else:
                st.session_state.df = df
                st.success(f"Loaded {len(df)} rows from '{found}'.")
    except Exception:
        st.error("Upload failed:")
        st.code(traceback.format_exc())

# ————————————— Step: Scoring —————————————
elif current == "Scoring":
    try:
        df = st.session_state.df.copy()
        if df.empty:
            st.warning("Please complete Upload first.")
        else:
            df["score"] = df["creative_text"].str.len().mod(10).add(1)
            df["emotion_detected"] = df["creative_text"].apply(detect_emotion)
            st.session_state.df = df
            st.dataframe(df)
    except Exception:
        st.error("Scoring failed:")
        st.code(traceback.format_exc())

# ————————————— Step: Keyword Planner —————————————
elif current == "Keyword Planner":
    try:
        st.markdown("Enter one seed keyword per line:")
        seeds = [w.strip() for w in st.text_area("", height=120).splitlines() if w.strip()]
        geo = st.selectbox("Geo",{"US":"geoTargetConstants/2840","UK":"geoTargetConstants/2826","CA":"geoTargetConstants/2124"}.items(), format_func=lambda x:x[0])[1]
        lang= st.selectbox("Language",{"English":"1000","Spanish":"1003"}.items(), format_func=lambda x:x[0])[1]
        if st.button("Fetch Keyword Ideas"):
            if not seeds:
                st.warning("Add at least one seed keyword.")
            else:
                with st.spinner("Fetching…"):
                    kws = fetch_keyword_ideas(st.secrets.google_ads.customer_id, seeds, lang, [geo])
                st.session_state.df = kws
                st.success(f"Retrieved {len(kws)} ideas.")
                st.dataframe(kws)
    except Exception:
        st.error("Keyword Planner error:")
        st.code(traceback.format_exc())

# ————————————— Step: Review + Annotate —————————————
elif current == "Review + Annotate":
    df = st.session_state.df
    if df.empty:
        st.warning("No data to review.")
    else:
        edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)
        st.session_state.df = edited

# ————————————— Step: GPT Rewrite —————————————
elif current == "GPT Rewrite":
    try:
        df = st.session_state.df.copy()
        if df.empty or "creative_text" not in df:
            st.warning("Run Upload/Planner first.")
        else:
            client = OpenAI(api_key=st.secrets.openai.api_key)
            styles = {
                "Bold":"Make this copy sound polished and bold.",
                "Snappy":"Short, punchy, attention-grabbing.",
                "Empathetic":"Emotionally supportive, human tone.",
                "Rude":"Blunt, no-nonsense voice.",
                "Inquisitive":"In the form of a curiosity-driven question.",
            }
            style = st.selectbox("Style", list(styles))
            if st.button("Rewrite with GPT"):
                out_rewrites, out_reasons = [], []
                prog = st.progress(0)
                for i, txt in enumerate(df["creative_text"],1):
                    try:
                        resp = client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role":"system","content":"You are a creative ad writer."},
                                {"role":"user","content":f"{styles[style]}\n\nOriginal: {txt}\n\nWhy better?"}
                            ],
                            temperature=0.8,
                        )
                        c = resp.choices[0].message["content"]
                        parts = c.split("\n\n",1)
                        out_rewrites.append(parts[0].strip())
                        out_reasons.append(parts[1].strip() if len(parts)>1 else "—")
                    except Exception as e:
                        out_rewrites.append("ERROR")
                        out_reasons.append(str(e))
                    prog.progress(i/len(df))
                df[f"rewrite_{style}"] = out_rewrites
                df[f"reason_{style}"] = out_reasons
                st.session_state.df = df
                st.dataframe(df)
    except Exception:
        st.error("GPT Rewrite failed:")
        st.code(traceback.format_exc())

# ————————————— Step: Clustering —————————————
elif current == "Clustering":
    try:
        df = st.session_state.df
        if df.empty or "creative_text" not in df:
            st.warning("No creatives available.")
        else:
            X = TfidfVectorizer(max_features=50).fit_transform(df["creative_text"])
            xyz = PCA(3).fit_transform(X.toarray())
            df[["x","y","z"]] = xyz
            fig = px.scatter_3d(df.head(50), x="x", y="y", z="z", text="creative_text")
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.error("Clustering failed:")
        st.code(traceback.format_exc())

# ————————————— Step: Export —————————————
elif current == "Export":
    df = st.session_state.df
    if df.empty:
        st.warning("Nothing to export.")
    else:
        st.download_button("Download CSV", data=df.to_csv(index=False), file_name="output.csv")

# ————————————— Navigation —————————————
col1, _, col3 = st.columns([1,2,1])
with col1:
    if st.button("← Back"):
        prev_step()
with col3:
    if st.button("Next →"):
        next_step()
