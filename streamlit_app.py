import streamlit as st
import pandas as pd
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.config import load_from_dict

st.set_page_config(page_title="Creative Intelligence OS", layout="wide")

# Inject custom sidebar font styles
st.markdown("""
    <style>
    [data-testid="stSidebar"] .css-1d391kg {
        font-size: 1.1rem;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        font-size: 1.3rem !important;
    }
    </style>
""", unsafe_allow_html=True)


if "step_idx" not in st.session_state:
    st.session_state.step_idx = 0

def go_next():
    if st.session_state.step_idx < len(steps) - 1:
        st.session_state.step_idx += 1

def go_back():
    if st.session_state.step_idx > 0:
        st.session_state.step_idx -= 1

current_step = steps[st.session_state.step_idx]
st.sidebar.title("🧭 Navigation")
st.sidebar.progress((st.session_state.step_idx + 1) / len(steps))
st.sidebar.radio("Jump to step", steps, index=st.session_state.step_idx, key="manual_step")

if st.session_state.manual_step != current_step:
    st.session_state.step_idx = steps.index(st.session_state.manual_step)
    current_step = st.session_state.manual_step

st.title(f"🧠 Step {st.session_state.step_idx + 1}: {current_step}")

if current_step == "Upload":
    file = st.file_uploader("📤 Upload CSV with 'creative_text' or 'Text' column", type="csv")
    if file:
        df = pd.read_csv(file)
        if "Text" in df.columns:
            df.rename(columns={"Text": "creative_text"}, inplace=True)
        if "creative_text" not in df.columns:
            st.error("Missing 'creative_text' column.")
        else:
            st.session_state.df = df
            st.success("✅ File uploaded!")
            st.dataframe(df)

elif current_step == "Scoring":
    if "df" in st.session_state:
        df = st.session_state.df.copy()
        df["score"] = df["creative_text"].apply(lambda x: len(str(x)) % 10 + 1)
        st.session_state.df = df
        st.success("✅ Creatives scored.")
        st.dataframe(df)
    else:
        st.warning("Upload data first.")

elif current_step == "Keyword Planner":
    config_dict = {
        "developer_token": st.secrets["google_ads"]["developer_token"],
        "client_id": st.secrets["google_ads"]["client_id"],
        "client_secret": st.secrets["google_ads"]["client_secret"],
        "refresh_token": st.secrets["google_ads"]["refresh_token"],
        "login_customer_id": "9816127168",  # Manager account ID
        "use_proto_plus": True
    }

    try:
        client = GoogleAdsClient.load_from_dict(config_dict)
    except Exception as e:
        st.error(f"Google Ads error: {e}")
        st.stop()

    customer_id = "2933192176"  # Newly created account

    keyword = st.text_input("💡 Seed keyword", "life insurance")
    geo = st.selectbox("🌍 Geo", {
        "United States": "geoTargetConstants/2840",
        "UK": "geoTargetConstants/2826",
        "Canada": "geoTargetConstants/2124"
    }.items())
    lang = st.selectbox("🈯 Language", {"English": "1000", "Spanish": "1003"}.items())

    if st.button("🔎 Fetch Keyword Ideas"):
        try:
            service = client.get_service("KeywordPlanIdeaService")
            request = client.get_type("GenerateKeywordIdeasRequest")
            request.customer_id = customer_id
            request.language = lang[1]
            request.geo_target_constants.append(geo[1])
            request.keyword_seed.keywords.append(keyword)

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
            st.success("✅ Keywords pulled")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Keyword error: {e}")

# Placeholder for remaining steps
elif current_step == "Explorer":
    if "df" in st.session_state:
        st.dataframe(st.session_state.df)
elif current_step == "Export":
    if "df" in st.session_state:
        st.download_button("⬇️ Download Results", data=st.session_state.df.to_csv(index=False), file_name="creative_output.csv")
elif current_step == "GPT Rewrite":

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.session_state.step_idx > 0:
        st.button("⬅️ Back", on_click=go_back)
with col3:
    if st.session_state.step_idx < len(steps) - 1:
        st.button("Next ➡️", on_click=go_next)

elif current_step == "Emotional Lens + Scoring":
    st.title("🎭 Emotional Lens + Emotion Scoring")
    if "df" in st.session_state:
        df = st.session_state.df.copy()

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
        df["suggested_rewrite"] = df["creative_text"].apply(lambda x: f"🔥 {x.strip()}")
        st.session_state.df = df
        st.success("✅ Emotional tone & rewrites applied")
        st.dataframe(df)
    else:
        st.warning("Upload creative content first.")

elif current_step == "Clustering":
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA

    st.title("📈 Creative Clustering")

    if "df" in st.session_state and "creative_text" in st.session_state.df.columns:
        df = st.session_state.df.copy()
        vectorizer = TfidfVectorizer(max_features=50)
        X = vectorizer.fit_transform(df["creative_text"].astype(str))
        reduced = PCA(n_components=2).fit_transform(X.toarray())
        df["x"] = reduced[:, 0]
        df["y"] = reduced[:, 1]
        st.session_state.df = df

        st.success("🧠 Text clustered using PCA")

        fig, ax = plt.subplots()
        ax.scatter(df["x"], df["y"])
        for i, txt in enumerate(df["creative_text"].head(50)):
            ax.annotate(txt[:20], (df["x"].iloc[i], df["y"].iloc[i]), fontsize=6)
        st.pyplot(fig)
    else:
        st.warning("No creative data found.")

elif current_step == "GPT Rewrite":
    import openai

    st.title("✍️ GPT-Powered Rewrite & Justification")
    openai.api_key = st.secrets["openai"]["api_key"]

    if "df" in st.session_state:
        df = st.session_state.df.copy()
        styles = {
            "Urgent CTA": "Rewrite this ad to sound more urgent and include a clear call to action.",
            "Conversational & Friendly": "Rewrite this ad in a conversational tone that feels warm and friendly.",
            "Authoritative & Expert": "Rewrite this ad to sound authoritative and professional, as if written by a subject matter expert.",
            "Curiosity Driven": "Rewrite this ad to make the user curious and want to learn more.",
            "Problem → Solution": "Rewrite this ad starting with the user's problem and ending with a solution."
        }

        style_choice = st.selectbox("🎨 Choose rewrite style", list(styles.keys()))
        prompt_template = styles[style_choice]

        if st.button("🔁 Rewrite with GPT"):
            rewritten, reasons = [], []
            for text in df["creative_text"]:
                try:
                    messages = [
                        {"role": "system", "content": "You are a copywriting assistant that rewrites ad texts using different tones and styles."},
                        {"role": "user", "content": f"{prompt_template}

Original: {text}

Also explain in 1 sentence why this rewrite might perform better."}
                    ]
                    response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=messages,
                        temperature=0.7
                    )
                    output = response.choices[0].message["content"]
                    if "

" in output:
                        split = output.split("

", 1)
                        rewritten.append(split[0].strip())
                        reasons.append(split[1].strip())
                    else:
                        rewritten.append(output.strip())
                        reasons.append("—")
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
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    st.title("📈 3D Clustering of Creative Text")

    if "df" in st.session_state and "creative_text" in st.session_state.df.columns:
        df = st.session_state.df.copy()
        vectorizer = TfidfVectorizer(max_features=50)
        X = vectorizer.fit_transform(df["creative_text"].astype(str))
        reduced = PCA(n_components=3).fit_transform(X.toarray())
        df["x"], df["y"], df["z"] = reduced[:, 0], reduced[:, 1], reduced[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df["x"], df["y"], df["z"], s=20)
        for i, txt in enumerate(df["creative_text"].head(25)):
            ax.text(df["x"].iloc[i], df["y"].iloc[i], df["z"].iloc[i], txt[:15], fontsize=7)
        st.pyplot(fig)
        st.success("✅ 3D visualization complete")
        st.session_state.df = df
    else:
        st.warning("Upload creative data to visualize.")
