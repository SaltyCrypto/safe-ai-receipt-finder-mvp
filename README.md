# 🧠 Safe AI Receipt Finder – PRO Creative Analyzer

This Streamlit app is a full-featured Creative Intelligence OS built for marketers, performance creatives, and media buyers. It analyzes and scores marketing hooks using OpenAI's LLMs and embeddings, then visualizes, optimizes, and ranks them.

---

## 🚀 Features

- **Multi-Prompt Embedding & LLM Scoring**: Emotion, Persuasion, Monetization, Novelty, Trust, Urgency, Authority, Reward.
- **Weighted Ranking System**: Rank creatives with your own custom weights.
- **Clustering Visualizer**: Explore idea space with UMAP/PCA 2D plots.
- **Magic Rewrite Suggestions**: Auto-rewrite low-scoring creatives.
- **A/B Testing Simulator**: Compare two hooks side by side with GPT-4 insights.
- **Export Intelligence**: Download full CSV or JSON creative profiles.

---

## 📦 Installation

```bash
pip install -r requirements_pro.txt
streamlit run streamlit_app_pro.py
```

You'll need a valid [OpenAI API key](https://platform.openai.com/account/api-keys) to run the app.

---

## 📁 Input Format

Upload a `.csv` file with at least one column titled `Text`, containing your ad hooks or copy variations.

---

## ✅ Output

- `creative_intelligence.csv`: Scored and ranked hooks with topic-specific columns.
- `creative_profiles.json`: Vector embeddings and full metadata for each hook.

---

Built with ❤️ by [SaltyCrypto](https://github.com/SaltyCrypto)****
