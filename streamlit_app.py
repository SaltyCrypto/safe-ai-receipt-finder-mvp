# ==== TAB 3: Emotion Scoring ====
with tab3:
    st.header("🎯 Rank Creatives by Target Emotion")

    uploaded_score = st.file_uploader("📤 Upload Embedded CSV with Emotion", type=["csv"], key="score_emotion")

    if uploaded_score:
        df = pd.read_csv(uploaded_score)

        if 'Predicted Emotion' not in df.columns or 'Text' not in df.columns:
            st.error("🚫 CSV must have 'Predicted Emotion' and 'Text' columns!")
        else:
            emotions = ['Fear', 'Hope', 'Curiosity', 'Love', 'Greed', 'Excitement', 'Pride', 'Anger', 'Envy', 'Other']
            target = st.selectbox("🎯 Choose target emotion", emotions)

            def score_match(predicted, target):
                if pd.isna(predicted): return 0.0
                predicted = predicted.strip().lower()
                target = target.strip().lower()
                if predicted == target:
                    return 1.0
                elif target in predicted or predicted in target:
                    return 0.5
                else:
                    return 0.0

            df['Emotion Score'] = df['Predicted Emotion'].apply(lambda e: score_match(e, target))
            df_sorted = df.sort_values(by='Emotion Score', ascending=False)

            st.success(f"🎯 Ranked by how closely creatives match: **{target}**")
            st.dataframe(df_sorted[['Text', 'Predicted Emotion', 'Emotion Score']].head(20))

            csv = df_sorted.to_csv(index=False)
            st.download_button(
                label="📥 Download Ranked CSV",
                data=csv,
                file_name=f'ranked_creatives_{target.lower()}.csv',
                mime='text/csv'
            )
