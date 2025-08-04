import streamlit as st
import pandas as pd
import joblib
import re

# For YouTube comments
from youtube_comment_downloader import YoutubeCommentDownloader

# Load your trained model and vectorizer
model = joblib.load('toxic_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\S+|@\\w+", "", text)
    text = re.sub(r"[^a-z\\s]", "", text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text


def get_comments_from_youtube(url, max_comments=100):
    video_id = url.split('v=')[-1]
    downloader = YoutubeCommentDownloader()
    comments = []
    for comment in downloader.get_comments_from_url(f"https://www.youtube.com/watch?v={video_id}"):
        if comment['text']:  # Only take non-empty comments
            comments.append(comment['text'])
        if len(comments) >= max_comments:
            break
    return comments


st.title("YouTube Top 10 Toxic Comments Extractor")

url = st.text_input("Paste a YouTube video URL:")

if st.button("Get Top 10 Toxic Comments"):
    if url:
        with st.spinner('Fetching comments...'):
            comments = get_comments_from_youtube(url, max_comments=200)
        if comments:
            df = pd.DataFrame({'comment': comments})
            df['clean'] = df['comment'].apply(clean_text)
            X = vectorizer.transform(df['clean'])
            probs = model.predict_proba(X)[:, 1]
            df['toxicity_score'] = probs
            df_top10 = df.sort_values(
                'toxicity_score', ascending=False).head(10)

            st.subheader("Top 10 Toxic Comments:")
            for idx, row in df_top10.iterrows():
                st.write(
                    f"**Toxicity Score: {row['toxicity_score']:.2f}** - {row['comment']}")
        else:
            st.warning("Could not fetch comments from that URL.")
    else:
        st.info("Please enter a valid YouTube URL.")

st.markdown("---")
st.caption("Made with Streamlit")
