# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import plotly.express as px
import nltk
import warnings
warnings.filterwarnings("ignore")

nltk.download("stopwords")
from nltk.corpus import stopwords

st.set_page_config(page_title="YouTube Trending Forecast App", layout="wide")
st.title("📈 YouTube Trending Video View Forecasting")
st.markdown("""
This app helps forecast views of trending YouTube videos using **ARIMA time series modeling**, and provides **keyword analysis**, **word clouds**, and **channel-level comparisons**.
""")

uploaded_file = st.file_uploader("Upload YouTube Trending Data (e.g., USvideos.csv)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    st.write("📋 Columns in Dataset:", df.columns.tolist())

    if 'trending_date' not in df.columns or 'video_id' not in df.columns or 'views' not in df.columns:
        st.error("❌ Required columns missing: Ensure the CSV contains 'trending_date', 'video_id', 'views', etc.")
        st.stop()

    try:
        df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')
    except:
        df['trending_date'] = pd.to_datetime(df['trending_date'], errors='coerce')

    df = df.dropna(subset=['trending_date'])
    df['text'] = df['title'].fillna('') + ' ' + df['tags'].fillna('')
    df = df.drop_duplicates(subset=['video_id', 'trending_date'])

    st.subheader("📊 Dataset Overview")
    st.dataframe(df.head(10))

    top_videos = (
        df.groupby(['video_id', 'title'])['views']
        .max().reset_index()
        .sort_values(by='views', ascending=False).head(10)
    )

    video_option = st.selectbox("Select a Trending Video for Forecasting", options=top_videos['title'])
    selected_video_id = top_videos[top_videos['title'] == video_option]['video_id'].values[0]

    video_df = df[df['video_id'] == selected_video_id].sort_values('trending_date')
    video_df = video_df.drop_duplicates(subset='trending_date')

    ts_df = video_df[['trending_date', 'views']].set_index('trending_date')

    st.subheader("📈 View Count Over Time")
    st.line_chart(ts_df)

    st.markdown("""
    ---
    ### 🔧 ARIMA Time Series Forecast
    Forecasting future views based on past trend.
    """)

    train = ts_df[:-3]
    test = ts_df[-3:]

    model = ARIMA(train, order=(2, 1, 2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=3)

    rmse = np.sqrt(mean_squared_error(test, forecast))
    forecast_df = pd.DataFrame({"Actual": test["views"], "Forecast": forecast}, index=test.index)

    st.dataframe(forecast_df)
    st.metric("📉 RMSE", f"{rmse:,.2f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train.index, train["views"], label="Train", color='blue')
    ax.plot(test.index, test["views"], label="Actual", color='green', marker='o')
    ax.plot(test.index, forecast, label="Forecast", color='red', linestyle='--', marker='x')
    ax.legend()
    ax.set_title("ARIMA Forecast vs Actual Views")
    st.pyplot(fig)

    st.markdown("""
    ---
    ## 🧠 NLP Analysis on Titles and Tags
    Analyze frequent keywords and generate word cloud.
    """)

    stop_words = stopwords.words("english")
    vectorizer = CountVectorizer(stop_words=stop_words, max_features=50)
    X = vectorizer.fit_transform(df["text"])
    word_freq = X.sum(axis=0).A1
    keywords = vectorizer.get_feature_names_out()

    keyword_df = pd.DataFrame({"Keyword": keywords, "Frequency": word_freq})
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=keyword_df.sort_values(by="Frequency", ascending=False), x="Frequency", y="Keyword", palette="viridis", ax=ax2)
    ax2.set_title("Top Keywords in Titles + Tags")
    st.pyplot(fig2)

    st.markdown("### ☁️ Word Cloud")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['text'].values))
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")
    st.pyplot(fig_wc)

    st.markdown("""
    ---
    ## 📺 Channel-level View Trends
    Compare viewership across channels.
    """)

    if 'channel_title' in df.columns:
        channel_summary = df.groupby('channel_title')['views'].sum().reset_index().sort_values(by='views', ascending=False).head(10)
        fig3 = px.bar(channel_summary, x='channel_title', y='views', title='Top 10 Channels by Views', color='views', color_continuous_scale='Blues')
        st.plotly_chart(fig3)
    else:
        st.warning("'channel_title' column not found in dataset.")

    st.markdown("""
    ---
    ## 📌 Summary
    - **Forecasting** views using ARIMA
    - **RMSE** calculated for prediction accuracy
    - **Keyword** analysis using CountVectorizer and Word Cloud
    - **Channel** level comparison using Plotly
    
    This tool provides insights into trending content on YouTube.
    """)

else:
    st.info("Please upload a valid CSV file to begin.")

st.markdown("""
---
Developed as part of **Applied Data Science (CSDC8013)** project.
""")