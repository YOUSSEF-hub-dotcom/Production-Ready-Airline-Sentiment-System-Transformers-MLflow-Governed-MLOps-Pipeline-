import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Hedaya_city\Downloads\Tweets.csv", encoding="latin1")
    df = df.drop_duplicates()

    df['negativereason'] = df['negativereason'].fillna(df['negativereason'].mode()[0])
    df['negativereason_confidence'] = df['negativereason_confidence'].fillna(df['negativereason_confidence'].median())
    df['tweet_location'] = df['tweet_location'].fillna(df['tweet_location'].mode()[0])
    df['user_timezone'] = df['user_timezone'].fillna(df['user_timezone'].mode()[0])

    df = df.drop(columns=['tweet_coord', 'negativereason_gold', 'airline_sentiment_gold'], errors='ignore')

    df['lower_text'] = df['text'].str.lower()
    df['tokenized_text'] = df['lower_text'].apply(nltk.word_tokenize)
    df['no_specials'] = df['tokenized_text'].apply(lambda x: [re.sub(r'[^a-zA-Z]', '', w) for w in x])
    stop_words = set(stopwords.words('english'))
    df['no_stopwords'] = df['no_specials'].apply(lambda x: [w for w in x if w not in stop_words and w != ''])
    stemmer = PorterStemmer()
    df['stemmed_tokens'] = df['no_stopwords'].apply(lambda x: [stemmer.stem(w) for w in x])
    lemmatizer = WordNetLemmatizer()
    df['lemmatized_text'] = df['no_stopwords'].apply(lambda x: [lemmatizer.lemmatize(w) for w in x])
    df['cleaned_text'] = df['lemmatized_text'].apply(lambda x: " ".join(x))

    df['tweet_created'] = pd.to_datetime(df['tweet_created'])
    df['hour'] = df['tweet_created'].dt.hour
    df['text_length'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['tweet_length'] = df['cleaned_text'].apply(lambda x: len(x.split()))

    return df

df = load_data()

st.set_page_config(page_title="✈️ Airline Sentiment Dashboard", layout="wide", page_icon="✈️")
st.title("✈️ Airline Customer Sentiment Dashboard")
st.markdown("---")

tab_option = st.sidebar.selectbox("Choose a Tab", [
    "📘 Sentiment Overview",
    "📊 Sentiment per Airline",
    "📉 Negative Feedback",
    "📈 Positive Feedback",
    "📝 Word Clouds",
    "🕒 Users & Timing",
    "📏 Tweet Metrics",
    "📌 Additional Metrics"
])

# ---------------- Tab 1: Sentiment Overview ----------------
if tab_option == "📘 Sentiment Overview":
    col1, col2 = st.columns(2)
    with col1:
        sentiment_percent = round(df['airline_sentiment'].value_counts(normalize=True)*100,2)
        fig, ax = plt.subplots(figsize=(4,4))
        sentiment_percent.plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_ylabel('')
        ax.set_title('Sentiment Percentage')
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5,4))
        sns.countplot(x='airline', data=df, ax=ax)
        ax.set_title('Number of Tweets per Airline')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

# ---------------- Tab 2: Sentiment per Airline ----------------
elif tab_option == "📊 Sentiment per Airline":
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(x='airline', hue='airline_sentiment', data=df, ax=ax)
        ax.set_title('Sentiment Distribution per Airline')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5,4))
        sns.boxplot(x='airline_sentiment', y='tweet_length', data=df, ax=ax)
        ax.set_title('Tweet Length by Sentiment')
        st.pyplot(fig)

# ---------------- Tab 3: Negative Feedback ----------------
elif tab_option == "📉 Negative Feedback":
    col1, col2 = st.columns(2)
    with col1:
        neg_reason = df[df['airline_sentiment']=='negative']['negativereason'].value_counts()
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(x=neg_reason.head(5).index, y=neg_reason.head(5).values, ax=ax)
        ax.set_title('Top Negative Reasons')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    with col2:
        neg_airline = df[df['airline_sentiment']=='negative']['airline'].value_counts()
        fig, ax = plt.subplots(figsize=(5,4))
        sns.barplot(x=neg_airline.index, y=neg_airline.values, ax=ax)
        ax.set_title('Most Criticized Airlines')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

# ---------------- Tab 4: Positive Feedback ----------------
elif tab_option == "📈 Positive Feedback":
    col1, col2 = st.columns(2)
    with col1:
        pos_airline = df[df['airline_sentiment']=='positive']['airline'].value_counts()
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(x=pos_airline.index, y=pos_airline.values, ax=ax)
        ax.set_title('Most Praised Airlines')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    with col2:
        sentiment_airline = df.groupby(['airline', 'airline_sentiment']).size().unstack()
        fig, ax = plt.subplots(figsize=(6,4))
        sentiment_airline[['positive','negative']].plot(kind='bar', ax=ax)
        ax.set_title('Positive vs Negative Tweets per Airline')
        st.pyplot(fig)

# ---------------- Tab 5: Word Clouds ----------------
elif tab_option == "📝 Word Clouds":
    col1, col2 = st.columns(2)
    with col1:
        positive_text = " ".join(df[df['airline_sentiment']=='positive']['cleaned_text'])
        wordcloud_pos = WordCloud(width=600, height=300, background_color='white').generate(positive_text)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.imshow(wordcloud_pos, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Positive Tweets Words')
        st.pyplot(fig)
    with col2:
        negative_text = " ".join(df[df['airline_sentiment']=='negative']['cleaned_text'])
        wordcloud_neg = WordCloud(width=600, height=300, background_color='black', colormap="Reds").generate(negative_text)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.imshow(wordcloud_neg, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Negative Tweets Words')
        st.pyplot(fig)

# ---------------- Tab 6: Users & Timing ----------------
elif tab_option == "🕒 Users & Timing":
    col1, col2 = st.columns(2)
    with col1:
        top_users = df['name'].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(x=top_users.index, y=top_users.values, ax=ax)
        ax.set_title('Top Active Users')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    with col2:
        top_locations = df['tweet_location'].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(x=top_locations.index, y=top_locations.values, ax=ax)
        ax.set_title('Top Tweet Locations')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

# ---------------- Tab 7: Tweet Metrics ----------------
elif tab_option == "📏 Tweet Metrics":
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(x='hour', data=df, ax=ax)
        ax.set_title('Tweets by Hour of Day')
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.boxplot(x='airline_sentiment', y='retweet_count', data=df, ax=ax)
        ax.set_title('Retweet Count by Sentiment')
        st.pyplot(fig)

# ---------------- Tab 8: Additional Metrics ----------------
elif tab_option == "📌 Additional Metrics":
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.boxplot(x='airline_sentiment', y='tweet_length', data=df, ax=ax)
        ax.set_title('Tweet Length by Sentiment')
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.boxplot(x='airline_sentiment', y='airline_sentiment_confidence', data=df, ax=ax)
        ax.set_title('Sentiment Confidence by Type')
        st.pyplot(fig)

