# eda_visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

def run_eda_visualization(df):
    logger.info("===================>>> Exploratory Data Analysis (EDA) && Visualization")

    # Convert tweet_created to datetime
    logger.info("Convert tweet_created column to datetime")
    df['tweet_created'] = pd.to_datetime(df['tweet_created'])
    print(df.dtypes)

    logger.info("Distribution of Sentiments in the Dataset:")
    sentiment_counts = df['airline_sentiment'].value_counts()
    logger.info(f"Sentiment Distribution:\n{sentiment_counts}")

    logger.info("Number of Tweets per Airline:")
    print(df['airline'].value_counts())

    plt.style.use('seaborn-v0_8')
    sns.set_palette('coolwarm')

    plt.figure(figsize=(6, 4))
    sns.countplot(x='airline_sentiment', data=df)
    plt.title('Distribution of Sentiments')
    plt.show()

    logger.info("What is the percentage of each sentiment category?")
    sentiment_percent = round(df['airline_sentiment'].value_counts(normalize=True) * 100, 2)
    print(sentiment_percent)
    print("Negative tweets represent around two-thirds of the dataset.\n")

    plt.figure(figsize=(5, 5))
    sentiment_percent.plot.pie(autopct='%1.1f%%')
    plt.title('Sentiment Percentage')
    plt.ylabel('')
    plt.show()

    logger.info("Which airline was mentioned the most?")
    airline_counts = df['airline'].value_counts()
    print(airline_counts.head(1))
    logger.info(f"The most mentioned airline is {airline_counts.index[0]} with {airline_counts.iloc[0]} tweets.\n")

    plt.figure(figsize=(8, 4))
    sns.countplot(x='airline', data=df)
    plt.title('Number of Tweets per Airline')
    plt.xticks(rotation=45)
    plt.show()

    logger.info("How are sentiments distributed across each airline?")
    cross_tab = pd.crosstab(df['airline'], df['airline_sentiment'])
    print(cross_tab)
    logger.info("Some airlines like United and US Airways have more negative tweets than others.\n")

    plt.figure(figsize=(10, 6))
    sns.countplot(x='airline', hue='airline_sentiment', data=df)
    plt.title('Sentiment Distribution per Airline')
    plt.xticks(rotation=45)
    plt.show()

    logger.info("What is the most common reason for negative tweets?")
    neg_reason = df[df['airline_sentiment'] == 'negative']['negativereason'].value_counts()
    print(neg_reason.head())
    logger.info(f"The most common negative reason is '{neg_reason.index[0]}'.\n")

    plt.figure(figsize=(8, 4))
    sns.barplot(x=neg_reason.head(5).index, y=neg_reason.head(5).values)
    plt.title('Top Negative Reasons')
    plt.xticks(rotation=45)
    plt.show()


    logger.info("Which airline received the most negative tweets?")
    neg_airline = df[df['airline_sentiment'] == 'negative']['airline'].value_counts()
    print(neg_airline.head())
    logger.info(f"The airline with the most negative tweets is {neg_airline.index[0]}.\n")

    plt.figure(figsize=(8, 4))
    sns.barplot(x=neg_airline.index, y=neg_airline.values)
    plt.title('Most Criticized Airlines (Negative Tweets)')
    plt.xticks(rotation=45)
    plt.show()

    logger.info("Which airline received the most positive tweets?")
    pos_airline = df[df['airline_sentiment'] == 'positive']['airline'].value_counts()
    print(pos_airline.head())
    logger.info(f"The airline with the most positive tweets is {pos_airline.index[0]}.\n")

    plt.figure(figsize=(8, 4))
    sns.barplot(x=pos_airline.index, y=pos_airline.values)
    plt.title('Most Praised Airlines (Positive Tweets)')
    plt.xticks(rotation=45)
    plt.show()

    logger.info("How are tweets distributed throughout the day?")
    df['hour'] = df['tweet_created'].dt.hour
    hour_counts = df['hour'].value_counts().sort_index()
    print(hour_counts.head(10))
    logger.info("Tweet activity is generally higher during the morning and early afternoon.\n")

    plt.figure(figsize=(8, 4))
    sns.countplot(x='hour', data=df)
    plt.title('Tweets by Hour of Day')
    plt.show()

    logger.info("What is the average tweet length (in words)?")
    df['tweet_length'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    print(df['tweet_length'].describe())
    logger.info("Most tweets contain between 15 and 25 words.\n")

    plt.figure(figsize=(8, 4))
    sns.histplot(df['tweet_length'], bins=30, kde=True)
    plt.title('Tweet Length Distribution')
    plt.show()

    logger.info("Are negative tweets longer than positive ones?")
    avg_len_by_sentiment = df.groupby('airline_sentiment')['tweet_length'].mean()
    print(avg_len_by_sentiment)
    logger.info("Negative tweets tend to be slightly longer on average 65%.\n")

    sns.boxplot(x='airline_sentiment', y='tweet_length', data=df)
    plt.title('Tweet Length by Sentiment')
    plt.show()

    logger.info("Is there a difference in retweet count between sentiments?")
    mean_retweet = df.groupby('airline_sentiment')['retweet_count'].mean()
    print(mean_retweet)
    logger.info("Negative tweets often receive more retweets because they provoke stronger reactions.\n")

    sns.boxplot(x='airline_sentiment', y='retweet_count', data=df)
    plt.title('Retweet Count by Sentiment')
    plt.show()

    print("Who are the most active users (with the most tweets)?")
    top_users = df['name'].value_counts().head(5)
    print(top_users)
    logger.info(f"These users tweeted the most about airlines {top_users}.\n")

    sns.barplot(x=top_users.index, y=top_users.values)
    plt.title('Top Active Users')
    plt.xticks(rotation=45)
    plt.show()


    logger.info("What are the most common tweet locations?")
    top_locations = df['tweet_location'].value_counts().head(5)
    print(top_locations)
    logger.info(f"The most active location is {top_locations.index[0]}.\n")

    sns.barplot(x=top_locations.index, y=top_locations.values)
    plt.title('Top Tweet Locations')
    plt.xticks(rotation=45)
    plt.show()

    logger.info("What is the average confidence level for each sentiment?")
    avg_conf = df.groupby('airline_sentiment')['airline_sentiment_confidence'].mean()
    print(avg_conf)
    logger.info(f"Negative sentiment confidence is usually higher since the language is more explicit.{avg_conf}\n")

    sns.boxplot(x='airline_sentiment', y='airline_sentiment_confidence', data=df)
    plt.title('Sentiment Confidence by Type')
    plt.show()

    logger.info("How is the confidence in negative reasons distributed?")
    sns.histplot(df['negativereason_confidence'], bins=20, kde=True)
    plt.title('Distribution of Negative Reason Confidence')
    plt.show()

    logger.info("What is the most common negative reason for each airline?")
    most_reason_per_airline = df[df['airline_sentiment'] == 'negative'].groupby('airline')['negativereason'].agg(lambda x: x.value_counts().index[0])
    print(most_reason_per_airline)
    logger.info("Each airline has different top negative reasons depending on user complaints.\n")

    logger.info("Do retweet counts differ across airlines?")
    sns.boxplot(x='airline', y='retweet_count', data=df)
    plt.title('Retweet Count by Airline')
    plt.xticks(rotation=45)
    plt.show()

    logger.info("How do positive and negative tweets compare across airlines?")
    sentiment_airline = df.groupby(['airline', 'airline_sentiment']).size().unstack()
    print(sentiment_airline)
    logger.info("This shows which airlines receive the most positive vs negative feedback.\n")

    sentiment_airline[['positive', 'negative']].plot(kind='bar', figsize=(10, 5))
    plt.title('Positive vs Negative Tweets per Airline')
    plt.show()

    logger.info("===================>>> EDA & Visualization Completed Successfully!")


    return df
