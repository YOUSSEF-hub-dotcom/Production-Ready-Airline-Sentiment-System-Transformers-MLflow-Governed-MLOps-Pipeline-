# text_preprocessing.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)


def run_text_preprocessing(df):
    logger.info("===================>>> Starting Text Preprocessing")

    logger.info("Converting text to lowercase...")
    df['lower_text'] = df['text'].str.lower()

    logger.info("Tokenizing text...")
    df['tokenized_text'] = df['lower_text'].apply(nltk.word_tokenize)

    logger.info("Removing special characters and numbers...")
    df['no_specials'] = df['tokenized_text'].apply(
        lambda x: [re.sub(r'[^a-zA-Z]', '', word) for word in x]
    )

    logger.info("Removing stopwords...")
    stop_words = set(stopwords.words('english'))
    df['no_stopwords'] = df['no_specials'].apply(
        lambda x: [word for word in x if word not in stop_words and word != '']
    )

    logger.info("Applying Porter Stemming...")
    stemmer = PorterStemmer()
    df['stemmed_tokens'] = df['no_stopwords'].apply(
        lambda tokens: [stemmer.stem(word) for word in tokens]
    )

    logger.info("Joining cleaned tokens back into text...")
    df['cleaned_text'] = df['stemmed_tokens'].apply(lambda x: " ".join(x))

    logger.info("\nSample of preprocessing steps:")
    print(df[['text', 'lower_text', 'tokenized_text', 'no_specials',
              'no_stopwords', 'stemmed_tokens', 'cleaned_text']].head(5))

    logger.info("\nGenerating WordCloud for Positive Tweets...")
    positive_text = " ".join(df[df['airline_sentiment'] == 'positive']['cleaned_text'])
    if positive_text.strip():
        wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_pos, interpolation='bilinear')
        plt.axis("off")
        plt.title("Most Common Words in Positive Tweets")
        plt.tight_layout()
        plt.show()
    else:
        logger.info("No positive tweets found.")

    logger.info("Generating WordCloud for Negative Tweets...")
    negative_text = " ".join(df[df['airline_sentiment'] == 'negative']['cleaned_text'])
    if negative_text.strip():
        wordcloud_neg = WordCloud(width=800, height=400, background_color='black', colormap="Reds").generate(negative_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_neg, interpolation='bilinear')
        plt.axis("off")
        plt.title("Most Common Words in Negative Tweets")
        plt.tight_layout()
        plt.show()
    else:
        logger.info("No negative tweets found.")


    original_vocab = len(set(" ".join(df['text']).split()))
    cleaned_vocab = len(set(" ".join(df['cleaned_text']).split()))
    logger.info(
        f"Vocabulary Reduction: From {original_vocab} to {cleaned_vocab} unique tokens ({(1 - cleaned_vocab / original_vocab) * 100:.2f}% reduction)")


    logger.info("===================>>> Text Preprocessing Completed Successfully!")


    return df