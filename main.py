import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import text2emotion as te
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("vader_lexicon")

df = pd.read_csv("tweets.csv")


def preprocess_data(df):
    columns_to_drop = [
        "id",
        "conversation_id",
        "user_id",
        "user_id_str",
        "date",
        "timezone",
        "place",
        "cashtags",
        "hashtags",
        "username",
        "link",
        "nlikes",
        "nreplies",
        "nretweets",
        "day",
        "hour",
        "urls",
        "photos",
        "video",
        "thumbnail",
        "retweet",
        "quote_url",
        "search",
        "name",
        "near",
        "geo",
        "source",
        "user_rt_id",
        "user_rt",
        "retweet_id",
        "reply_to",
        "retweet_date",
        "translate",
        "trans_src",
        "trans_dest",
    ]
    df.drop(columns_to_drop, axis=1, inplace=True)
    df = df.iloc[:, 1:]

    df = df[df["language"] == "en"]
    df["tweet"] = df["tweet"].str.lower()
    df.drop_duplicates(subset="tweet", keep="first", inplace=True)
    df["created_at"] = pd.to_datetime(df["created_at"], unit="ms")
    df["created_at"] = df["created_at"].dt.hour

    return df


def tokenize(df):
    df["tweet_words"] = df["tweet"].apply(word_tokenize)

    return df


def stop_words(df):
    stop_words = set(stopwords.words("english"))
    additional_stopwords = ["http", "https"]
    stop_words.update(additional_stopwords)
    df["tweet_words"] = df["tweet_words"].apply(
        lambda x: [
            word
            for word in x
            if word.isalpha() and word not in stop_words and len(word) > 1
        ]
    )

    return df


def lemmatize(df):
    lemmatizer = WordNetLemmatizer()
    df["tweet_words_lem"] = df["tweet_words"].apply(
        lambda x: [lemmatizer.lemmatize(word) for word in x]
    )

    return df


def sentiment_analysis(df):
    sia = SentimentIntensityAnalyzer()

    df["sentiment_scores"] = df["tweet_words_lem"].apply(
        lambda x: sia.polarity_scores(" ".join(x))
    )
    df["compound"] = df["sentiment_scores"].apply(
        lambda score_dict: score_dict["compound"]
    )
    df["sentiment"] = df["compound"].apply(
        lambda x: "positive" if x > 0 else "neutral" if x == 0 else "negative"
    )

    return df


def emotion_analysis(df):
    df_emotions = df.copy()
    df_emotions["emotions"] = df["tweet_words_lem"].apply(
        lambda x: te.get_emotion(" ".join(x))
    )
    df_emotions["happy"] = df_emotions["emotions"].apply(lambda x: x["Happy"])
    df_emotions["angry"] = df_emotions["emotions"].apply(lambda x: x["Angry"])
    df_emotions["surprise"] = df_emotions["emotions"].apply(lambda x: x["Surprise"])
    df_emotions["sad"] = df_emotions["emotions"].apply(lambda x: x["Sad"])
    df_emotions["fear"] = df_emotions["emotions"].apply(lambda x: x["Fear"])

    return df_emotions


def analyze_and_cluster(df):
    df_copy = df[df["language"] == "en"].copy()

    vectorizer = TfidfVectorizer(
        max_df=0.95, min_df=2, stop_words="english"
    )  # max_df i min_df - ignorowania terminów, które pojawiają się zbyt często lub zbyt rzadko
    tfidf = vectorizer.fit_transform(
        df_copy["tweet_words_lem"].apply(lambda x: " ".join(x))
    )

    lda = LatentDirichletAllocation(
        n_components=5, random_state=0
    )  # Inicjalizujemy LDA z 5 komponentami (tematami)
    lda.fit(tfidf)  # Dopasowujemy model LDA do naszych danych

    feature_names = vectorizer.get_feature_names_out()
    for i, topic in enumerate(lda.components_):
        print(f"{i}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-11:-1]]))

    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(tfidf)

    df_copy["cluster"] = kmeans.labels_

    return df_copy


def freq_plot(df):
    word_freq = Counter(word for words in df["tweet_words_lem"] for word in words)

    most_common_words = word_freq.most_common(10)

    words, frequencies = zip(*most_common_words)

    plt.figure(figsize=(10, 5))
    plt.bar(words, frequencies)
    plt.title("10 najczęściej występujących słów")
    plt.xlabel("Słowa")
    plt.ylabel("Częstość")
    plt.show()


def words_cloud(df):
    df["tweet_words_lem"] = df["tweet_words_lem"].astype(str)
    wordcloud = WordCloud(width=1000, height=500).generate(
        " ".join(df["tweet_words_lem"].explode())
    )
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


def plot_clusters(df):
    cluster_counts = df["cluster"].value_counts()

    cluster_numbers = [str(cluster) for cluster in cluster_counts.index]

    plt.figure(figsize=(10, 6))
    plt.bar(cluster_numbers, cluster_counts.values, color="blue", alpha=0.7)
    plt.xlabel("Numer klastra")
    plt.ylabel("Liczba tweetów")
    plt.title("Liczba tweetów w poszczególnych klastrach")
    plt.xticks(rotation="vertical", fontsize="small")
    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv("tweets.csv")
    # print(df.shape)

    df = preprocess_data(df)
    # print(df.head())
    # print(df.shape)

    df = tokenize(df)

    # print(df["tweet_words"].head())
    df = stop_words(df)
    # print(df["tweet_words"].head())

    # print(df["tweet_words"].explode().value_counts().head(10))  # Częstotliwość słów

    df = lemmatize(df)
    df["equals"] = df["tweet_words"] == df["tweet_words_lem"]
    print(df["equals"].value_counts())

    freq_plot(df)
    words_cloud(df)

    # pd.set_option("display.max_columns", None)
    # print(df)

    # Analiza sentymentu
    df = sentiment_analysis(df)
    sns.countplot(x="sentiment", data=df)
    plt.show()

    sentiment_over_time = df.groupby("created_at")["compound"].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="created_at", y="compound", data=sentiment_over_time)
    plt.ylim(-1, 1)
    plt.xlabel("Godzina")
    plt.ylabel("Sentyment")
    plt.title("Sentyment tweetów w zależności od godziny")
    plt.axhline(0, color="red", linestyle="--")
    plt.show()

    # Analiza emocji
    df_emotions = emotion_analysis(df)
    print(df_emotions)
    emotions_over_time = (
        df_emotions.groupby("created_at")[["happy", "angry", "surprise", "sad", "fear"]]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(10, 6))

    for emotion in ["happy", "angry", "surprise", "sad", "fear"]:
        sns.lineplot(x="created_at", y=emotion, data=emotions_over_time, label=emotion)

    plt.xlabel("Godzina")
    plt.ylabel("Intensywność emocji")
    plt.title("Emocje tweetów w zależności od godziny")
    plt.legend()
    plt.show()

    # Analiza tematyki opinii i klasteryzacja
    df_ac = analyze_and_cluster(df)
    plot_clusters(df_ac)


if __name__ == "__main__":
    main()
