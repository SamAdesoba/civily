import flask
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from apscheduler.schedulers.background import BackgroundScheduler
import datetime
import snscrape.modules.twitter as sntwitter
import atexit
import random

atiku_model = pickle.load(open('model_training/atiku/atiku_model_pickle.pkl', 'rb'))

obi_model = pickle.load(open('model_training/obi/obi_model_pickle.pkl', 'rb'))

tinubu_model = pickle.load(open('model_training/tinubu/log_reg_tinubu.pkl', 'rb'))

vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 2), max_df=500)

atiku_df = pd.read_csv('util/atiku.csv')
atiku_tweet_df = atiku_df['tweet']

obi_df = pd.read_csv('util/peterobi.csv')
obi_tweet_df = obi_df['tweet']

tinubu_df = pd.read_csv('util/tinubu.csv')
tinubu_tweet_df = tinubu_df['tweet']

# last_date = datetime.date.today()
# new_date = datetime.timedelta(hours=24)



def cleanText(text):
    text = re.sub(r'@[A-Za-z0-9^\w]+', '', text)
    text = re.sub(r'#[A-Za-z0-9^\w]+', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r"[^a-zA-Z0-9]+", ' ', text)
    text = re.sub(r'\:', '', text)
    text = re.sub(r'\...', '', text)
    return text


def reformat_json(text):
    text = re.sub(r'\"', ' ', text)
    text = re.sub(r'\,.', ' ', text)
    text = re.sub(r'\(', ' ', text)
    text = re.sub(r'\)..', ' ', text)
    return text


app = flask.Flask(__name__)

# @app.route('/home')
# def sensor():
#     result_1 = []
#     search_1 = f'(peterobi OR #peterobi OR #obidatti2023) until:{last_date} since:{last_date - new_date}'
#     for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_1).get_items()):
#         if i > 10:
#             break
#         else:
#             result_1.append([tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location, tweet.likeCount, tweet.retweetCount])
#     df = pd.DataFrame(result_1, columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
#     return df.to_dict()



# df_extracted_tweet = sensor().copy()

# df = pd.DataFrame(df_extracted_tweet)

# df_tweet = df['tweet']


@app.route('/api/v1/atiku', methods=['GET', 'POST'])
def atiku_sentiment():
    cleaned_data = atiku_tweet_df.apply(cleanText)

    clean_df = pd.DataFrame(cleaned_data, columns=['tweet'])

    vectorizer.fit(clean_df['tweet'].values)

    vectorized = vectorizer.transform(clean_df['tweet'])

    vectorized_df = pd.DataFrame(vectorized.toarray(), columns=vectorizer.get_feature_names_out())

    result = atiku_model.predict(vectorized_df.values)

    result_df = pd.DataFrame(result)

    # format_result = result_df.value_counts().to_json(orient='index')

    # return reformat_json(format_result)
    return result_df.value_counts().to_json(orient='index')


@app.route('/api/v1/obi', methods=['GET', 'POST'])
def obi_sentiment():
    cleaned_data = obi_tweet_df.apply(cleanText)

    clean_df = pd.DataFrame(cleaned_data, columns=['tweet'])

    vectorizer.fit(clean_df['tweet'].values)

    vectorized = vectorizer.transform(clean_df['tweet'])

    vectorized_df = pd.DataFrame(vectorized.toarray(), columns=vectorizer.get_feature_names_out())

    result = obi_model.predict(vectorized_df.values)

    result_df = pd.DataFrame(result)

    # format_result = result_df.value_counts().to_json(orient='index')

    # return reformat_json(format_result)
    return result_df.value_counts().to_json(orient='index')


@app.route('/api/v1/tinubu', methods=['GET', 'POST'])
def tinubu_sentiment():
    cleaned_data = tinubu_tweet_df.apply(cleanText)

    clean_df = pd.DataFrame(cleaned_data, columns=['tweet'])

    vectorizer.fit(clean_df['tweet'].values)

    vectorized = vectorizer.transform(clean_df['tweet'])

    vectorized_df = pd.DataFrame(vectorized.toarray(), columns=vectorizer.get_feature_names_out())

    result = obi_model.predict(vectorized_df.values)

    result_df = pd.DataFrame(result)

    # format_result = result_df.value_counts().to_json(orient='index')

    # return reformat_json(format_result)
    return result_df.value_counts().to_json(orient='index')






if __name__ == "__main__":
    app.run()
