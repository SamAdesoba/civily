import flask
import pickle
import re
import pandas as pd
from sklearn import *
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

# atiku_df = pd.read_csv('util/atiku.csv')
# atiku_tweet_df = atiku_df['tweet']

# obi_df = pd.read_csv('util/peterobi.csv')
# obi_tweet_df = obi_df['tweet']

# tinubu_df = pd.read_csv('util/tinubu.csv')
# tinubu_tweet_df = tinubu_df['tweet']

current_date = datetime.date.today()
last_date = datetime.timedelta(hours=72)
    




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


result_atiku = []
result_obi = []
result_tinubu = []
# @app.route('/api/v1/scrape')
def sensor():
    for candidates in ['atiku', 'obi', 'tinubu']:
        if (candidates == 'atiku'):
            result_atiku.clear()
            search_atiku = f'(atikuabubakar OR atikuokowa OR #atikuokowa2023 OR #atikuabubakar OR #atikulated2023) until:{current_date} since:{current_date - last_date}'     
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_atiku).get_items()):
                if i > 1000:
                    break
                else:
                    result_atiku.append([tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location, tweet.likeCount, tweet.retweetCount])
         

        if (candidates == 'obi'):
            result_obi.clear()
            search_obi = f'(peterobi OR #peterobi OR #obidatti2023) until:{current_date} since:{current_date - last_date}'
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_obi).get_items()):
                if i > 1000:
                    break
                else:
                    result_obi.append([tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location, tweet.likeCount, tweet.retweetCount])


        if (candidates == 'tinubu'):
            result_tinubu.clear()
            search_tinubu = f'(bolatinubu OR #bolatinubu OR #bat2023) until:{current_date} since:{current_date - last_date}'
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_tinubu).get_items()):
                if i > 1000:
                    break
                else:
                    result_tinubu.append([tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location, tweet.likeCount, tweet.retweetCount])
    
    combined = [result_atiku, result_obi, result_tinubu]
    return combined

combined_list = sensor().copy()

atiku_df = pd.DataFrame(combined_list[0].copy(), columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
atiku_tweet_df = atiku_df['tweet']


obi_df = pd.DataFrame(combined_list[1].copy(), columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
obi_tweet_df = obi_df['tweet']

tinubu_df = pd.DataFrame(combined_list[2].copy(), columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
tinubu_tweet_df = tinubu_df['tweet']



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


def obi_sentiment():
    # obi_tweet_df = pd.DataFrame(sensor().copy())['tweet']
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


@app.route('/api/v1/<candidate>', methods=['GET', 'POST'])
def sentiments(candidate):
    # assert candidate == request.args.get('candidate')
    if candidate == 'atiku':
        return atiku_sentiment()
    elif candidate == 'obi':
        return obi_sentiment()
    else:
        return tinubu_sentiment()




if __name__ == "__main__":
    app.run()
