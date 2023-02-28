import flask
from flask_cors import CORS
from flask import jsonify
import pickle
import re
import pandas as pd
from sklearn import *
from sklearn.feature_extraction.text import CountVectorizer
import datetime
import snscrape.modules.twitter as sntwitter
from collections import Counter


app = flask.Flask(__name__)
CORS(app)


# trained models for each candidate and vectorizer
atiku_model = pickle.load(open('model_training/atiku/atiku_model_pickle.pkl', 'rb'))
obi_model = pickle.load(open('model_training/obi/obi_model_pickle.pkl', 'rb'))
tinubu_model = pickle.load(open('model_training/tinubu/log_reg_tinubu.pkl', 'rb'))

vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 2), max_df=500)
    

# function to clean extracted tweets
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


# function to clean extra characters fron the json returned ia the api
def reformat_json(text):
    text = re.sub(r'\"', ' ', text)
    text = re.sub(r'\,.', ' ', text)
    text = re.sub(r'\(', ' ', text)
    text = re.sub(r'\)..', ' ', text)
    return text


def sentiment_json_fromat(result):
    counts = Counter(result)
    value_df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    value_df.columns = ['Analysis', 'Sentiment_Count']
    value_df.sort_values(by='Sentiment_Count', ascending=False, inplace=True)
    return value_df.set_index(value_df['Analysis']).drop("Analysis", axis=1).to_json(orient='columns')


# lists to capture extracted data from twitter
result_atiku = []
result_obi = []
result_tinubu = []


# time range for extracted twitter data
current_date = datetime.date.today()
last_date = datetime.timedelta(hours=72)


# function to extract twitter data
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


# compile extracted data into a list from which we pick individual data for the candidates
combined_list = sensor().copy()

atiku_df = pd.DataFrame(combined_list[0].copy(), columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
atiku_tweet_df = atiku_df['tweet']

obi_df = pd.DataFrame(combined_list[1].copy(), columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
obi_tweet_df = obi_df['tweet']

tinubu_df = pd.DataFrame(combined_list[2].copy(), columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
tinubu_tweet_df = tinubu_df['tweet']


def mention(tweet):
    mentions = re.findall(r'@(\w+)', tweet)
    return ' '.join(mentions)


def hashtag(tweet):
    tags = re.findall(r'#(\w+)', tweet)
    return ' '.join(tags)


atiku_df['mentions'] = atiku_df['tweet'].apply(mention)
atiku_mentions_list = atiku_df['mentions'].tolist()


obi_df['mentions'] = obi_df['tweet'].apply(mention)
obi_mentions_list = obi_df['mentions'].tolist()


tinubu_df['mentions'] = tinubu_df['tweet'].apply(mention)
tinubu_mentions_list = tinubu_df['mentions'].tolist()


atiku_df['hashtags'] = atiku_df['tweet'].apply(hashtag)
atiku_hashtags_list = atiku_df['hashtags'].tolist()


obi_df['hashtags'] = obi_df['tweet'].apply(hashtag)
obi_hashtags_list = obi_df['hashtags'].tolist()


tinubu_df['hashtags'] = tinubu_df['tweet'].apply(hashtag)
tinubu_hashtags_list = tinubu_df['hashtags'].tolist()


def get_atiku_mention():
    atiku_mentions = []
    atiku_mentions.clear()
    for item in atiku_mentions_list:
        item = item.split()
        for i in item:
            atiku_mentions.append(i)

    counts = Counter(atiku_mentions)
    mentions_df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    mentions_df.columns = ['Mentions', 'Count']
    mentions_df.sort_values(by='Count', ascending=False, inplace=True)
    atiku_mentions_df = mentions_df[mentions_df['Mentions']=='atiku']
    return atiku_mentions_df.set_index(atiku_mentions_df['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


def get_atiku_hash_tag():
    atiku_hashtags = []
    atiku_hashtags.clear()
    for item in atiku_hashtags_list:
        item = item.split()
        for i in item:
            atiku_hashtags.append(i)

    counts = Counter(atiku_hashtags)
    hashtags_df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    hashtags_df.columns = ['Hashtags', 'Count']
    hashtags_df.sort_values(by='Count', ascending=False, inplace=True)
    sort = hashtags_df.head(10)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


def atiku_sentiment():
    cleaned_data = atiku_tweet_df.apply(cleanText)
    clean_df = pd.DataFrame(cleaned_data, columns=['tweet'])
    vectorizer.fit(clean_df['tweet'].values)
    vectorized = vectorizer.transform(clean_df['tweet'])
    vectorized_df = pd.DataFrame(vectorized.toarray(), columns=vectorizer.get_feature_names_out())
    result = atiku_model.predict(vectorized_df.values)
    return sentiment_json_fromat(result)


def get_obi_mention():
    obi_mentions = []
    obi_mentions.clear()
    for item in obi_mentions_list:
        item = item.split()
        for i in item:
            obi_mentions.append(i)

    counts = Counter(obi_mentions)
    mentions_df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    mentions_df.columns = ['Mentions', 'Count']
    mentions_df.sort_values(by='Count', ascending=False, inplace=True)
    obi_mentions_df = mentions_df[mentions_df['Mentions']=='PeterObi']
    return obi_mentions_df.set_index(obi_mentions_df['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


def get_obi_hash_tag():
    obi_hashtags = []
    obi_hashtags.clear()
    for item in obi_hashtags_list:
        item = item.split()
        for i in item:
            obi_hashtags.append(i)

    counts = Counter(obi_hashtags)
    hashtags_df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    hashtags_df.columns = ['Hashtags', 'Count']
    hashtags_df.sort_values(by='Count', ascending=False, inplace=True)
    sort = hashtags_df.head(10)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


def obi_sentiment():
    cleaned_data = obi_tweet_df.apply(cleanText)
    clean_df = pd.DataFrame(cleaned_data, columns=['tweet'])
    vectorizer.fit(clean_df['tweet'].values)
    vectorized = vectorizer.transform(clean_df['tweet'])
    vectorized_df = pd.DataFrame(vectorized.toarray(), columns=vectorizer.get_feature_names_out())
    result = obi_model.predict(vectorized_df.values)
    return sentiment_json_fromat(result)


def get_tinubu_mention():
    tinubu_mentions = []
    tinubu_mentions.clear()
    for item in tinubu_mentions_list:
        item = item.split()
        for i in item:
            tinubu_mentions.append(i)

    counts = Counter(tinubu_mentions)
    mentions_df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    mentions_df.columns = ['Mentions', 'Count']
    mentions_df.sort_values(by='Count', ascending=False, inplace=True)
    tinubu_mentions_df = mentions_df[mentions_df['Mentions']=='officialABAT']
    return tinubu_mentions_df.set_index(tinubu_mentions_df['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


def get_tinubu_hash_tag():
    tinubu_hashtags = []
    tinubu_hashtags.clear()
    for item in tinubu_hashtags_list:
        item = item.split()
        for i in item:
            tinubu_hashtags.append(i)

    counts = Counter(tinubu_hashtags)
    hashtags_df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    hashtags_df.columns = ['Hashtags', 'Count']
    hashtags_df.sort_values(by='Count', ascending=False, inplace=True)
    sort = hashtags_df.head(10)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


def tinubu_sentiment():
    cleaned_data = tinubu_tweet_df.apply(cleanText)
    clean_df = pd.DataFrame(cleaned_data, columns=['tweet'])
    vectorizer.fit(clean_df['tweet'].values)
    vectorized = vectorizer.transform(clean_df['tweet'])
    vectorized_df = pd.DataFrame(vectorized.toarray(), columns=vectorizer.get_feature_names_out())
    result = obi_model.predict(vectorized_df.values)
    return sentiment_json_fromat(result)


@app.route('/api/v1/sentiments/<candidate>', methods=['GET', 'POST'])
def get_sentiments(candidate):
    sensor()
    if candidate == 'atiku':
        return atiku_sentiment()
    elif candidate == 'obi':
        return obi_sentiment()
    else:
        return tinubu_sentiment()


@app.route('/api/v1/hashtags/<candidate>', methods=['GET', 'POST'])
def get_hashtags(candidate):
    if candidate == 'atiku':
        return get_atiku_hash_tag()
    elif candidate == 'obi':
        return get_obi_hash_tag()
    else:
        return get_tinubu_hash_tag()
    
    
@app.route('/api/v1/mentions/<candidate>', methods=['GET', 'POST'])
def get_mentions(candidate):
    if candidate == 'atiku':
        return get_atiku_mention()
    elif candidate == 'obi':
        return get_obi_mention()
    else:
        return get_tinubu_mention()


if __name__ == "__main__":
    app.run(debug=True)
