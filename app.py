import flask
from flask_cors import CORS
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import datetime
import snscrape.modules.twitter as sntwitter
from collections import Counter
from itertools import zip_longest

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


# function to clean extra characters from the json returned ia the api
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


app = flask.Flask(__name__)
CORS(app)

# lists to capture extracted data from twitter
result_atiku = []
result_obi = []
result_tinubu = []

# time range for extracted twitter data
current_date = datetime.date.today()
last_date = datetime.timedelta(hours=72)


# function to extract twitter data
def sensor():
    for candidate in ['atiku', 'obi', 'tinubu']:
        if (candidate == 'atiku'):
            result_atiku.clear()
            search_atiku = f'(atikuabubakar OR atikuokowa OR #atikuokowa2023 OR #atikuabubakar OR #atikulated2023) until:{current_date} since:{current_date - last_date}'
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_atiku).get_items()):
                if i > 1000:
                    break
                else:
                    result_atiku.append(
                        [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                         tweet.likeCount, tweet.retweetCount])

        if (candidate == 'obi'):
            result_obi.clear()
            search_obi = f'(peterobi OR #peterobi OR #obidatti2023) until:{current_date} since:{current_date - last_date}'
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_obi).get_items()):
                if i > 1000:
                    break
                else:
                    result_obi.append(
                        [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                         tweet.likeCount, tweet.retweetCount])

        if (candidate == 'tinubu'):
            result_tinubu.clear()
            search_tinubu = f'(bolatinubu OR #bolatinubu OR #bat2023) until:{current_date} since:{current_date - last_date}'
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_tinubu).get_items()):
                if i > 1000:
                    break
                else:
                    result_tinubu.append(
                        [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                         tweet.likeCount, tweet.retweetCount])

    combined = [result_atiku, result_obi, result_tinubu]
    return combined


# compile extracted data into a list from which we pick individual data for the candidates
combined_list = sensor().copy()

atiku_df = pd.DataFrame(combined_list[0].copy(),
                        columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
atiku_tweet_df = atiku_df['tweet']

obi_df = pd.DataFrame(combined_list[1].copy(),
                      columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
obi_tweet_df = obi_df['tweet']

tinubu_df = pd.DataFrame(combined_list[2].copy(),
                         columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
tinubu_tweet_df = tinubu_df['tweet']

atiku_sentiment_list = []

obi_sentiment_list = []

tinubu_sentiment_list = []


def mention(tweet):
    mentions = re.findall(r'@(\w+)', tweet)
    return ' '.join(mentions)


def mentions(candidate_mention_list):
    mention = []
    mention.clear()
    for item in candidate_mention_list:
        item = item.split()
        for i in item:
            mention.append(i)
    counts = Counter(mention)
    mentions_df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    mentions_df.columns = ['Mentions', 'Count']
    mentions_df.sort_values(by='Count', ascending=False, inplace=True)
    return mentions_df


def hashtag(tweet):
    tags = re.findall(r'#(\w+)', tweet)
    return ' '.join(tags)


def hash_tag(candidate_hashtag_list):
    hashtags = []
    hashtags.clear()
    for item in candidate_hashtag_list:
        item = item.split()
        for i in item:
            hashtags.append(i)
    counts = Counter(hashtags)
    hashtags_df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    hashtags_df.columns = ['Hashtags', 'Hashtags_Count']
    hashtags_df.sort_values(by='Hashtags_Count', ascending=False, inplace=True)
    sort = hashtags_df.head(10)
    return sort


def sentiment(candidate_tweet_df, candidate_model):
    cleaned_data = candidate_tweet_df.apply(cleanText)
    clean_df = pd.DataFrame(cleaned_data, columns=['tweet'])
    vectorizer.fit(clean_df['tweet'].values)
    vectorized = vectorizer.transform(clean_df['tweet'])
    vectorized_df = pd.DataFrame(vectorized.toarray(), columns=vectorizer.get_feature_names_out())
    result = candidate_model.predict(vectorized_df.values)
    return result


def get_location_counts(sentiment_df):
    location = list(sentiment_df['location'])
    location = [str(i) for i in location]
    location = [i.lower() for i in location]
    pattern = r"\b(abia|abuja|adamawa|akwa ibom|anambra|bauchi|bayelsa|benue|borno|cross river|delta|ebonyi|edo|ekiti|enugu|gombe|imo|jigawa|kaduna|kano|katsina|kebbi|kogi|kwara|lagos|nasarawa|niger|ogun|ondo|osun|oyo|plateau|rivers|sokoto|taraba|yobe|zamfara)\b"
    counts = {}
    for item in location:
        matches = re.findall(pattern, item, flags=re.IGNORECASE)
        for match in matches:
            key = match.lower()
            if key in counts:
                counts[key] += 1
            else:
                counts[key] = 1
    return counts


def neutral_location(candidate_tweet_df, candidate_model):
    result = sentiment(candidate_tweet_df, candidate_model)
    location_df = atiku_df
    location_df['sentiment'] = result
    neu_df = location_df[location_df['sentiment'] == 'neutral']
    counts = get_location_counts(neu_df)
    return counts


def get_random_sentiment(candidate_df, result):
    # single_df = candidate_df
    # single_df['sentiment'] = result
    candidate_df['sentiment'] = result
    neu_df = candidate_df[candidate_df['sentiment'] == 'neutral'].sample()
    neg_df = candidate_df[candidate_df['sentiment'] == 'negative'].sample()
    pos_df = candidate_df[candidate_df['sentiment'] == 'positive'].sample()
    test = [neu_df, neg_df, pos_df]
    result = pd.concat(test)
    filtered_result = result[['username', 'tweet', 'sentiment', 'likeCount', 'retweetCount']]
    return filtered_result


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


# Mentions functions
def get_atiku_mention():
    atiku_mentions_df = mentions(atiku_mentions_list)
    atiku_mentions = atiku_mentions_df[atiku_mentions_df['Mentions'] == 'atiku']
    return atiku_mentions.set_index(atiku_mentions['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


def get_obi_mention():
    mentions_df = mentions(obi_mentions_list)
    obi_mentions_df = mentions_df[mentions_df['Mentions'] == 'PeterObi']
    return obi_mentions_df.set_index(obi_mentions_df['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


def get_tinubu_mention():
    mentions_df = mentions(tinubu_mentions_list)
    tinubu_mentions = mentions_df[mentions_df['Mentions'] == 'officialABAT']
    return tinubu_mentions.set_index(tinubu_mentions['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


# Hashtag functions
def get_atiku_hash_tag():
    sort = hash_tag(atiku_hashtags_list)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


def get_obi_hash_tag():
    sort = hash_tag(obi_hashtags_list)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


def get_tinubu_hash_tag():
    sort = hash_tag(tinubu_hashtags_list)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


# General value_count sentiment functions
def atiku_sentiment():
    result = sentiment(atiku_tweet_df, atiku_model)
    return sentiment_json_fromat(result)


def obi_sentiment():
    result = sentiment(obi_tweet_df, obi_model)
    return sentiment_json_fromat(result)


def tinubu_sentiment():
    result = sentiment(tinubu_tweet_df, tinubu_model)
    return sentiment_json_fromat(result)


# Single sentiment functions
def atiku_single_tweet_sentiments():
    result = sentiment(atiku_tweet_df, atiku_model)
    filtered_result = get_random_sentiment(atiku_df, result)
    return filtered_result.to_dict(orient='records')


def obi_single_tweet_sentiments():
    result = sentiment(obi_tweet_df, obi_model)
    filtered_result = get_random_sentiment(obi_df, result)
    return filtered_result.to_dict(orient='records')


def tinubu_single_tweet_sentiments():
    result = sentiment(tinubu_tweet_df, tinubu_model)
    filtered_result = get_random_sentiment(tinubu_df, result)
    return filtered_result.to_dict(orient='records')


# Location functions
def atiku_neutral_location():
    counts = neutral_location(atiku_tweet_df, atiku_model)
    return counts


def atiku_positive_location():
    result = sentiment(atiku_tweet_df, atiku_model)
    location_df = atiku_df
    location_df['sentiment'] = result
    pos_df = location_df[location_df['sentiment'] == 'positive']
    counts = get_location_counts(pos_df)
    return counts


def atiku_negative_location():
    result = sentiment(atiku_tweet_df, atiku_model)
    location_df = atiku_df
    location_df['sentiment'] = result
    neg_df = location_df[location_df['sentiment'] == 'negative']
    counts = get_location_counts(neg_df)
    return counts


def obi_neutral_location():
    counts = neutral_location(obi_tweet_df, obi_model)
    return counts


def obi_positive_location():
    result = sentiment(obi_tweet_df, obi_model)
    location_df = obi_df
    location_df['sentiment'] = result
    pos_df = location_df[location_df['sentiment'] == 'positive']
    counts = get_location_counts(pos_df)
    return counts


def obi_negative_location():
    result = sentiment(obi_tweet_df, obi_model)
    location_df = obi_df
    location_df['sentiment'] = result
    neg_df = location_df[location_df['sentiment'] == 'negative']
    counts = get_location_counts(neg_df)
    return counts


def tinubu_neutral_location():
    result = sentiment(tinubu_tweet_df, tinubu_model)
    counts = neutral_location(tinubu_tweet_df, tinubu_model)
    return counts


def tinubu_positive_location():
    result = sentiment(tinubu_tweet_df, tinubu_model)
    location_df = tinubu_df
    location_df['sentiment'] = result
    pos_df = location_df[location_df['sentiment'] == 'positive']
    counts = get_location_counts(pos_df)
    return counts


def tinubu_negative_location():
    result = sentiment(tinubu_tweet_df, tinubu_model)
    location_df = tinubu_df
    location_df['sentiment'] = result
    neg_df = location_df[location_df['sentiment'] == 'negative']
    counts = get_location_counts(neg_df)
    return counts


@app.route('/api/v1/scrape')
def scrapper():
    sensor()
    return 'Scrapping successful'


@app.route('/api/v1/single_sentiment/<candidate>')
def get_single_sentiment(candidate):
    if candidate == 'atiku':
        return atiku_single_tweet_sentiments()
    elif candidate == 'obi':
        return obi_single_tweet_sentiments()
    else:
        return tinubu_single_tweet_sentiments()


@app.route('/api/v1/sentiments/<candidate>')
def get_sentiments(candidate):
    if candidate == 'atiku':
        return atiku_sentiment()
    elif candidate == 'obi':
        return obi_sentiment()
    else:
        return tinubu_sentiment()


@app.route('/api/v1/hashtags/<candidate>')
def get_hashtags(candidate):
    if candidate == 'atiku':
        return get_atiku_hash_tag()
    elif candidate == 'obi':
        return get_obi_hash_tag()
    else:
        return get_tinubu_hash_tag()


@app.route('/api/v1/mentions/<candidate>')
def get_mentions(candidate):
    if candidate == 'atiku':
        return get_atiku_mention()
    elif candidate == 'obi':
        return get_obi_mention()
    else:
        return get_tinubu_mention()


@app.route('/api/v1/neutral-location/<candidate>')
def get_neutral_locations(candidate):
    if candidate == 'atiku':
        return atiku_neutral_location()
    elif candidate == 'obi':
        return obi_neutral_location()
    else:
        return tinubu_neutral_location()


@app.route('/api/v1/positive-location/<candidate>')
def get_positive_locations(candidate):
    if candidate == 'atiku':
        return atiku_positive_location()
    elif candidate == 'obi':
        return obi_positive_location()
    else:
        return tinubu_positive_location()


@app.route('/api/v1/negative-location/<candidate>')
def get_negative_locations(candidate):
    if candidate == 'atiku':
        return atiku_negative_location()
    elif candidate == 'obi':
        return obi_negative_location()
    else:
        return tinubu_negative_location()


if __name__ == "__main__":
    app.run()
