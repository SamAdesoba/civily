import flask
from flask_cors import CORS
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import datetime
import snscrape.modules.twitter as sntwitter
from collections import Counter
from apscheduler.schedulers.background import BackgroundScheduler

app = flask.Flask(__name__)
CORS(app)





# trained models for each candidate and vectorizer
atiku_model = pickle.load(open('model/atiku_model.pkl', 'rb'))
obi_model = pickle.load(open('model/obi_model_pickle.pkl', 'rb'))
tinubu_model = pickle.load(open('model/tinubu_model_pickle.pkl', 'rb'))

sanwo_model = pickle.load(open('model/gubernitorial_models/sanwo-olu_model.pkl', 'rb'))
gbadebo_model = pickle.load(open('model/gubernitorial_models/gbadebo_model.pkl', 'rb'))
jandor_model = pickle.load(open('model/gubernitorial_models/jandor_model.pkl', 'rb'))
folarin_model = pickle.load(open('model/gubernitorial_models/folarin_model.pkl', 'rb'))
seyi_model = pickle.load(open('model/gubernitorial_models/seyi_model.pkl', 'rb'))
tonye_model = pickle.load(open('model/gubernitorial_models/tonye_model.pkl', 'rb'))
itubo_model = pickle.load(open('model/gubernitorial_models/itubo_model.pkl', 'rb'))
fubara_model = pickle.load(open('model/gubernitorial_models/fubara_model.pkl', 'rb'))
sani_model = pickle.load(open('model/gubernitorial_models/sani_model.pkl', 'rb'))
asake_model = pickle.load(open('model/gubernitorial_models/asake_model.pkl', 'rb'))
ashiru_model = pickle.load(open('model/gubernitorial_models/ashiru_model.pkl', 'rb'))
nnaji_model = pickle.load(open('model/gubernitorial_models/nnaji_model.pkl', 'rb'))
peter_model = pickle.load(open('model/gubernitorial_models/peter_model.pkl', 'rb'))
nentawe_model = pickle.load(open('model/gubernitorial_models/nentawe_model.pkl', 'rb'))
dakum_model = pickle.load(open('model/gubernitorial_models/dakum_model.pkl', 'rb'))
caleb_model = pickle.load(open('model/gubernitorial_models/caleb_model.pkl', 'rb'))
joel_model = pickle.load(open('model/gubernitorial_models/joel_model.pkl', 'rb'))
kefas_model = pickle.load(open('model/gubernitorial_models/kefas_model.pkl', 'rb'))

vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 2), max_df=500)
atiku_vectorizer = pickle.load(open('model/atiku_vectorizer.pkl', 'rb'))

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


def sentiment_json_format(result):
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
    for candidate in ['abubakar', 'obi', 'tinubu']:
        if (candidate == 'abubakar'):
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

        
        print('scheduling')

sched = BackgroundScheduler()
sched.add_job(sensor, 'cron', minute='0-59/10')

sched.start()


# compile extracted data into a list from which we pick individual data for the candidates
sensor()

atiku_df = pd.DataFrame(result_atiku.copy(),
                        columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
atiku_tweet_df = atiku_df['tweet']

obi_df = pd.DataFrame(result_obi.copy(),
                      columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
obi_tweet_df = obi_df['tweet']

tinubu_df = pd.DataFrame(result_tinubu.copy(),
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


def neutral_location(candidate_df, candidate_tweet_df, candidate_model):
    result = sentiment(candidate_tweet_df, candidate_model)
    candidate_df['sentiment'] = result
    neu_df = candidate_df[candidate_df['sentiment'] == 'neutral']
    counts = get_location_counts(neu_df)
    return counts


def positive_location(candidate_df, candidate_tweet_df, candidate_model):
    result = sentiment(candidate_tweet_df, candidate_model)
    candidate_df['sentiment'] = result
    neu_df = candidate_df[candidate_df['sentiment'] == 'positive']
    counts = get_location_counts(neu_df)
    return counts


def negative_location(candidate_df, candidate_tweet_df, candidate_model):
    result = sentiment(candidate_tweet_df, candidate_model)
    candidate_df['sentiment'] = result
    neu_df = candidate_df[candidate_df['sentiment'] == 'negative']
    counts = get_location_counts(neu_df)
    return counts


def get_random_sentiment(candidate_df, result):
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
    cleaned_data = atiku_tweet_df.apply(cleanText)
    clean_df = pd.DataFrame(cleaned_data, columns=['tweet'])
    vectorized = atiku_vectorizer.transform(clean_df['tweet'])
    vectorized_df = pd.DataFrame(vectorized.toarray(), columns=vectorizer.get_feature_names_out())
    result = atiku_model.predict(vectorized_df.values)
    return sentiment_json_format(result)


def obi_sentiment():
    result = sentiment(obi_tweet_df, obi_model)
    return sentiment_json_format(result)


def tinubu_sentiment():
    result = sentiment(tinubu_tweet_df, tinubu_model)
    return sentiment_json_format(result)


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
    return neutral_location(atiku_df, atiku_tweet_df, atiku_model)


def obi_neutral_location():
    return neutral_location(obi_df, obi_tweet_df, obi_model)


def tinubu_neutral_location():
    return neutral_location(tinubu_df, tinubu_tweet_df, tinubu_model)


def atiku_positive_location():
    return positive_location(atiku_df, atiku_tweet_df, atiku_model)


def obi_positive_location():
    return positive_location(obi_df, obi_tweet_df, obi_model)


def tinubu_positive_location():
    return positive_location(tinubu_df, tinubu_tweet_df, tinubu_model)


def atiku_negative_location():
    return negative_location(atiku_df, atiku_tweet_df, atiku_model)


def obi_negative_location():
    return negative_location(obi_df, obi_tweet_df, obi_model)


def tinubu_negative_location():
    return negative_location(tinubu_df, tinubu_tweet_df, tinubu_model)


@app.route('/api/v1/single-sentiment/<candidate>')
def get_single_sentiment(candidate: str):
    if candidate.lower() == 'abubakar':
        return atiku_single_tweet_sentiments()
    elif candidate.lower() == 'obi':
        return obi_single_tweet_sentiments()
    elif candidate.lower() == 'tinubu':
        return tinubu_single_tweet_sentiments()


@app.route('/api/v1/sentiments/<candidate>')
def get_sentiments(candidate: str):
    if candidate.lower() == 'abubakar':
        return atiku_sentiment()
    elif candidate.lower() == 'obi':
        return obi_sentiment()
    elif candidate.lower() == 'tinubu':
        return tinubu_sentiment()


@app.route('/api/v1/hashtags/<candidate>')
def get_hashtags(candidate: str):
    if candidate.lower() == 'abubakar':
        return get_atiku_hash_tag()
    elif candidate.lower() == 'obi':
        return get_obi_hash_tag()
    elif candidate.lower() == 'tinubu':
        return get_tinubu_hash_tag()


@app.route('/api/v1/mentions/<candidate>')
def get_mentions(candidate: str):
    if candidate.lower() == 'abubakar':
        return get_atiku_mention()
    elif candidate.lower() == 'obi':
        return get_obi_mention()
    elif candidate.lower() == 'tinubu':
        return get_tinubu_mention()


@app.route('/api/v1/neutral-location/<candidate>')
def get_neutral_locations(candidate: str):
    if candidate.lower() == 'abubakar':
        return atiku_neutral_location()
    elif candidate.lower() == 'obi':
        return obi_neutral_location()
    elif candidate.lower() == 'tinubu':
        return tinubu_neutral_location()


@app.route('/api/v1/positive-location/<candidate>')
def get_positive_locations(candidate: str):
    if candidate.lower() == 'abubakar':
        return atiku_positive_location()
    elif candidate.lower() == 'obi':
        return obi_positive_location()
    elif candidate.lower() == 'tinubu':
        return tinubu_positive_location()


@app.route('/api/v1/negative-location/<candidate>')
def get_negative_locations(candidate: str):
    if candidate.lower() == 'abubakar':
        return atiku_negative_location()
    elif candidate.lower() == 'obi':
        return obi_negative_location()
    elif candidate.lower() == 'tinubu':
        return tinubu_negative_location()


if __name__ == "__main__":
    app.run()
