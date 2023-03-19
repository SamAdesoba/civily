import flask
from flask_cors import CORS
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from Gubernitorial.gubernitorial_scrapper import *

app = flask.Flask(__name__)
CORS(app)

# trained models for each candidate and vectorizer
atiku_model = pickle.load(open('model/Presidential/atiku_model.pkl', 'rb'))
obi_model = pickle.load(open('model/Presidential/obi_model_pickle.pkl', 'rb'))
tinubu_model = pickle.load(open('model/Presidential/tinubu_model.pkl', 'rb'))

# vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 2), max_df=500)
atiku_vectorizer = pickle.load(open('model/Presidential/atiku_vectorizer.pkl', 'rb'))
obi_vectorizer = pickle.load(open('model/Presidential/atiku_vectorizer.pkl', 'rb'))
tinubu_vectorizer = pickle.load(open('model/Presidential/atiku_vectorizer.pkl', 'rb'))



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

        # if (candidate == '')

        # print('scheduling')


# sched = BackgroundScheduler()
# sched.add_job(sensor, 'cron', minute='0-59/10')

# sched.start()

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

gbadebo_df = pd.DataFrame(scrape_gbadebo(),
                          columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
gbadebo_tweet_df = gbadebo_df['tweet']

jandor_df = pd.DataFrame(scrape_jandor(),
                         columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
jandor_tweet_df = jandor_df['tweet']

sanwoolu_df = pd.DataFrame(scrape_sanwoolu(),
                           columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
sanwoolu_tweet_df = sanwoolu_df['tweet']

tonye_df = pd.DataFrame(scrape_tonye(),
                           columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
tonye_tweet_df = tonye_df['tweet']

itubo_df = pd.DataFrame(scrape_itubo(),
                           columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
itubo_tweet_df = itubo_df['tweet']

fubara_df = pd.DataFrame(scrape_fubara(),
                           columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
fubara_tweet_df = fubara_df['tweet']

folarin_df = pd.DataFrame(scrape_folarin(),
                           columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
folarin_tweet_df = folarin_df['tweet']

seyi_df = pd.DataFrame(scrape_seyi(),
                           columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
seyi_tweet_df = seyi_df['tweet']

sani_df = pd.DataFrame(scrape_sani(),
                           columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
sani_tweet_df = sani_df['tweet']

asake_df = pd.DataFrame(scrape_asake(),
                           columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
asake_tweet_df = asake_df['tweet']

ashiru_df = pd.DataFrame(scrape_ashiru(),
                           columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
ashiru_tweet_df = ashiru_df['tweet']

nentawe_df = pd.DataFrame(scrape_nentawe(),
                           columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
nentawe_tweet_df = nentawe_df['tweet']

dakum_df = pd.DataFrame(scrape_dakum(),
                           columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
dakum_tweet_df = dakum_df['tweet']

caleb_df = pd.DataFrame(scrape_caleb(),
                           columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
caleb_tweet_df = caleb_df['tweet']

nnaji_df = pd.DataFrame(scrape_nnaji(),
                           columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
nnaji_tweet_df = nnaji_df['tweet']

peter_df = pd.DataFrame(scrape_peter(),
                           columns=['date', 'username', 'sourceLabel', 'tweet', 'location', 'likeCount', 'retweetCount'])
peter_tweet_df = peter_df['tweet']

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


def sentiment(candidate_tweet_df, candidate_model, candidate_vectorizer):
    cleaned_data = candidate_tweet_df.apply(cleanText)
    clean_df = pd.DataFrame(cleaned_data, columns=['tweet'])
    vectorized = candidate_vectorizer.transform(clean_df['tweet'])
    # vectorized_df = pd.DataFrame(vectorized.toarray(), columns=vectorizer.get_feature_names_out())
    result = candidate_model.predict(vectorized)
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


def neutral_location(candidate_df, candidate_tweet_df, candidate_model, candidate_vectorizer):
    result = sentiment(candidate_tweet_df, candidate_model, candidate_vectorizer)
    candidate_df['sentiment'] = result
    neu_df = candidate_df[candidate_df['sentiment'] == 'neutral']
    counts = get_location_counts(neu_df)
    return counts


def positive_location(candidate_df, candidate_tweet_df, candidate_model, candidate_vectorizer):
    result = sentiment(candidate_tweet_df, candidate_model, candidate_vectorizer)
    candidate_df['sentiment'] = result
    neu_df = candidate_df[candidate_df['sentiment'] == 'positive']
    counts = get_location_counts(neu_df)
    return counts


def negative_location(candidate_df, candidate_tweet_df, candidate_model, candidate_vectorizer):
    result = sentiment(candidate_tweet_df, candidate_model, candidate_vectorizer)
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

gbadebo_df['mentions'] = gbadebo_df['tweet'].apply(mention)
gbadebo_mentions_list = gbadebo_df['mentions'].tolist()

jandor_df['mentions'] = jandor_df['tweet'].apply(mention)
jandor_mentions_list = jandor_df['mentions'].tolist()

sanwoolu_df['mentions'] = sanwoolu_df['tweet'].apply(mention)
sanwoolu_mentions_list = sanwoolu_df['mentions'].tolist()

tonye_df['mentions'] = tonye_df['tweet'].apply(mention)
tonye_mentions_list = tonye_df['mentions'].tolist()

itubo_df['mentions'] = itubo_df['tweet'].apply(mention)
itubo_mentions_list = itubo_df['mentions'].tolist()

fubara_df['mentions'] = fubara_df['tweet'].apply(mention)
fubara_mentions_list = fubara_df['mentions'].tolist()

folarin_df['mentions'] = folarin_df['tweet'].apply(mention)
folarin_mentions_list = folarin_df['mentions'].tolist()

seyi_df['mentions'] = seyi_df['tweet'].apply(mention)
seyi_mentions_list = seyi_df['mentions'].tolist()

sani_df['mentions'] = sani_df['tweet'].apply(mention)
sani_mentions_list = sani_df['mentions'].tolist()

asake_df['mentions'] = asake_df['tweet'].apply(mention)
asake_mentions_list = asake_df['mentions'].tolist()

ashiru_df['mentions'] = ashiru_df['tweet'].apply(mention)
ashiru_mentions_list = ashiru_df['mentions'].tolist()

nentawe_df['mentions'] = nentawe_df['tweet'].apply(mention)
nentawe_mentions_list = nentawe_df['mentions'].tolist()

dakum_df['mentions'] = dakum_df['tweet'].apply(mention)
dakum_mentions_list = dakum_df['mentions'].tolist()

caleb_df['mentions'] = caleb_df['tweet'].apply(mention)
caleb_mentions_list = caleb_df['mentions'].tolist()

nnaji_df['mentions'] = nnaji_df['tweet'].apply(mention)
nnaji_mentions_list = nnaji_df['mentions'].tolist()

peter_df['mentions'] = peter_df['tweet'].apply(mention)
peter_mentions_list = peter_df['mentions'].tolist()

atiku_df['hashtags'] = atiku_df['tweet'].apply(hashtag)
atiku_hashtags_list = atiku_df['hashtags'].tolist()

obi_df['hashtags'] = obi_df['tweet'].apply(hashtag)
obi_hashtags_list = obi_df['hashtags'].tolist()

tinubu_df['hashtags'] = tinubu_df['tweet'].apply(hashtag)
tinubu_hashtags_list = tinubu_df['hashtags'].tolist()

gbadebo_df['hashtags'] = gbadebo_df['tweet'].apply(hashtag)
gbadebo_hashtags_list = gbadebo_df['hashtags'].tolist()

jandor_df['hashtags'] = jandor_df['tweet'].apply(hashtag)
jandor_hashtags_list = jandor_df['hashtags'].tolist()

sanwoolu_df['hashtags'] = sanwoolu_df['tweet'].apply(hashtag)
sanwoolu_hashtags_list = sanwoolu_df['hashtags'].tolist()

tonye_df['hashtags'] = tonye_df['tweet'].apply(hashtag)
tonye_hashtags_list = tonye_df['hashtags'].tolist()

itubo_df['hashtags'] = itubo_df['tweet'].apply(hashtag)
itubo_hashtags_list = itubo_df['hashtags'].tolist()

fubara_df['hashtags'] = fubara_df['tweet'].apply(hashtag)
fubara_hashtags_list = fubara_df['hashtags'].tolist()

folarin_df['hashtags'] = folarin_df['tweet'].apply(hashtag)
folarin_hashtags_list = folarin_df['hashtags'].tolist()

seyi_df['hashtags'] = seyi_df['tweet'].apply(hashtag)
seyi_hashtags_list = seyi_df['hashtags'].tolist()

sani_df['hashtags'] = sani_df['tweet'].apply(hashtag)
sani_hashtags_list = sani_df['hashtags'].tolist()

asake_df['hashtags'] = asake_df['tweet'].apply(hashtag)
asake_hashtags_list = asake_df['hashtags'].tolist()

ashiru_df['hashtags'] = ashiru_df['tweet'].apply(hashtag)
ashiru_hashtags_list = ashiru_df['hashtags'].tolist()

nentawe_df['hashtags'] = nentawe_df['tweet'].apply(hashtag)
nentawe_hashtags_list = nentawe_df['hashtags'].tolist()

dakum_df['hashtags'] = dakum_df['tweet'].apply(hashtag)
dakum_hashtags_list = dakum_df['hashtags'].tolist()

caleb_df['hashtags'] = caleb_df['tweet'].apply(hashtag)
caleb_hashtags_list = caleb_df['hashtags'].tolist()

nnaji_df['hashtags'] = nnaji_df['tweet'].apply(hashtag)
nnaji_hashtags_list = nnaji_df['hashtags'].tolist()

peter_df['hashtags'] = peter_df['tweet'].apply(hashtag)
peter_hashtags_list = peter_df['hashtags'].tolist()


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


def get_gbadebo_mention():
    mentions_df = mentions(gbadebo_mentions_list)
    gbadebo_mentions = mentions_df[mentions_df['Mentions'] == 'GRVlagos']
    return gbadebo_mentions.set_index(gbadebo_mentions['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


def get_jandor_mention():
    mentions_df = mentions(jandor_mentions_list)
    jandor_mentions = mentions_df[mentions_df['Mentions'] == 'officialjandor']
    return jandor_mentions.set_index(jandor_mentions['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


def get_sanwoolu_mention():
    mentions_df = mentions(sanwoolu_mentions_list)
    sanwoolu_mentions = mentions_df[mentions_df['Mentions'] == 'jidesanwoolu']
    return sanwoolu_mentions.set_index(sanwoolu_mentions['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


def get_tonye_mention():
    mentions_df = mentions(tonye_mentions_list)
    tonye_mentions = mentions_df[mentions_df['Mentions'] == 'TonyeCole1']
    return tonye_mentions.set_index(tonye_mentions['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


def get_itubo_mention():
    mentions_df = mentions(itubo_mentions_list)
    itubo_mentions = mentions_df[mentions_df['Mentions'] == 'GovCan_Beatrice']
    return itubo_mentions.set_index(itubo_mentions['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


def get_fubara_mention():
    mentions_df = mentions(fubara_mentions_list)
    fubara_mentions = mentions_df[mentions_df['Mentions'] == 'SimFubaraKSC']
    return fubara_mentions.set_index(fubara_mentions['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


def get_folarin_mention():
    mentions_df = mentions(folarin_mentions_list)
    folarin_mentions = mentions_df[mentions_df['Mentions'] == 'teslimkfolarin']
    return folarin_mentions.set_index(folarin_mentions['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


def get_seyi_mention():
    mentions_df = mentions(seyi_mentions_list)
    seyi_mentions = mentions_df[mentions_df['Mentions'] == 'seyiamakinde']
    return seyi_mentions.set_index(seyi_mentions['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


def get_sani_mention():
    mentions_df = mentions(sani_mentions_list)
    sani_mentions = mentions_df[mentions_df['Mentions'] == 'ubasanius']
    return sani_mentions.set_index(sani_mentions['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


def get_asake_mention():
    mentions_df = mentions(asake_mentions_list)
    asake_mentions = mentions_df[mentions_df['Mentions'] == 'joe_asake']
    return asake_mentions.set_index(asake_mentions['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


def get_ashiru_mention():
    mentions_df = mentions(ashiru_mentions_list)
    ashiru_mentions = mentions_df[mentions_df['Mentions'] == 'IsaAshiruKudan']
    return ashiru_mentions.set_index(ashiru_mentions['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


def get_nentawe_mention():
    mentions_df = mentions(nentawe_mentions_list)
    nentawe_mentions = mentions_df[mentions_df['Mentions'] == 'nentawe1']
    return nentawe_mentions.set_index(nentawe_mentions['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


def get_dakum_mention():
    mentions_df = mentions(dakum_mentions_list)
    dakum_mentions = mentions_df[mentions_df['Mentions'] == 'PatrickDakum']
    return dakum_mentions.set_index(dakum_mentions['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


def get_caleb_mention():
    mentions_df = mentions(caleb_mentions_list)
    caleb_mentions = mentions_df[mentions_df['Mentions'] == 'CalebMutfwang']
    return caleb_mentions.set_index(caleb_mentions['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


def get_nnaji_mention():
    mentions_df = mentions(nnaji_mentions_list)
    nnaji_mentions = mentions_df[mentions_df['Mentions'] == 'Nwakaibie4Gov']
    return nnaji_mentions.set_index(nnaji_mentions['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


def get_peter_mention():
    mentions_df = mentions(peter_mentions_list)
    peter_mentions = mentions_df[mentions_df['Mentions'] == 'PNMbah']
    return peter_mentions.set_index(peter_mentions['Mentions']).drop('Mentions', axis=1).to_json(orient='columns')


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


def get_gbadebo_hash_tag():
    sort = hash_tag(gbadebo_hashtags_list)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


def get_jandor_hash_tag():
    sort = hash_tag(jandor_hashtags_list)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


def get_sanwoolu_hash_tag():
    sort = hash_tag(sanwoolu_hashtags_list)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


def get_tonye_hash_tag():
    sort = hash_tag(tonye_hashtags_list)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


def get_itubo_hash_tag():
    sort = hash_tag(itubo_hashtags_list)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


def get_fubara_hash_tag():
    sort = hash_tag(fubara_hashtags_list)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


def get_folarin_hash_tag():
    sort = hash_tag(folarin_hashtags_list)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


def get_seyi_hash_tag():
    sort = hash_tag(seyi_hashtags_list)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


def get_sani_hash_tag():
    sort = hash_tag(sani_hashtags_list)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


def get_asake_hash_tag():
    sort = hash_tag(asake_hashtags_list)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


def get_ashiru_hash_tag():
    sort = hash_tag(ashiru_hashtags_list)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


def get_nentawe_hash_tag():
    sort = hash_tag(nentawe_hashtags_list)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


def get_dakum_hash_tag():
    sort = hash_tag(dakum_hashtags_list)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


def get_caleb_hash_tag():
    sort = hash_tag(caleb_hashtags_list)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


def get_nnaji_hash_tag():
    sort = hash_tag(nnaji_hashtags_list)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


def get_peter_hash_tag():
    sort = hash_tag(peter_hashtags_list)
    return sort.set_index(sort['Hashtags']).drop("Hashtags", axis=1).to_json(orient='columns')


# General value_count sentiment functions
def atiku_sentiment():
    result = sentiment(atiku_tweet_df, atiku_model, atiku_vectorizer)
    return sentiment_json_format(result)


def obi_sentiment():
    result = sentiment(obi_tweet_df, obi_model, obi_vectorizer)
    return sentiment_json_format(result)


def tinubu_sentiment():
    result = sentiment(tinubu_tweet_df, tinubu_model, tinubu_vectorizer)
    return sentiment_json_format(result)


def gbadebo_sentiment():
    result = sentiment(gbadebo_tweet_df, gbadebo_model, gbadebo_vectorizer)
    return sentiment_json_format(result)


def jandor_sentiment():
    result = sentiment(jandor_tweet_df, jandor_model, jandor_vectorizer)
    return sentiment_json_format(result)


def sanwoolu_sentiment():
    result = sentiment(sanwoolu_tweet_df, sanwoolu_model, sanwoolu_vectorizer)
    return sentiment_json_format(result)


def tonye_sentiment():
    result = sentiment(tonye_tweet_df, tonye_model, tonye_vectorizer)
    return sentiment_json_format(result)


def itubo_sentiment():
    result = sentiment(itubo_tweet_df, itubo_model, itubo_vectorizer)
    return sentiment_json_format(result)


def fubara_sentiment():
    result = sentiment(fubara_tweet_df, fubara_model, fubara_vectorizer)
    return sentiment_json_format(result)


def folarin_sentiment():
    result = sentiment(folarin_tweet_df, folarin_model, folarin_vectorizer)
    return sentiment_json_format(result)


def seyi_sentiment():
    result = sentiment(seyi_tweet_df, seyi_model, seyi_vectorizer)
    return sentiment_json_format(result)


def sani_sentiment():
    result = sentiment(sani_tweet_df, sani_model, sani_vectorizer)
    return sentiment_json_format(result)


def asake_sentiment():
    result = sentiment(asake_tweet_df, asake_model, asake_vectorizer)
    return sentiment_json_format(result)


def ashiru_sentiment():
    result = sentiment(ashiru_tweet_df, ashiru_model, ashiru_vectorizer)
    return sentiment_json_format(result)


def nentawe_sentiment():
    result = sentiment(nentawe_tweet_df, nentawe_model, nentawe_vectorizer)
    return sentiment_json_format(result)


def dakum_sentiment():
    result = sentiment(dakum_tweet_df, dakum_model, dakum_vectorizer)
    return sentiment_json_format(result)


def caleb_sentiment():
    result = sentiment(caleb_tweet_df, caleb_model, caleb_vectorizer)
    return sentiment_json_format(result)


def nnaji_sentiment():
    result = sentiment(nnaji_tweet_df, nnaji_model, nnaji_vectorizer)
    return sentiment_json_format(result)


def peter_sentiment():
    result = sentiment(peter_tweet_df, peter_model, peter_vectorizer)
    return sentiment_json_format(result)


# Single sentiment functions
def atiku_single_tweet_sentiments():
    try:
        result = sentiment(atiku_tweet_df, atiku_model, atiku_vectorizer)
        filtered_result = get_random_sentiment(atiku_df, result)
        return filtered_result.to_dict(orient='records')
    except ValueError:
        return "data not enough to make up complete sentiments"


def obi_single_tweet_sentiments():
    try:
        result = sentiment(obi_tweet_df, obi_model, obi_vectorizer)
        filtered_result = get_random_sentiment(obi_df, result)
        return filtered_result.to_dict(orient='records')
    except ValueError:
        return "data not enough to make up complete sentiments"


def tinubu_single_tweet_sentiments():
    try:
        result = sentiment(tinubu_tweet_df, tinubu_model, tinubu_vectorizer)
        filtered_result = get_random_sentiment(tinubu_df, result)
        return filtered_result.to_dict(orient='records')
    except ValueError:
        return "data not enough to make up complete sentiments"


def gbadebo_single_tweet_sentiments():
    try:
        result = sentiment(gbadebo_tweet_df, gbadebo_model, gbadebo_vectorizer)
        filtered_result = get_random_sentiment(gbadebo_df, result)
        return filtered_result.to_dict(orient='records')
    except ValueError:
        return "data not enough to make up complete sentiments"


def jandor_single_tweet_sentiments():
    try:
        result = sentiment(jandor_tweet_df, jandor_model, jandor_vectorizer)
        filtered_result = get_random_sentiment(jandor_df, result)
        return filtered_result.to_dict(orient='records')
    except ValueError:
        return "data not enough to make up complete sentiments"


def sanwoolu_single_tweet_sentiments():
    try:
        result = sentiment(sanwoolu_tweet_df, sanwoolu_model, sanwoolu_vectorizer)
        filtered_result = get_random_sentiment(tinubu_df, result)
        return filtered_result.to_dict(orient='records')
    except ValueError:
        return "data not enough to make up complete sentiments"


def tonye_single_tweet_sentiments():
    try:
        result = sentiment(tonye_tweet_df, tonye_model, tonye_vectorizer)
        filtered_result = get_random_sentiment(tonye_df, result)
        return filtered_result.to_dict(orient='records')
    except ValueError:
        return "data not enough to make up complete sentiments"


def itubo_single_tweet_sentiments():
    try:
        result = sentiment(itubo_tweet_df, itubo_model, itubo_vectorizer)
        filtered_result = get_random_sentiment(itubo_df, result)
        return filtered_result.to_dict(orient='records')
    except ValueError:
        return "data not enough to make up complete sentiments"


def fubara_single_tweet_sentiments():
    try:
        result = sentiment(fubara_tweet_df, fubara_model, fubara_vectorizer)
        filtered_result = get_random_sentiment(fubara_df, result)
        return filtered_result.to_dict(orient='records')
    except ValueError:
        return "data not enough to make up complete sentiments"


def folarin_single_tweet_sentiments():
    try:
        result = sentiment(folarin_tweet_df, folarin_model, folarin_vectorizer)
        filtered_result = get_random_sentiment(folarin_df, result)
        return filtered_result.to_dict(orient='records')
    except ValueError:
        return "data not enough to make up complete sentiments"


def seyi_single_tweet_sentiments():
    try:
        result = sentiment(seyi_tweet_df, seyi_model, seyi_vectorizer)
        filtered_result = get_random_sentiment(seyi_df, result)
        return filtered_result.to_dict(orient='records')
    except ValueError:
        return "data not enough to make up complete sentiments"


def sani_single_tweet_sentiments():
    try:
        result = sentiment(sani_tweet_df, sani_model, sani_vectorizer)
        filtered_result = get_random_sentiment(sani_df, result)
        return filtered_result.to_dict(orient='records')
    except ValueError:
        return "data not enough to make up complete sentiments"


def asake_single_tweet_sentiments():
    try:
        result = sentiment(asake_tweet_df, asake_model, asake_vectorizer)
        filtered_result = get_random_sentiment(asake_df, result)
        return filtered_result.to_dict(orient='records')
    except ValueError:
        return "data not enough to make up complete sentiments"


def ashiru_single_tweet_sentiments():
    try:
        result = sentiment(ashiru_tweet_df, ashiru_model, ashiru_vectorizer)
        filtered_result = get_random_sentiment(ashiru_df, result)
        return filtered_result.to_dict(orient='records')
    except ValueError:
        return "data not enough to make up complete sentiments"


def nentawe_single_tweet_sentiments():
    try:
        result = sentiment(nentawe_tweet_df, nentawe_model, nentawe_vectorizer)
        filtered_result = get_random_sentiment(nentawe_df, result)
        return filtered_result.to_dict(orient='records')
    except ValueError:
        return "data not enough to make up complete sentiments"


def dakum_single_tweet_sentiments():
    try:
        result = sentiment(dakum_tweet_df, dakum_model, dakum_vectorizer)
        filtered_result = get_random_sentiment(dakum_df, result)
        return filtered_result.to_dict(orient='records')
    except ValueError:
        return "data not enough to make up complete sentiments"


def caleb_single_tweet_sentiments():
    try:
        result = sentiment(caleb_tweet_df, caleb_model, caleb_vectorizer)
        filtered_result = get_random_sentiment(caleb_df, result)
        return filtered_result.to_dict(orient='records')
    except ValueError:
        return "data not enough to make up complete sentiments"


def nnaji_single_tweet_sentiments():
    try:
        result = sentiment(nnaji_tweet_df, nnaji_model, nnaji_vectorizer)
        filtered_result = get_random_sentiment(nnaji_df, result)
        return filtered_result.to_dict(orient='records')
    except ValueError:
        return "data not enough to make up complete sentiments"


def peter_single_tweet_sentiments():
    try:
        result = sentiment(peter_tweet_df, peter_model, peter_vectorizer)
        filtered_result = get_random_sentiment(peter_df, result)
        return filtered_result.to_dict(orient='records')
    except ValueError:
        return "data not enough to make up complete sentiments"


# Location functions
def atiku_neutral_location():
    return neutral_location(atiku_df, atiku_tweet_df, atiku_model, atiku_vectorizer)


def obi_neutral_location():
    return neutral_location(obi_df, obi_tweet_df, obi_model, obi_vectorizer)


def tinubu_neutral_location():
    return neutral_location(tinubu_df, tinubu_tweet_df, tinubu_model, tinubu_vectorizer)


def gbadebo_neutral_location():
    return neutral_location(gbadebo_df, gbadebo_tweet_df, gbadebo_model, gbadebo_vectorizer)


def jandor_neutral_location():
    return neutral_location(jandor_df, jandor_tweet_df, jandor_model, jandor_vectorizer)


def sanwoolu_neutral_location():
    return neutral_location(sanwoolu_df, sanwoolu_tweet_df, sanwoolu_model, sanwoolu_vectorizer)


def tonye_neutral_location():
    return neutral_location(tonye_df, tonye_tweet_df, tonye_model, tonye_vectorizer)


def itubo_neutral_location():
    return neutral_location(itubo_df, itubo_tweet_df, itubo_model, itubo_vectorizer)


def fubara_neutral_location():
    return neutral_location(fubara_df, fubara_tweet_df, fubara_model, fubara_vectorizer)


def folarin_neutral_location():
    return neutral_location(folarin_df, folarin_tweet_df, folarin_model, folarin_vectorizer)


def seyi_neutral_location():
    return neutral_location(seyi_df, seyi_tweet_df, seyi_model, seyi_vectorizer)


def sani_neutral_location():
    return neutral_location(sani_df, sani_tweet_df, sani_model, sani_vectorizer)


def asake_neutral_location():
    return neutral_location(asake_df, asake_tweet_df, asake_model, asake_vectorizer)


def ashiru_neutral_location():
    return neutral_location(ashiru_df, ashiru_tweet_df, ashiru_model, ashiru_vectorizer)


def nentawe_neutral_location():
    return neutral_location(nentawe_df, nentawe_tweet_df, nentawe_model, nentawe_vectorizer)


def dakum_neutral_location():
    return neutral_location(dakum_df, dakum_tweet_df, dakum_model, dakum_vectorizer)


def caleb_neutral_location():
    return neutral_location(caleb_df, caleb_tweet_df, caleb_model, caleb_vectorizer)


def nnaji_neutral_location():
    return neutral_location(nnaji_df, nnaji_tweet_df, nnaji_model, nnaji_vectorizer)


def peter_neutral_location():
    return neutral_location(peter_df, peter_tweet_df, peter_model, peter_vectorizer)


def atiku_positive_location():
    return positive_location(atiku_df, atiku_tweet_df, atiku_model, atiku_vectorizer)


def obi_positive_location():
    return positive_location(obi_df, obi_tweet_df, obi_model, obi_vectorizer)


def tinubu_positive_location():
    return positive_location(tinubu_df, tinubu_tweet_df, tinubu_model, tinubu_vectorizer)


def gbadebo_positive_location():
    return positive_location(gbadebo_df, gbadebo_tweet_df, gbadebo_model, gbadebo_vectorizer)


def jandor_positive_location():
    return positive_location(jandor_df, jandor_tweet_df, jandor_model, jandor_vectorizer)


def sanwoolu_positive_location():
    return positive_location(sanwoolu_df, sanwoolu_tweet_df, sanwoolu_model, sanwoolu_vectorizer)


def tonye_positive_location():
    return positive_location(tonye_df, tonye_tweet_df, tonye_model, tonye_vectorizer)


def itubo_positive_location():
    return positive_location(itubo_df, itubo_tweet_df, itubo_model, itubo_vectorizer)


def fubara_positive_location():
    return positive_location(fubara_df, fubara_tweet_df, fubara_model, fubara_vectorizer)


def folarin_positive_location():
    return positive_location(folarin_df, folarin_tweet_df, folarin_model, folarin_vectorizer)


def seyi_positive_location():
    return positive_location(seyi_df, seyi_tweet_df, seyi_model, seyi_vectorizer)


def sani_positive_location():
    return positive_location(sani_df, sani_tweet_df, sani_model, sani_vectorizer)


def asake_positive_location():
    return positive_location(asake_df, asake_tweet_df, asake_model, asake_vectorizer)


def ashiru_positive_location():
    return positive_location(ashiru_df, ashiru_tweet_df, ashiru_model, ashiru_vectorizer)


def nentawe_positive_location():
    return positive_location(nentawe_df, nentawe_tweet_df, nentawe_model, nentawe_vectorizer)


def dakum_positive_location():
    return positive_location(dakum_df, dakum_tweet_df, dakum_model, dakum_vectorizer)


def caleb_positive_location():
    return positive_location(caleb_df, caleb_tweet_df, caleb_model, caleb_vectorizer)


def nnaji_positive_location():
    return positive_location(nnaji_df, nnaji_tweet_df, nnaji_model, nnaji_vectorizer)


def peter_positive_location():
    return positive_location(peter_df, peter_tweet_df, peter_model, peter_vectorizer)


def atiku_negative_location():
    return negative_location(atiku_df, atiku_tweet_df, atiku_model, atiku_vectorizer)


def obi_negative_location():
    return negative_location(obi_df, obi_tweet_df, obi_model, obi_vectorizer)


def tinubu_negative_location():
    return negative_location(tinubu_df, tinubu_tweet_df, tinubu_model, tinubu_vectorizer)


def gbadebo_negative_location():
    return negative_location(gbadebo_df, gbadebo_tweet_df, gbadebo_model, gbadebo_vectorizer)


def jandor_negative_location():
    return negative_location(jandor_df, jandor_tweet_df, jandor_model, jandor_vectorizer)


def sanwoolu_negative_location():
    return negative_location(sanwoolu_df, sanwoolu_tweet_df, sanwoolu_model, sanwoolu_vectorizer)


def tonye_negative_location():
    return negative_location(tonye_df, tonye_tweet_df, tonye_model, tonye_vectorizer)


def itubo_negative_location():
    return negative_location(itubo_df, itubo_tweet_df, itubo_model, itubo_vectorizer)


def fubara_negative_location():
    return negative_location(fubara_df, fubara_tweet_df, fubara_model, fubara_vectorizer)


def folarin_negative_location():
    return negative_location(folarin_df, folarin_tweet_df, folarin_model, folarin_vectorizer)


def seyi_negative_location():
    return negative_location(seyi_df, seyi_tweet_df, seyi_model, seyi_vectorizer)


def sani_negative_location():
    return negative_location(sani_df, sani_tweet_df, sani_model, sani_vectorizer)


def asake_negative_location():
    return negative_location(asake_df, asake_tweet_df, asake_model, asake_vectorizer)


def ashiru_negative_location():
    return negative_location(ashiru_df, ashiru_tweet_df, ashiru_model, ashiru_vectorizer)


def nentawe_negative_location():
    return negative_location(nentawe_df, nentawe_tweet_df, nentawe_model, nentawe_vectorizer)


def dakum_negative_location():
    return negative_location(dakum_df, dakum_tweet_df, dakum_model, dakum_vectorizer)


def caleb_negative_location():
    return negative_location(caleb_df, caleb_tweet_df, caleb_model, caleb_vectorizer)


def nnaji_negative_location():
    return negative_location(nnaji_df, nnaji_tweet_df, nnaji_model, nnaji_vectorizer)


def peter_negative_location():
    return negative_location(peter_df, peter_tweet_df, peter_model, peter_vectorizer)


@app.route('/api/v1/single-sentiment/<candidate>')
def get_single_sentiment(candidate: str):
    if candidate.lower() == 'abubakar':
        return atiku_single_tweet_sentiments()
    elif candidate.lower() == 'obi':
        return obi_single_tweet_sentiments()
    elif candidate.lower() == 'tinubu':
        return tinubu_single_tweet_sentiments()
    elif candidate.lower() == 'gbadebo':
        return gbadebo_single_tweet_sentiments()
    elif candidate.lower() == 'jandor':
        return jandor_single_tweet_sentiments()
    elif candidate.lower() == 'sanwoolu':
        return sanwoolu_single_tweet_sentiments()
    elif candidate.lower() == 'tonye':
        return tonye_single_tweet_sentiments()
    elif candidate.lower() == 'itubo':
        return itubo_single_tweet_sentiments()
    elif candidate.lower() == 'fubara':
        return fubara_single_tweet_sentiments()
    elif candidate.lower() == 'folarin':
        return folarin_single_tweet_sentiments()
    elif candidate.lower() == 'seyi':
        return seyi_single_tweet_sentiments()
    elif candidate.lower() == 'sani':
        return sani_single_tweet_sentiments()
    elif candidate.lower() == 'asake':
        return asake_single_tweet_sentiments()
    elif candidate.lower() == 'ashiru':
        return ashiru_single_tweet_sentiments()
    elif candidate.lower() == 'nentawe':
        return nentawe_single_tweet_sentiments()
    elif candidate.lower() == 'dakum':
        return dakum_single_tweet_sentiments()
    elif candidate.lower() == 'caleb':
        return caleb_single_tweet_sentiments()
    elif candidate.lower() == 'nnaji':
        return nnaji_single_tweet_sentiments()
    elif candidate.lower() == 'peter':
        return peter_single_tweet_sentiments()


@app.route('/api/v1/sentiments/<candidate>')
def get_sentiments(candidate: str):
    if candidate.lower() == 'abubakar':
        return atiku_sentiment()
    elif candidate.lower() == 'obi':
        return obi_sentiment()
    elif candidate.lower() == 'tinubu':
        return tinubu_sentiment()
    elif candidate.lower() == 'gbadebo':
        return gbadebo_sentiment()
    elif candidate.lower() == 'jandor':
        return jandor_sentiment()
    elif candidate.lower() == 'sanwoolu':
        return sanwoolu_sentiment()
    elif candidate.lower() == 'tonye':
        return tonye_sentiment()
    elif candidate.lower() == 'itubo':
        return itubo_sentiment()
    elif candidate.lower() == 'fubara':
        return fubara_sentiment()
    elif candidate.lower() == 'folarin':
        return folarin_sentiment()
    elif candidate.lower() == 'seyi':
        return seyi_sentiment()
    elif candidate.lower() == 'sani':
        return sani_sentiment()
    elif candidate.lower() == 'asake':
        return asake_sentiment()
    elif candidate.lower() == 'ashiru':
        return ashiru_sentiment()
    elif candidate.lower() == 'nentawe':
        return nentawe_sentiment()
    elif candidate.lower() == 'dakum':
        return dakum_sentiment()
    elif candidate.lower() == 'caleb':
        return caleb_sentiment()
    elif candidate.lower() == 'nnaji':
        return nnaji_sentiment()
    elif candidate.lower() == 'peter':
        return peter_sentiment()


@app.route('/api/v1/hashtags/<candidate>')
def get_hashtags(candidate: str):
    if candidate.lower() == 'abubakar':
        return get_atiku_hash_tag()
    elif candidate.lower() == 'obi':
        return get_obi_hash_tag()
    elif candidate.lower() == 'tinubu':
        return get_tinubu_hash_tag()
    elif candidate.lower() == 'gbadebo':
        return get_gbadebo_hash_tag()
    elif candidate.lower() == 'jandor':
        return get_jandor_hash_tag()
    elif candidate.lower() == 'sanwoolu':
        return get_sanwoolu_hash_tag()
    elif candidate.lower() == 'tonye':
        return get_tonye_hash_tag()
    elif candidate.lower() == 'itubo':
        return get_itubo_hash_tag()
    elif candidate.lower() == 'fubara':
        return get_fubara_hash_tag()
    elif candidate.lower() == 'folarin':
        return get_folarin_hash_tag()
    elif candidate.lower() == 'seyi':
        return get_seyi_hash_tag()
    elif candidate.lower() == 'sani':
        return get_sani_hash_tag()
    elif candidate.lower() == 'asake':
        return get_asake_hash_tag()
    elif candidate.lower() == 'ashiru':
        return get_ashiru_hash_tag()
    elif candidate.lower() == 'nentawe':
        return get_nentawe_hash_tag()
    elif candidate.lower() == 'dakum':
        return get_dakum_hash_tag()
    elif candidate.lower() == 'caleb':
        return get_caleb_hash_tag()
    elif candidate.lower() == 'nnaji':
        return get_nnaji_hash_tag()
    elif candidate.lower() == 'peter':
        return get_peter_hash_tag()


@app.route('/api/v1/mentions/<candidate>')
def get_mentions(candidate: str):
    if candidate.lower() == 'abubakar':
        return get_atiku_mention()
    elif candidate.lower() == 'obi':
        return get_obi_mention()
    elif candidate.lower() == 'tinubu':
        return get_tinubu_mention()
    elif candidate.lower() == 'gbadebo':
        return get_gbadebo_mention()
    elif candidate.lower() == 'jandor':
        return get_jandor_mention()
    elif candidate.lower() == 'sanwoolu':
        return get_sanwoolu_mention()
    elif candidate.lower() == 'tonye':
        return get_tonye_mention()
    elif candidate.lower() == 'itubo':
        return get_itubo_mention()
    elif candidate.lower() == 'fubara':
        return get_fubara_mention()
    elif candidate.lower() == 'folarin':
        return get_folarin_mention()
    elif candidate.lower() == 'seyi':
        return get_seyi_mention()
    elif candidate.lower() == 'sani':
        return get_sani_mention()
    elif candidate.lower() == 'asake':
        return get_asake_mention()
    elif candidate.lower() == 'ashiru':
        return get_ashiru_mention()
    elif candidate.lower() == 'nentawe':
        return get_nentawe_mention()
    elif candidate.lower() == 'dakum':
        return get_dakum_mention()
    elif candidate.lower() == 'caleb':
        return get_caleb_mention()
    elif candidate.lower() == 'nnaji':
        return get_nnaji_mention()
    elif candidate.lower() == 'peter':
        return get_peter_mention()


@app.route('/api/v1/neutral-location/<candidate>')
def get_neutral_locations(candidate: str):
    if candidate.lower() == 'abubakar':
        return atiku_neutral_location()
    elif candidate.lower() == 'obi':
        return obi_neutral_location()
    elif candidate.lower() == 'tinubu':
        return tinubu_neutral_location()
    elif candidate.lower() == 'gbadebo':
        return gbadebo_neutral_location()
    elif candidate.lower() == 'jandor':
        return jandor_neutral_location()
    elif candidate.lower() == 'sanwoolu':
        return sanwoolu_neutral_location()
    elif candidate.lower() == 'tonye':
        return tonye_neutral_location()
    elif candidate.lower() == 'itubo':
        return itubo_neutral_location()
    elif candidate.lower() == 'fubara':
        return fubara_neutral_location()
    elif candidate.lower() == 'folarin':
        return folarin_neutral_location()
    elif candidate.lower() == 'seyi':
        return seyi_neutral_location()
    elif candidate.lower() == 'sani':
        return sani_neutral_location()
    elif candidate.lower() == 'asake':
        return asake_neutral_location()
    elif candidate.lower() == 'ashiru':
        return ashiru_neutral_location()
    elif candidate.lower() == 'nentawe':
        return nentawe_neutral_location()
    elif candidate.lower() == 'dakum':
        return dakum_neutral_location()
    elif candidate.lower() == 'caleb':
        return caleb_neutral_location()
    elif candidate.lower() == 'nnaji':
        return nnaji_neutral_location()
    elif candidate.lower() == 'peter':
        return peter_neutral_location()


@app.route('/api/v1/positive-location/<candidate>')
def get_positive_locations(candidate: str):
    if candidate.lower() == 'abubakar':
        return atiku_positive_location()
    elif candidate.lower() == 'obi':
        return obi_positive_location()
    elif candidate.lower() == 'tinubu':
        return tinubu_positive_location()
    elif candidate.lower() == 'gbadebo':
        return gbadebo_positive_location()
    elif candidate.lower() == 'jandor':
        return jandor_positive_location()
    elif candidate.lower() == 'sanwoolu':
        return sanwoolu_positive_location()
    elif candidate.lower() == 'tonye':
        return tonye_positive_location()
    elif candidate.lower() == 'itubo':
        return itubo_positive_location()
    elif candidate.lower() == 'fubara':
        return fubara_positive_location()
    elif candidate.lower() == 'folarin':
        return folarin_positive_location()
    elif candidate.lower() == 'seyi':
        return seyi_positive_location()
    elif candidate.lower() == 'sani':
        return sani_positive_location()
    elif candidate.lower() == 'asake':
        return asake_positive_location()
    elif candidate.lower() == 'ashiru':
        return ashiru_positive_location()
    elif candidate.lower() == 'nentawe':
        return nentawe_positive_location()
    elif candidate.lower() == 'dakum':
        return dakum_positive_location()
    elif candidate.lower() == 'caleb':
        return caleb_positive_location()
    elif candidate.lower() == 'nnaji':
        return nnaji_positive_location()
    elif candidate.lower() == 'peter':
        return peter_positive_location()


@app.route('/api/v1/negative-location/<candidate>')
def get_negative_locations(candidate: str):
    if candidate.lower() == 'abubakar':
        return atiku_negative_location()
    elif candidate.lower() == 'obi':
        return obi_negative_location()
    elif candidate.lower() == 'tinubu':
        return tinubu_negative_location()
    elif candidate.lower() == 'gbadebo':
        return gbadebo_negative_location()
    elif candidate.lower() == 'jandor':
        return jandor_negative_location()
    elif candidate.lower() == 'sanwoolu':
        return sanwoolu_negative_location()
    elif candidate.lower() == 'tonye':
        return tonye_negative_location()
    elif candidate.lower() == 'itubo':
        return itubo_negative_location()
    elif candidate.lower() == 'fubara':
        return fubara_negative_location()
    elif candidate.lower() == 'folarin':
        return folarin_negative_location()
    elif candidate.lower() == 'seyi':
        return seyi_negative_location()
    elif candidate.lower() == 'sani':
        return sani_negative_location()
    elif candidate.lower() == 'asake':
        return asake_negative_location()
    elif candidate.lower() == 'ashiru':
        return ashiru_negative_location()
    elif candidate.lower() == 'nentawe':
        return nentawe_negative_location()
    elif candidate.lower() == 'dakum':
        return dakum_negative_location()
    elif candidate.lower() == 'caleb':
        return caleb_negative_location()
    elif candidate.lower() == 'nnaji':
        return nnaji_negative_location()
    elif candidate.lower() == 'peter':
        return peter_negative_location()


if __name__ == "__main__":
    app.run()
