import snscrape.modules.twitter as sntwitter
import random
import atexit
import pandas as pd
import numpy as np
import pickle
import re
import datetime
import json

current_date = datetime.date.today()
last_date = datetime.timedelta(hours=24)

result_atiku = []
result_obi = []
result_tinubu = []


# with open(f'log_reg.pkl', 'rb') as file1:
#     model = pickle.load(file1)
# with open(f'vectorizer.pkl', 'rb') as file2:
#     vectorizer = pickle.load(file2)

def clean_tweet(text):
    text = re.sub(r'@[A-Za-z0-9^\w]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'#(\w+)', '', text)
    text = re.sub(r'[^\w]', ' ', text)
    text = ' '.join(c for c in text.split() if c.isalpha())
    return text


# def my_scheduled_job():
#     count = random.random()
#     print(count)
#     return count

# def get_tweet_parameter(names):
#     names = ''
#     match "atiku":


# def extract_tweet_atiku():

#     search_atiku = f'(atikuabubakar OR atikuokowa OR #atikuokowa2023 OR #atikuabubakar OR #atikulated2023) until:{current_date} since:{last_date - current_date}'

#     for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_atiku).get_items()):
#         if i > 10:
#             break
#         else:
#             result_atiku.append([tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location, tweet.likeCount, tweet.retweetCount])

#     df_atiku = pd.DataFrame(result_atiku, columns=['date', 'user', 'source', 'tweet', 'location', 'like_count', 'retweet_count'])

#     return df_atiku


# def extract_tweet_obi():

#     search_obi = f'(peterobi OR #peterobi OR #obidatti2023) until:{current_date} since:{last_date - current_date}'

#     for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_obi).get_items()):
#         if i > 10:
#             break
#         else:
#             result_obi.append([tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location, tweet.likeCount, tweet.retweetCount])

#     df_obi = pd.DataFrame(result_obi, columns=['date', 'user', 'source', 'tweet', 'location', 'like_count', 'retweet_count'])

#     return df_obi


# def extract_tweet_tinubu():

#     search_tinubu = f'(bolatinubu OR #bolatinubu OR #bat2023) until:{current_date} since:{last_date - current_date}'

#     for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_tinubu).get_items()):
#         if i > 10:
#             break
#         else:
#             result_tinubu.append([tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location, tweet.likeCount, tweet.retweetCount])

#     df_tinubu = pd.DataFrame(result_tinubu, columns=['date', 'user', 'source', 'tweet', 'location', 'like_count', 'retweet_count'])

#     return df_tinubu


# @app.route('/')
# def index():

#     if request.method == 'GET':
#         print("======================================")
#         return f'<h1>Predicting Twitter Users Sentiments</h1>'
#         # return render_template('index.html')

# @app.route('/extract')
def extract_tweet():
    for i in ['atiku', 'obi', 'tinubu']:
        if (i == 'atiku'):
            search_atiku = f'(atikuabubakar OR atikuokowa OR #atikuokowa2023 OR #atikuabubakar OR #atikulated2023) until:{current_date} since:{current_date - last_date}'
            result_atiku.clear()

            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_atiku).get_items()):
                if i > 10:
                    break
                else:
                    result_atiku.append(
                        [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                         tweet.likeCount, tweet.retweetCount])

            df_atiku = pd.DataFrame(result_atiku, columns=['date', 'user', 'source', 'tweet', 'location', 'like_count',
                                                           'retweet_count'])

            # return 'success1'

        elif (i == 'obi'):
            search_obi = f'(peterobi OR #peterobi OR #obidatti2023) until:{current_date} since:{current_date - last_date}'
            result_obi.clear()

            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_obi).get_items()):
                if i > 10:
                    break
                else:
                    result_obi.append(
                        [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                         tweet.likeCount, tweet.retweetCount])

            df_obi = pd.DataFrame(result_obi, columns=['date', 'user', 'source', 'tweet', 'location', 'like_count',
                                                       'retweet_count'])
            print(df_obi.head())
            print('==================================================')
            return 'success2'

        else:
            search_tinubu = f'(bolatinubu OR #bolatinubu OR #bat2023) until:{current_date} since:{current_date - last_date}'
            result_tinubu.clear()

            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_tinubu).get_items()):
                if i > 10:
                    break
                else:
                    result_tinubu.append(
                        [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                         tweet.likeCount, tweet.retweetCount])

            df_tinubu = pd.DataFrame(result_tinubu,
                                     columns=['date', 'user', 'source', 'tweet', 'location', 'like_count',
                                              'retweet_count'])

            return 'success3'
