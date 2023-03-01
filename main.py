from flask import Flask
from flask_restx import Api, Resource
import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
import pickle
import re
import datetime
import json


obi_model = pickle.load(open('model_training/log_reg_obi.pkl', 'rb'))
obi_vectorizer = pickle.load(open('model_training/vectorizer_obi.pkl', 'rb'))
# atiku_model = pickle.load(open('model_training/log_reg_atiku.pkl', 'rb'))
# atiku_vectorizer = pickle.load(open('model_training/vectorizer_atiku.pkl', 'rb'))


current_date = datetime.date.today()
last_date = datetime.timedelta(hours=24)

    
result_atiku = []
result_obi = []
result_tinubu = []


def clean_tweet(text):
    text = re.sub(r'@[A-Za-z0-9^\w]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'#(\w+)', '', text)
    text = re.sub(r'[^\w]', ' ', text)
    text = ' '.join(c for c in text.split() if c.isalpha())
    return text


app = Flask(__name__)
api = Api(app)
    
    
@api.route('/home')
class Index(Resource):
    def get(self):
        return '<h1>Predicting Twitter Users Sentiments</h1>'


@api.route('/extract/<name>')
class ExtractTweet(Resource):
    def get(self, name):
        # for i in name:
        match name:
            case 'atiku':
                search_atiku = f'(atikuabubakar OR atikuokowa OR #atikuokowa2023 OR #atikuabubakar OR #atikulated2023) until:{current_date} since:{current_date - last_date}'
                result_atiku.clear()
                
                for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_atiku).get_items()):
                    if i > 10:
                        break
                    else:
                        result_atiku.append([tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location, tweet.likeCount, tweet.retweetCount])
                        
                # df_atiku = pd.DataFrame(result_atiku, columns=['date', 'user', 'source', 'tweet', 'location', 'like_count', 'retweet_count'])
                
                # final_text = atiku_vectorizer.transform(df_atiku['tweet'])
                # prediction = atiku_model.predict(final_text)
                
                # return prediction.value_counts().to_json(orient='index')
                return 'success1'
            
            case 'obi':
                search_obi = f'(peterobi OR #peterobi OR #obidatti2023) until:{current_date} since:{current_date - last_date}'
                result_obi.clear()
            
                for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_obi).get_items()):
                    if i > 10:
                        break
                    else:
                        result_obi.append([tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location, tweet.likeCount, tweet.retweetCount])
                        
                return 'success2'
            
            case 'tinubu':
                search_tinubu = f'(bolatinubu OR #bolatinubu OR #bat2023) until:{current_date} since:{current_date - last_date}'
                result_tinubu.clear()
            
                for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_tinubu).get_items()):
                    if i > 10:
                        break
                    else:
                        result_tinubu.append([tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location, tweet.likeCount, tweet.retweetCount])
                        
                return 'success3'
            
            
@api.route('/predict')
class Predict(Resource):
    def get(self):
        # tweet = request.form['tweet']
        # df = pd.DataFrame([tweet], columns=['tweet'])
        
        df_obi = pd.DataFrame(result_obi, columns=['date', 'user', 'source', 'tweet', 'location', 'like_count', 'retweet_count'])
        print(df_obi.head())
        print("======================================")
        
        
        df_obi['tweet'] = df_obi['tweet'].apply(clean_tweet)
        
        # final_text = df['tweet']
        # final_text.iloc[0] = ' '.join(final_text.iloc[0])
        
        final_text = obi_vectorizer.transform(df_obi['tweet'])
        prediction = obi_model.predict(final_text)
        print(prediction)
        print("======================================")
        # return prediction
        # result = json.dumps(prediction)
        return json.dumps(str(prediction))
        # return prediction
            

if __name__ == '__main__':
    app.run(debug=True)