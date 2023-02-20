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

model = pickle.load(open('model/model_pickle.pkl', 'rb'))

vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 2), max_df=500)

# df = pd.read_csv('util/atiku.csv')
df_tweet = ['tweet']

last_date = datetime.date.today()
new_date = datetime.timedelta(hours=24)



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

@app.route('/home')
def sensor():
    print("I love my facilitator")
    num = random.randint(0,10)
    # result_1 = []
    # search_1 = f'(peterobi OR #peterobi OR #obidatti2023) until:{last_date} since:{last_date - new_date}'
    # for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_1).get_items()):
    #     if i > 10:
    #         break
    #     else:
    #         result_1.append([tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location, tweet.likeCount, tweet.retweetCount])

    # return pd.DataFrame(result_1, columns=['date', 'username', 'sourceLabel', 'content', 'location', 'likeCount', 'retweetCount'])
    return num



sched = BackgroundScheduler(daemon=True)
# sched.add_job(sensor,'interval',seconds=5)
sched.start()


@app.route('/', methods=['GET', 'POST'])
def sentiment():
    cleaned_data = df_tweet.apply(cleanText)

    clean_df = pd.DataFrame(cleaned_data, columns=['tweet'])

    vectorizer.fit(clean_df['tweet'].values)

    vectorized = vectorizer.transform(clean_df['tweet'])

    vectorized_df = pd.DataFrame(vectorized.toarray(), columns=vectorizer.get_feature_names_out())

    result = model.predict(vectorized_df.values)

    result_df = pd.DataFrame(result, columns=['Analysis'])

    format_result = result_df.value_counts().to_json(orient='columns')

    return reformat_json(format_result)




atexit.register(lambda: sched.shutdown())

if __name__ == "__main__":
    app.run()
