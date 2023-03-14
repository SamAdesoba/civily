import datetime
import snscrape.modules.twitter as sntwitter
import pickle

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
# lists to capture extracted data from twitter
gbadebo_result = []
jandor_result = []
sanwoolu_result = []

search_1 = f'(#GbadeboIsBetter OR #GRV4Lagos OR @GRVlagos) until:2022-11-30 since:2022-06-01'
search_2 = f'(#JandorFunke2023 OR #JafunEko OR @officialjandor) until:2022-11-30 since:2022-06-01'
search_3 = f'(#GreaterLagosRising OR @jidesanwoolu) until:2022-11-30 since:2022-06-01'

# time range for extracted twitter data
current_date = datetime.date.today()
last_date = datetime.timedelta(hours=72)


# function to extract twitter data
def scrape_gbadebo():
    gbadebo_result.clear()
    # search_atiku = f'(atikuabubakar OR atikuokowa OR #atikuokowa2023 OR #atikuabubakar OR #atikulated2023) until:{current_date} since:{current_date - last_date}'
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_1).get_items()):
        if i > 1000:
            break
        else:
            gbadebo_result.append(
                [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                 tweet.likeCount, tweet.retweetCount])
    return gbadebo_result


def scrape_jandor():
    jandor_result.clear()
    # search_obi = f'(peterobi OR #peterobi OR #obidatti2023) until:{current_date} since:{current_date - last_date}'
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_2).get_items()):
        if i > 1000:
            break
        else:
            jandor_result.append(
                [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                 tweet.likeCount, tweet.retweetCount])
    return jandor_result


def scrape_sanwoolu():
    sanwoolu_result.clear()
    # search_tinubu = f'(bolatinubu OR #bolatinubu OR #bat2023) until:{current_date} since:{current_date - last_date}'
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_3).get_items()):
        if i > 1000:
            break
        else:
            sanwoolu_result.append(
                [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                 tweet.likeCount, tweet.retweetCount])
    return sanwoolu_result
