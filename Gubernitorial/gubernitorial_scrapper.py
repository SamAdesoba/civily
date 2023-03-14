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
tonye_result = []
itubo_result = []
fubara_result = []
folarin_result = []
seyi_result = []
sani_result = []
asake_result = []
ashiru_result = []
nentawe_result = []
dakum_result = []
caleb_result = []
nnaji_result = []
peter_result = []
joel_result = []
kefas_result = []



# time range for extracted twitter data
current_date = datetime.date.today()
last_date = datetime.timedelta(hours=72)


search_1 = f'(#GbadeboIsBetter OR #GRV4Lagos OR @GRVlagos) until:{current_date} since:{current_date - last_date}'
search_2 = f'(#JandorFunke2023 OR #JafunEko OR @officialjandor) until:{current_date} since:{current_date - last_date}'
search_3 = f'(#GreaterLagosRising OR @jidesanwoolu) until:{current_date} since:{current_date - last_date}'
search_4 = f'(#tonyecole2023 OR #tonyecoleforgovernor OR @TonyeCole1) until:{current_date} since:{current_date - last_date}'
search_5 = f'(#beatriceituboforgovernor OR @GovCan_Beatrice) until:{current_date} since:{current_date - last_date}'
search_6 = f'(@SimFubaraKSC) until:{current_date} since:{current_date - last_date}'
search_7 = f'(@teslimkfolarin OR #tkf4itesiwajuoyo2023 OR #TKF2023) until:{current_date} since:{current_date - last_date}'
search_8 = f'(@seyiamakinde) until:{current_date} since:{current_date - last_date}'
search_9 = f'(@ubasanius) until:{current_date} since:{current_date - last_date}'
search_10 = f'(#AsakeKaduna OR @joe_asake) until:{current_date} since:{current_date - last_date}'
search_11 = f'(#AshiruAssurance2023 OR #KadunaForKudan @IsaAshiruKudan) until:{current_date} since:{current_date - last_date}'
search_12 = f'(@nentawe1) until:{current_date} since:{current_date - last_date}'
search_13 = f'(#DAKUMISCOMING OR #DakumPwajok2023 OR @PatrickDakum) until:{current_date} since:{current_date - last_date}'
search_14 = f'(#MutfwangIsComing OR @CalebMutfwang) until:{current_date} since:{current_date - last_date}'
search_15 = f'(#NWAKAIBIE4GOV OR @Nwakaibie4Gov) until:{current_date} since:{current_date - last_date}'
search_16 = f'(#PeterGoWork OR #ThePeterMbahWeKnow OR #VotePeterMbah2023 OR @PNMbah) until:{current_date} since:{current_date - last_date}'
search_17 = f'(@SenIkenya) until:{current_date} since:{current_date - last_date}'
search_18 = f'(#TarabaIsPDP OR #AAKmedia OR @hon_kefas) until:{current_date} since:{current_date - last_date}'


# function to extract twitter data
def scrape_gbadebo():
    gbadebo_result.clear()
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
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_3).get_items()):
        if i > 1000:
            break
        else:
            sanwoolu_result.append(
                [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                 tweet.likeCount, tweet.retweetCount])
    return sanwoolu_result


def scrape_tonye():
    tonye_result.clear()
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_4).get_items()):
        if i > 1000:
            break
        else:
            tonye_result.append(
                [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                 tweet.likeCount, tweet.retweetCount])
    return tonye_result


def scrape_itubo():
    itubo_result.clear()
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_5).get_items()):
        if i > 1000:
            break
        else:
            itubo_result.append(
                [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                 tweet.likeCount, tweet.retweetCount])
    return itubo_result


def scrape_fubara():
    fubara_result.clear()
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_6).get_items()):
        if i > 1000:
            break
        else:
            fubara_result.append(
                [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                 tweet.likeCount, tweet.retweetCount])
    return fubara_result


def scrape_folarin():
    folarin_result.clear()
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_7).get_items()):
        if i > 1000:
            break
        else:
            folarin_result.append(
                [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                 tweet.likeCount, tweet.retweetCount])
    return folarin_result


def scrape_seyi():
    seyi_result.clear()
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_8).get_items()):
        if i > 1000:
            break
        else:
            seyi_result.append(
                [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                 tweet.likeCount, tweet.retweetCount])
    return seyi_result


def scrape_sani():
    sani_result.clear()
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_9).get_items()):
        if i > 1000:
            break
        else:
            sani_result.append(
                [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                 tweet.likeCount, tweet.retweetCount])
    return sani_result


def scrape_asake():
    asake_result.clear()
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_10).get_items()):
        if i > 1000:
            break
        else:
            asake_result.append(
                [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                 tweet.likeCount, tweet.retweetCount])
    return asake_result


def scrape_ashiru():
    ashiru_result.clear()
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_11).get_items()):
        if i > 1000:
            break
        else:
            ashiru_result.append(
                [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                 tweet.likeCount, tweet.retweetCount])
    return ashiru_result


def scrape_nentawe():
    nentawe_result.clear()
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_12).get_items()):
        if i > 1000:
            break
        else:
            nentawe_result.append(
                [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                 tweet.likeCount, tweet.retweetCount])
    return nentawe_result


def scrape_dakum():
    dakum_result.clear()
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_13).get_items()):
        if i > 1000:
            break
        else:
            dakum_result.append(
                [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                 tweet.likeCount, tweet.retweetCount])
    return dakum_result


def scrape_caleb():
    caleb_result.clear()
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_14).get_items()):
        if i > 1000:
            break
        else:
            caleb_result.append(
                [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                 tweet.likeCount, tweet.retweetCount])
    return caleb_result


def scrape_nnaji():
    nnaji_result.clear()
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_15).get_items()):
        if i > 1000:
            break
        else:
            nnaji_result.append(
                [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                 tweet.likeCount, tweet.retweetCount])
    return nnaji_result


def scrape_peter():
    peter_result.clear()
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_16).get_items()):
        if i > 1000:
            break
        else:
            peter_result.append(
                [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                 tweet.likeCount, tweet.retweetCount])
    return peter_result


def scrape_joel():
    joel_result.clear()
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_17).get_items()):
        if i > 1000:
            break
        else:
            joel_result.append(
                [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                 tweet.likeCount, tweet.retweetCount])
    return joel_result


def scrape_kefas():
    kefas_result.clear()
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_18).get_items()):
        if i > 1000:
            break
        else:
            kefas_result.append(
                [tweet.date, tweet.user.username, tweet.sourceLabel, tweet.content, tweet.user.location,
                 tweet.likeCount, tweet.retweetCount])
    return kefas_result