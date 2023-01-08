import datetime

import pandas as pd

from UserItemData import UserItemData
from MovieData import MovieData
from RandomPredictor import RandomPredictor
from Recommender import Recommender
from AveragePredictor import AveragePredictor
from ViewsPredictor import ViewsPredictor
from ItemBasedPredictor import ItemBasedPredictor
from SlopeOnePredictor import SlopeOnePredictor


def usr_i_data_test():
    print("UserItemData")
    uim = UserItemData('data/user_ratedmovies.dat')
    print("\t", uim.no_of_ratings())

    uim = UserItemData('data/user_ratedmovies.dat', start_date='12.1.2007', end_date='16.2.2008', min_ratings=100)
    print("\t", uim.no_of_ratings())


def movie_data_test():
    print("MovieData test")
    md = MovieData('data/movies.dat')
    print("\t", md.get_title(1))


def item_based_test():
    print("ItemBasedPredictor test")
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000)
    rp = ItemBasedPredictor()
    rec = Recommender(rp)
    rec.fit(uim)
    print("\t", "Similarity")
    print("\t\t", "Podobnost med filmoma 'Men in black'(1580) in 'Ghostbusters'(2716): ", rp.similarity(1580, 2716))
    print("\t\t", "Podobnost med filmoma 'Men in black'(1580) in 'Schindler's List'(527): ", rp.similarity(1580, 527))
    print("\t\t", "Podobnost med filmoma 'Men in black'(1580) in 'Independence day'(780): ", rp.similarity(1580, 780))

    print("\t", "Predictions for User ID 78: ")
    rec_items = rec.recommend(78, n=15, rec_seen=False)
    for idmovie, val in rec_items:
        print("\t\t", "Film: {}, ocena: {}".format(md.get_title(idmovie), val))


def rand_test():
    print("RandomPredictor test")
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat')
    rp = RandomPredictor(1, 5)
    rp.fit(uim)
    pred = rp.predict(78)
    print("\t", type(pred))
    items = [1, 3, 20, 50, 100]
    for item in items:
        print("\t", "Film: {}, ocena: {}".format(md.get_title(item), pred[item]))


def recommender_test():
    print("Recommender test")
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat')
    rp = RandomPredictor(1, 5)
    rec = Recommender(rp)
    rec.fit(uim)
    rec_items = rec.recommend(78, n=5, rec_seen=False)
    for idmovie, val in rec_items:
        print("\t", "Film: {}, ocena: {}".format(md.get_title(idmovie), val))


def avg_test():
    print("AveragePredictor test")
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat')
    ap = AveragePredictor(100)
    rec = Recommender(ap)
    rec.fit(uim)
    rec_items = rec.recommend(user_id=78, n=5, rec_seen=False)
    for idmovie, val in rec_items:
        print("\t", "Film: {}, ocena: {}".format(md.get_title(idmovie), val))


def views_test():
    print("ViewsPredictor test")
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat')
    ap = ViewsPredictor()
    rec = Recommender(ap)
    rec.fit(uim)
    rec_items = rec.recommend(user_id=78, n=5, rec_seen=False)
    for idmovie, val in rec_items:
        print("\t", "Film: {}, ocena: {}".format(md.get_title(idmovie), val))


def similar_items_test():
    print("similar_items test")
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000)
    rp = ItemBasedPredictor()
    rec = Recommender(rp)
    rec.fit(uim)
    rec_items = rp.similar_items(4993, 10)
    print("\t", 'Filmi podobni "The Lord of the Rings: The Fellowship of the Ring": ')
    for idmovie, val in rec_items:
        print("\t\t", "Film: {}, ocena: {}".format(md.get_title(idmovie), val))


def slope_one_test():
    print("SlopeOnePredictor test")
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000)
    rp = SlopeOnePredictor()
    rec = Recommender(rp)
    rec.fit(uim)
    print("\t", "Predictions for 78: ")
    rec_items = rec.recommend(78, n=15, rec_seen=False)
    for idmovie, val in rec_items:
        print("\t\t", "Film: {}, ocena: {}".format(md.get_title(idmovie), val))


def item_based_personal():
    print("ItemBasedPredictor personal")
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000)
    my_movies = {55820: 4.1,
                 296: 4.3,
                 318: 4.4,
                 3246: 3.1,
                 1917: 3.8,
                 3142: 3.8,
                 4105: 3.7,
                 1994: 3.2,
                 1258: 3.2,
                 1387: 2.9,
                 5785: 3.9,
                 1097: 3.0,
                 4382: 4.7,
                 47997: 3.8,
                 61132: 4.3,
                 46337: 3.5,
                 1196: 3.9}

    for movieid, rating in my_movies.items():
        dt = datetime.datetime.now()
        row = {"userID": [69420], "movieID": [movieid], "rating": [rating], "day": [dt.day], "month": [dt.month],
               "year": [dt.year], "date_hour": [dt.hour], "date_minute": [dt.minute], "date_second": [dt.second]}
        uim.df = pd.concat([uim.df, pd.DataFrame(row)], axis=0)

    rp = ItemBasedPredictor()
    rec = Recommender(rp)
    rec.fit(uim)
    print("\t", "Predictions for me: ")
    rec_items = rec.recommend(78, n=10, rec_seen=False)
    for idmovie, val in rec_items:
        print("\t\t", "Film: {}, ocena: {}".format(md.get_title(idmovie), val))

def top_20_most_similar():
    print("top 20 most similar movies")
    md = MovieData('data/movies.dat')
    uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000)
    rp = ItemBasedPredictor()
    rec = Recommender(rp)
    rec.fit(uim)

    movies = dict.fromkeys(rp.df.columns)
    for movie in movies.keys():
        # Find max for each movie but exclude itself.
        sim = rp.df.loc[rp.df.index != movie, movie].max()
        sim_movie = rp.df.loc[rp.df.index != movie, movie].idxmax()
        movies[movie] = (sim, sim_movie)

    top_20 = sorted(movies.items(), key=lambda item: item[1][0], reverse=True)[:20]
    for movie, sim in top_20:
        print("\tFilm1: \"{}\", Film2: \"{}\", similarity: {}".format(
            md.get_title(movie), md.get_title(sim[1]), sim[0]))

usr_i_data_test()
movie_data_test()
rand_test()
recommender_test()
avg_test()
views_test()
item_based_test()
top_20_most_similar()
similar_items_test()
item_based_personal()
slope_one_test()

