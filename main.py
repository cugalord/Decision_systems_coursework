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


"""
RESULTS

UserItemData
	 855598
	 73584
MovieData test
	 Toy story
RandomPredictor test
	 <class 'dict'>
	 Film: Toy story, ocena: 3
	 Film: Grumpy Old Men, ocena: 3
	 Film: Money Train, ocena: 3
	 Film: The Usual Suspects, ocena: 5
	 Film: City Hall, ocena: 2
Recommender test
	 Film: The Abyss, ocena: 5
	 Film: Die Hard, ocena: 5
	 Film: Superman, ocena: 5
	 Film: Hart's War, ocena: 5
	 Film: Home Alone 3, ocena: 5
AveragePredictor test
	 Film: The Usual Suspects, ocena: 4.225944245560473
	 Film: The Godfather: Part II, ocena: 4.146907937910189
	 Film: Cidade de Deus, ocena: 4.116538340205236
	 Film: The Dark Knight, ocena: 4.10413904093503
	 Film: 12 Angry Men, ocena: 4.103639627096175
ViewsPredictor test
	 Film: The Lord of the Rings: The Fellowship of the Ring, ocena: 1576
	 Film: The Lord of the Rings: The Two Towers, ocena: 1528
	 Film: The Lord of the Rings: The Return of the King, ocena: 1457
	 Film: The Silence of the Lambs, ocena: 1431
	 Film: Shrek, ocena: 1404
ItemBasedPredictor test
	 Similarity
		 Podobnost med filmoma 'Men in black'(1580) in 'Ghostbusters'(2716):  0.23395523176756633
		 Podobnost med filmoma 'Men in black'(1580) in 'Schindler's List'(527):  0.0
		 Podobnost med filmoma 'Men in black'(1580) in 'Independence day'(780):  0.42466125844687624
	 Predictions for User ID 78: 
		 Film: Shichinin no samurai, ocena: 4.1604255346899635
		 Film: The Usual Suspects, ocena: 4.1598990035688015
		 Film: The Silence of the Lambs, ocena: 4.150210791271142
		 Film: Sin City, ocena: 4.121901722984434
		 Film: Monsters, Inc., ocena: 4.0913487080066435
		 Film: The Incredibles, ocena: 4.0861074311757575
		 Film: The Lord of the Rings: The Fellowship of the Ring, ocena: 4.058954193209058
		 Film: Batman Begins, ocena: 4.055765042884983
		 Film: Die Hard, ocena: 4.045515940651294
		 Film: Rain Man, ocena: 4.018325761014159
		 Film: The Lord of the Rings: The Return of the King, ocena: 3.99267686416339
		 Film: A Beautiful Mind, ocena: 3.9901293845673034
		 Film: Good Will Hunting, ocena: 3.987198542996294
		 Film: The Lord of the Rings: The Two Towers, ocena: 3.9532962920826806
		 Film: Indiana Jones and the Last Crusade, ocena: 3.8810463877243455
top 20 most similar movies
	Film1: "The Lord of the Rings: The Two Towers", Film2: "The Lord of the Rings: The Return of the King", similarity: 0.8439842148481417
	Film1: "The Lord of the Rings: The Return of the King", Film2: "The Lord of the Rings: The Two Towers", similarity: 0.8439842148481417
	Film1: "The Lord of the Rings: The Fellowship of the Ring", Film2: "The Lord of the Rings: The Two Towers", similarity: 0.8231885401761888
	Film1: "Kill Bill: Vol. 2", Film2: "Kill Bill: Vol. 2", similarity: 0.7372340224381029
	Film1: "Kill Bill: Vol. 2", Film2: "Kill Bill: Vol. 2", similarity: 0.7372340224381029
	Film1: "Star Wars", Film2: "Star Wars: Episode V - The Empire Strikes Back", similarity: 0.7021321132220318
	Film1: "Star Wars: Episode V - The Empire Strikes Back", Film2: "Star Wars", similarity: 0.7021321132220318
	Film1: "Ace Ventura: Pet Detective", Film2: "The Mask", similarity: 0.6616471778494046
	Film1: "The Mask", Film2: "Ace Ventura: Pet Detective", similarity: 0.6616471778494046
	Film1: "Star Wars: Episode VI - Return of the Jedi", Film2: "Star Wars: Episode V - The Empire Strikes Back", similarity: 0.5992253753778948
	Film1: "Independence Day", Film2: "Star Wars: Episode I - The Phantom Menace", similarity: 0.5610426219249997
	Film1: "Star Wars: Episode I - The Phantom Menace", Film2: "Independence Day", similarity: 0.5610426219249997
	Film1: "Austin Powers: The Spy Who Shagged Me", Film2: "Ace Ventura: Pet Detective", similarity: 0.5546511205201551
	Film1: "Speed", Film2: "Pretty Woman", similarity: 0.5452283115904596
	Film1: "Pretty Woman", Film2: "Speed", similarity: 0.5452283115904596
	Film1: "Mrs. Doubtfire", Film2: "The Mask", similarity: 0.5398021259282235
	Film1: "The Matrix Reloaded", Film2: "Star Wars: Episode I - The Phantom Menace", similarity: 0.539553095856011
	Film1: "Pulp Fiction", Film2: "Reservoir Dogs", similarity: 0.5325845218198639
	Film1: "Reservoir Dogs", Film2: "Pulp Fiction", similarity: 0.5325845218198639
	Film1: "The Shawshank Redemption", Film2: "The Usual Suspects", similarity: 0.517724533955058
similar_items test
	 Filmi podobni "The Lord of the Rings: The Fellowship of the Ring": 
		 Film: The Lord of the Rings: The Two Towers, ocena: 0.8231885401761888
		 Film: The Lord of the Rings: The Return of the King, ocena: 0.8079374897442496
		 Film: Star Wars: Episode V - The Empire Strikes Back, ocena: 0.2396194307349645
		 Film: Star Wars, ocena: 0.2196558652707407
		 Film: The Matrix, ocena: 0.2151555270688023
		 Film: Raiders of the Lost Ark, ocena: 0.19944276706345015
		 Film: The Usual Suspects, ocena: 0.18321188451910753
		 Film: Blade Runner, ocena: 0.16399681315410275
		 Film: Schindler's List, ocena: 0.16105905138148702
		 Film: Monty Python and the Holy Grail, ocena: 0.15780453798519137
ItemBasedPredictor personal
	 Predictions for me: 
		 Film: The Usual Suspects, ocena: 4.158191051120011
		 Film: Shichinin no samurai, ocena: 4.156351596370569
		 Film: The Silence of the Lambs, ocena: 4.152385812629725
		 Film: Sin City, ocena: 4.120795876527069
		 Film: Monsters, Inc., ocena: 4.1077176401463245
		 Film: Armageddon, ocena: 4.107558139534884
		 Film: The Evil Dead, ocena: 4.107558139534884
		 Film: The Incredibles, ocena: 4.085091822937998
		 Film: Batman Begins, ocena: 4.0719175781711785
		 Film: The Lord of the Rings: The Fellowship of the Ring, ocena: 4.067238625617369
SlopeOnePredictor test
	 Predictions for 78: 
		 Film: The Usual Suspects, ocena: 4.335258910037142
		 Film: The Lord of the Rings: The Fellowship of the Ring, ocena: 4.176253710580998
		 Film: The Lord of the Rings: The Return of the King, ocena: 4.175658164006772
		 Film: The Silence of the Lambs, ocena: 4.143370751103905
		 Film: Shichinin no samurai, ocena: 4.1312943805612505
		 Film: The Lord of the Rings: The Two Towers, ocena: 4.104738621697874
		 Film: Indiana Jones and the Last Crusade, ocena: 3.9891308248979476
		 Film: The Incredibles, ocena: 3.9869953130473985
		 Film: Good Will Hunting, ocena: 3.9760142737997723
		 Film: Batman Begins, ocena: 3.963328799538746
		 Film: Sin City, ocena: 3.960207926545804
		 Film: A Beautiful Mind, ocena: 3.9312917632653925
		 Film: Rain Man, ocena: 3.9216066900959454
		 Film: Monsters, Inc., ocena: 3.903567578941103
		 Film: Finding Nemo, ocena: 3.9027764265477227
"""