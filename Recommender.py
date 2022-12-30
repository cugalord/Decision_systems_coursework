from UserItemData import UserItemData
from MovieData import MovieData
from RandomPredictor import RandomPredictor


class Recommender:
    def __init__(self, pred):
        self.usr_i_data = None
        self.pred = None
        self.predictor = pred

    def fit(self, usr_i_data):
        self.usr_i_data = usr_i_data
        self.predictor.fit(usr_i_data)

    def recommend(self, user_id=1, n=10, rec_seen=False):
        self.pred = self.predictor.predict(user_id)
        usr_movies = set(self.usr_i_data.df[self.usr_i_data.df["userID"] == user_id]["movieID"])
        if not rec_seen:
            pred = {key: val for key, val in self.pred.items() if key not in usr_movies}
        else:
            pred = {key: val for key, val in self.pred.items()}
        return [(key, val) for key, val in sorted(pred.items(), key=lambda i: i[1], reverse=True)][0:n]
