from UserItemData import UserItemData
from MovieData import MovieData
import random as rand

# predicts ratings randomly
class RandomPredictor:
    def __init__(self, min, max):
        self.usr_i_data = None
        self.min_rating = min
        self.max_rating = max

    def fit(self, usr_i_data):
        self.usr_i_data = {key: 0 for key in usr_i_data.df["movieID"]}

    def predict(self, user_id):
        for item in self.usr_i_data.keys():
            self.usr_i_data[item] = rand.randint(self.min_rating, self.max_rating)
        return self.usr_i_data.copy()
