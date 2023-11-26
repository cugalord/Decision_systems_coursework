from UserItemData import UserItemData
from MovieData import MovieData
from Recommender import Recommender
import pandas as pd
import numpy as np

#  Calculates the deviation of items at i and j
def get_dev(i, j, users, npmatrix):
    dev = 0
    user_cnt = 0
    for user in range(users):
        if (npmatrix[user][i] != 0) and (npmatrix[user][j] != 0):
            user_cnt += 1
            dev += npmatrix[user][i] - npmatrix[user][j]
    if user_cnt == 0:
        return 0
    return dev / user_cnt

# predicts ratings based on the Slope One method
class SlopeOnePredictor:
    def __init__(self):
        self.df = None
        self.results = dict()

    def fit(self, uim):
        # Create matrix movie=column, user=row, cell = rating
        self.df = uim.df.groupby(["userID", "movieID"]).size().unstack()

        for col in self.df:
            self.df.loc[uim.df[uim.df["movieID"] == col]["userID"].tolist(), col] = [x for x in uim.df[uim.df["movieID"] == col]["rating"]]

        npmatrix = self.df.to_numpy()
        npmatrix[np.isnan(npmatrix)] = 0
        users = len(npmatrix)
        movies = len(npmatrix[0])
        dev = np.zeros((movies, movies))

        # calculate deviations
        for i in range(movies):
            for j in range(movies):
                if i != j:
                    dev_temp = get_dev(i, j, users, npmatrix)
                    dev[i][j] = dev_temp
                    dev[j][i] = np.negative(dev_temp)
                else:
                    break

        # calculate prediction matrix
        pred_mat = np.zeros((users, movies))
        for user in range(users):
            sel = np.where(npmatrix[user] != 0)[0]
            user_sel = npmatrix[user][sel]
            for j in range(movies):
                pred_mat[user][j] = (np.sum(dev[j][sel] + user_sel)) / len(sel)

        self.df = pd.DataFrame(pred_mat, columns=self.df.columns, index=self.df.index)

    def predict(self, user_id):
        items = {key: 0 for key in self.df.columns}
        for key in items.keys():
            items[key] = self.df.loc[user_id, key]
        return items
