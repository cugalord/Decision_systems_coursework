from UserItemData import UserItemData
from MovieData import MovieData
from Recommender import Recommender
import pandas as pd
import numpy as np


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


class SlopeOnePredictor:
    def __init__(self):
        self.df = None
        self.results = dict()

    def fit(self, uim):
        self.df = uim.df.groupby(["userID", "movieID"]).size().unstack()

        for col in self.df:
            self.df.loc[uim.df[uim.df["movieID"] == col]["userID"].tolist(), col] = [x for x in uim.df[uim.df["movieID"] == col]["rating"]]

        npmatrix = self.df.to_numpy()
        npmatrix[np.isnan(npmatrix)] = 0
        users = len(npmatrix)
        items = len(npmatrix[0])

        dev = np.zeros((items, items))
        for i in range(items):
            for j in range(items):
                if i == j:
                    break
                else:
                    dev_temp = get_dev(i, j, users, npmatrix)
                    dev[i][j] = dev_temp
                    dev[j][i] = (-1) * dev_temp

        pred_mat = np.zeros((users, items))
        for user in range(users):
            sel = np.where(npmatrix[user] != 0)[0]
            for j in range(items):
                pred_mat[user][j] = (np.sum(dev[j][sel] + npmatrix[user][sel])) / len(sel)

        self.df = pd.DataFrame(pred_mat, columns=self.df.columns, index=self.df.index)

    def predict(self, user_id):
        items = {key: 0 for key in self.df.columns}
        for key in items.keys():
            items[key] = self.df.loc[user_id, key]
        return items
