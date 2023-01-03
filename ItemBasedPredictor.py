from UserItemData import UserItemData
from MovieData import MovieData
from Recommender import Recommender
import pandas as pd
import numpy as np
import math


class ItemBasedPredictor:
    def __init__(self, min_values=0, threshold=0):
        self.avg_ratings = None
        self.usr_i_data = None
        self.min_values = min_values
        self.threshold = threshold

    def fit(self, usr_i_data):
        self.usr_i_data = usr_i_data
        self.df = usr_i_data.df.copy()
        self.avg_ratings = self.df.groupby("userID")["rating"].mean()
        self.df["rating"] = (self.df.set_index(["userID"])["rating"] - self.avg_ratings).values
        self.df.drop(columns=["date_day", "date_month", "date_year", "date_hour", "date_minute", "date_second"],
                     inplace=True)

        npmatrix = pd.DataFrame(columns=self.df["movieID"].unique(), index=self.df["movieID"].unique()).fillna(
            0).to_numpy().astype("float64")
        for i in range(len(npmatrix)):
            for j in range(len(npmatrix)):
                if i == j or npmatrix[i][j] != 0.0:
                    continue

                midi = self.df["movieID"].unique()[i]
                midj = self.df["movieID"].unique()[j]
                cross_users = set(self.df[self.df["movieID"] == midi]["userID"].values).intersection(
                    set(self.df[self.df["movieID"] == midj]["userID"].values))

                if len(cross_users) == 0 or len(cross_users) < self.min_values:
                    continue

                sim = np.sum(
                    self.df[(self.df["movieID"] == midi) & (self.df["userID"].isin(cross_users))]["rating"].values *
                    self.df[(self.df["movieID"] == midj) & (self.df["userID"].isin(cross_users))]["rating"].values)
                root = math.sqrt(np.sum(np.power(
                    self.df[(self.df["movieID"] == midi) & (self.df["userID"].isin(cross_users))]["rating"].values,
                    2))) * math.sqrt(np.sum(np.power(
                    self.df[(self.df["movieID"] == midj) & (self.df["userID"].isin(cross_users))]["rating"].values, 2)))

                npmatrix[i, j] = 0.0
                npmatrix[j, i] = 0.0
                if root == 0:
                    continue

                elif sim / root > self.threshold:
                    npmatrix[i, j] = sim / root
                    npmatrix[j, i] = sim / root

        self.df = pd.DataFrame(npmatrix, columns=self.df["movieID"].unique(), index=self.df["movieID"].unique())

    def predict(self, user_id):
        movieIDs = {key: 0 for key in self.usr_i_data.df["movieID"]}
        for movie in movieIDs.keys():
            prediction = 0
            divisor = 0
            for movie2 in movieIDs.keys():
                if movie != movie2 and len(self.usr_i_data.df[(self.usr_i_data.df["movieID"] == movie2) & (
                        self.usr_i_data.df["userID"] == user_id)]) != 0:
                    similarity = self.df.loc[movie, movie2]
                    prediction += similarity * self.usr_i_data.df[
                        (self.usr_i_data.df["movieID"] == movie2) & (self.usr_i_data.df["userID"] == user_id)][
                        "rating"].values[0]
                    divisor += similarity

            if divisor != 0:
                prediction = (self.avg_ratings[user_id] + (prediction / divisor)) / 2
            else:
                prediction = self.avg_ratings[user_id]
            movieIDs[movie] = prediction
        return movieIDs.copy()

    def similarity(self, p1, p2):
        return self.df.loc[p1, p2]

    def similar_items(self, item, n):
        movies = pd.DataFrame(self.df.loc[item, :]).sort_values(item, axis=0, ascending=False).iloc[:n, :].T.columns
        similar = pd.DataFrame(self.df.loc[item, :]).sort_values(item, axis=0, ascending=False).iloc[:n, :].T.values[0]

        return zip(movies, similar)
