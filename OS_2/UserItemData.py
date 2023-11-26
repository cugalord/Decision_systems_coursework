import pandas as pd


# class with pandas dataframe
# can be read from date, to date, or with minimal ratings for movie
class UserItemData:
    def __init__(self, path, start_date=None, end_date=None, min_ratings=None):
        self.path = path
        self.df = pd.read_table(path, encoding_errors="ignore")
        self.format_data = "%d.%m.%Y"
        self.df.rename(columns={"date_year": "year", "date_month": "month", "date_day": "day"}, inplace=True)


        if start_date is not None:
            self.df["date"] = pd.to_datetime(self.df[["day", "month", "year"]])
            self.df = self.df[self.df["date"] >= pd.to_datetime(start_date, format=self.format_data)]
            self.df.drop("date", axis=1)

        if end_date is not None:
            self.df["date"] = pd.to_datetime(self.df[["day", "month", "year"]])
            self.df = self.df[self.df["date"] < pd.to_datetime(end_date, format=self.format_data)]
            self.df.drop("date", axis=1)

        if min_ratings is not None:
            cnt = self.df["movieID"].value_counts()
            self.df = self.df[self.df["movieID"].isin(cnt[cnt >= min_ratings].index)]

    # returns no of ratings in the dataframe
    def no_of_ratings(self):
        return self.df.shape[0]
