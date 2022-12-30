class AveragePredictor:
    def __init__(self, b = 0) -> None:
        self.usr_i_data = None
        self.b = b
        if b < 0:
            raise Exception("AveragePredictor constructor error: argument b < 0")

    def fit(self, usr_i_data) -> None:
        def calculate(k):
            vs = sum(usr_i_data.df[usr_i_data.df["movieID"] == k]["rating"])
            g_avg = usr_i_data.df["rating"].sum() / usr_i_data.df.shape[0]
            n = usr_i_data.df[usr_i_data.df["movieID"] == k].shape[0]
            return (vs + self.b * g_avg) / (n + self.b)

        self.usr_i_data = {key: 0 for key in usr_i_data.df["movieID"]}
        for key in self.usr_i_data.keys():
            self.usr_i_data[key] = calculate(key)

    def predict(self, user_id):
        return self.usr_i_data.copy()