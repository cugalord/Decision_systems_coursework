class ViewsPredictor:
    def __init__(self) -> None:
        self.usr_i_data = None

    def fit(self, usr_i_data):
        self.usr_i_data = {key: 0 for key in usr_i_data.df["movieID"]}
        for key in self.usr_i_data.keys():
            self.usr_i_data[key] = usr_i_data.df[usr_i_data.df["movieID"] == key].shape[0]

    def predict(self, user_id):
        return self.usr_i_data.copy()