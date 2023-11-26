import pandas as pd



class MovieData:
    def __init__(self, path: str) -> None:
        self.path = path
        self.df = pd.read_table(path, encoding_errors="ignore")

    # returns title of movie, for movie ID
    def get_title(self, id):
        return self.df[self.df["id"] == id]["title"].to_list()[0]