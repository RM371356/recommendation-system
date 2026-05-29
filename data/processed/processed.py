import pandas as pd

path_rating = "../data/raw/ratings.csv"
path_movies = "../data/raw/movies.csv"

ratings = pd.read_csv(path_rating)
movies = pd.read_csv(path_movies)

print(ratings.head())
print(movies.head())
