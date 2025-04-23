import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds

rating_df = pd.read_csv('users.csv')
user_item_matrix = rating_df.pivot_table(index='user_id', columns='movie_title', values='rating')