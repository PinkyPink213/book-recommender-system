import pandas as pd
import numpy as np
import os

def load_books():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(base_dir, "..", "data", "books_with_emotions.csv")

    books = pd.read_csv(data_path )

    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "../data/cover-not-found.jpg",
        books["large_thumbnail"],
    )

    return books