import pandas as pd
import numpy as np


def load_books():

    books = pd.read_csv("../data/books_with_emotions.csv")

    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "../data/cover-not-found.jpg",
        books["large_thumbnail"],
    )

    return books