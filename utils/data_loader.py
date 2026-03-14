import pandas as pd
import numpy as np
from pathlib import Path

def load_books():
    
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir.parent / "data" / "books_with_emotions.csv"
    books = pd.read_csv(data_path)
    
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "../data/cover-not-found.jpg",
        books["large_thumbnail"],
    )

    return books