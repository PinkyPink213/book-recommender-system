def build_query(describe, category, emotion):

    if describe.strip():
        return f"I want {describe} in the {category} category with {emotion} emotional tone."

    return f"I want a book in the {category} category with {emotion} emotional tone."