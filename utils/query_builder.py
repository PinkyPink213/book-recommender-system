def build_query(describe, category):

    if describe.strip():
        return f"I want a book about {describe} in the {category} category."

    return f"I want a book in the {category} category."

