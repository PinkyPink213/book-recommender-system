import os
from langchain_openai import ChatOpenAI


def book_recommend(context):

    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    prompt = f"""
    You are the book recommender assistant.

    Here are candidate books retrieved from the database:

    {context}

    Choose the best 3 books ONLY from the candidate list.
    Do not invent new books.

    Return JSON:

    {{
    "books":[
    {{
    "title":"",
    "isbn":"",
    "reason":""
    }}
    ]
    }}
    """

    answer = llm.invoke(prompt)

    return answer.content