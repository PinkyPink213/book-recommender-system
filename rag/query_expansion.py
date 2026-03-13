import os
from langchain_openai import ChatOpenAI


def query_expansion(query):

    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    prompt = f"""
Generate 3 alternative search queries similar to:
{query}

Return each query on a new line.
"""

    result = llm.invoke(prompt).content

    queries = result.split("\n")

    queries.append(query)

    return queries