import os
from langchain_openai import ChatOpenAI
from openai import OpenAI

def query_expansion(query: str):

    prompt = f"""
Generate 3 alternative search queries similar to the following user query.

User query: {query}

Return ONLY the queries, one per line.
"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    result  = response.choices[0].message.content.strip()
    queries = [q.strip() for q in result.split("\n") if q.strip()]
    queries.append(query)

    return queries