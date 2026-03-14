import os
from openai import OpenAI

def book_recommend(context, query):

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a book recommendation assistant. Only choose books from the provided candidate list."
            },
            {
                "role": "user",
                "content": f"""
Candidate books retrieved from the database:

{context}

User request:
{query}

Choose EXACTLY 3 books from the candidate list that best match the user request.

Rules:
- Only choose books from the candidate list
- Do not invent books
- Prefer books that match the theme and category in the query

Return JSON only:

{{
  "books": [
    {{
      "title": "",
      "isbn": "",
      "reason": ""
    }}
  ]
}}
"""
            }
        ]
    )

    return response.choices[0].message.content