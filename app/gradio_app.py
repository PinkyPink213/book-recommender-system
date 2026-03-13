import gradio as gr
import json
from rag.pipeline import retrieve_semantic_recommendations
from utils.data_loader import load_books
from utils.query_builder import build_query


books = load_books()


def chat_recommend(describe, category, emotion, chat_history):

    answer = retrieve_semantic_recommendations(
        describe,
        category,
        emotion,
        books
    )

    answer = answer.strip()

    if answer.startswith("```"):
        answer = answer.replace("```json", "").replace("```", "").strip()

    data = json.loads(answer)

    gallery = []

    message = "Here are some books you might enjoy:\n\n"

    for item in data["books"]:

        isbn = int(item["isbn"])

        reason = item["reason"]

        book = books[books["isbn13"] == isbn].iloc[0]

        description = book["description"]

        truncated = " ".join(description.split()[:25]) + "..."

        caption = f"{book['title']} by {book['authors']}\n{truncated}"

        gallery.append((book["large_thumbnail"], caption))

        message += f"**{book['title']}** — {reason}\n\n"

    chat_history[-1] = (chat_history[-1][0], message)

    return chat_history, gallery


def add_user_message(describe, category, emotion, chat_history):

    query = build_query(describe, category, emotion)

    chat_history.append((query, None))

    return query, chat_history


with gr.Blocks() as dashboard:

    gr.Markdown("# 📚 Semantic Book Recommender")

    describe = gr.Textbox(label="Describe a book")

    category = gr.Dropdown(
        choices=["All"] + sorted(books["simple_categories"].unique()),
        value="All",
        label="Category"
    )

    emotion = gr.Dropdown(
        choices=["All","Happy","Surprising","Angry","Suspenseful","Sad"],
        value="All",
        label="Emotion"
    )

    submit = gr.Button("Find recommendations")

    chatbot = gr.Chatbot(height=200)

    gallery = gr.Gallery(columns=6)

    submit.click(
        fn=add_user_message,
        inputs=[describe, category, emotion, chatbot],
        outputs=[describe, chatbot]
    ).then(
        fn=chat_recommend,
        inputs=[describe, category, emotion, chatbot],
        outputs=[chatbot, gallery]
    )

dashboard.launch(server_name="0.0.0.0", server_port=7860)