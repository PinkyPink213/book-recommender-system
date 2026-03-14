from sentence_transformers import CrossEncoder

def rerank(query, docs, top_k=5):
    
    pairs       = [(query, doc.page_content) for doc in docs]
    reranker    = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores      = reranker.predict(pairs)
    scored_docs = list(zip(scores, docs))
    scored_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)

    return [doc for score, doc in scored_docs[:top_k]]