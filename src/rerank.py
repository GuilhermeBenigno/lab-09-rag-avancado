from sentence_transformers import CrossEncoder


def rerank_documents(original_query, retrieved_docs, top_n=3):
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [
        [original_query, doc["text"]]
        for doc in retrieved_docs
    ]

    scores = cross_encoder.predict(pairs)

    reranked = []

    for doc, score in zip(retrieved_docs, scores):
        reranked.append({
            "id": doc["id"],
            "text": doc["text"],
            "cross_score": float(score)
        })

    reranked = sorted(
        reranked,
        key=lambda x: x["cross_score"],
        reverse=True
    )

    return reranked[:top_n]
