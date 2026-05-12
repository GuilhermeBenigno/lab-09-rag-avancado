from sentence_transformers import CrossEncoder


def rerank_documents(original_query, retrieved_docs, top_n=3):
    cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

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
