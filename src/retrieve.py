import numpy as np
import faiss


def retrieve_top_k(query_text, embedder, index, documents, k=10):
    query_embedding = embedder.encode([query_text], convert_to_numpy=True)

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding.astype("float32"), k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "id": documents[idx]["id"],
            "text": documents[idx]["text"],
            "score": float(score)
        })

    return results
