import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_documents(path="data/manuals.json"):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_hnsw_index(documents, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embedder = SentenceTransformer(model_name)

    texts = [doc["text"] for doc in documents]
    embeddings = embedder.encode(texts, convert_to_numpy=True)

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    # HNSW explícito
    M = 32
    index = faiss.IndexHNSWFlat(dimension, M)
    index.hnsw.efConstruction = 100
    index.hnsw.efSearch = 64

    index.add(embeddings.astype("float32"))

    return index, embedder
