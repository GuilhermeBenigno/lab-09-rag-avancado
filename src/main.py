from build_index import load_documents, build_hnsw_index
from hyde_query import generate_hyde_document
from retrieve import retrieve_top_k
from rerank import rerank_documents


def main():
    documents = load_documents()

    print("Construindo índice HNSW...")
    index, embedder = build_hnsw_index(documents)

    query = "dor de cabeça latejante e luz incomodando"

    print("\nQuery original:")
    print(query)

    hyde_doc = generate_hyde_document(query)

    print("\nDocumento hipotético gerado pelo HyDE:")
    print(hyde_doc)

    retrieved_docs = retrieve_top_k(
        query_text=hyde_doc,
        embedder=embedder,
        index=index,
        documents=documents,
        k=10
    )

    print("\nTop-10 recuperados pelo HNSW:")
    for doc in retrieved_docs:
        print(f"[{doc['id']}] Score: {doc['score']:.4f} | {doc['text']}")

    final_docs = rerank_documents(query, retrieved_docs, top_n=3)

    print("\nTop-3 finais após Cross-Encoder:")
    for doc in final_docs:
        print(f"[{doc['id']}] Score: {doc['cross_score']:.4f} | {doc['text']}")


if __name__ == "__main__":
    main()
