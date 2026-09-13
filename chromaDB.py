import chromadb
from sentence_transformers import SentenceTransformer
from vectorDB import mongo_doc_to_text

# ── created once, used everywhere ──
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
chroma_client = chromadb.PersistentClient(path="./chroma_db")


def build_vector_store(mongo_docs: list, uid: str):

    collection_name = f"patient_{uid}".replace("-", "_")

    texts = [mongo_doc_to_text(doc) for doc in mongo_docs]

    metadatas = [
        {"file_name": doc.get("file_name", "unknown")}
        for doc in mongo_docs
    ]

    ids = [
        str(doc.get("file_name", i))
        for i, doc in enumerate(mongo_docs)
    ]

    embeddings = embedding_model.encode(texts).tolist()

    collection = chroma_client.get_or_create_collection(name=collection_name)

    #  only add new documents — no duplicates
    existing = collection.get(ids=ids)
    existing_ids = set(existing["ids"])

    new_indices = [i for i, id in enumerate(ids) if id not in existing_ids]

    if new_indices:
        collection.add(
            ids=        [ids[i]        for i in new_indices],
            documents=  [texts[i]      for i in new_indices],
            metadatas=  [metadatas[i]  for i in new_indices],
            embeddings= [embeddings[i] for i in new_indices]
        )
        print(f"Added {len(new_indices)} new documents for {uid}")
    else:
        print(f"All docs already stored for {uid}, skipping")

    return collection