import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from vectorDB import mongo_doc_to_text

# two models — one for encoding, one for LangChain wrapper
_encoder = SentenceTransformer("BAAI/bge-base-en-v1.5")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
chroma_client = chromadb.PersistentClient(path="./chroma_db")


def build_vector_store(mongo_docs: list, uid: str):
    collection_name = f"patient_{uid}".replace("-", "_")

    texts = [mongo_doc_to_text(doc) for doc in mongo_docs]
    metadatas = [{"file_name": doc.get("file_name", "unknown")} for doc in mongo_docs]
    ids = [str(doc.get("file_name", i)) for i, doc in enumerate(mongo_docs)]
    embeddings = _encoder.encode(texts).tolist()

    collection = chroma_client.get_or_create_collection(name=collection_name)

    existing = collection.get(ids=ids)
    existing_ids = set(existing["ids"])
    new_indices = [i for i, id_ in enumerate(ids) if id_ not in existing_ids]

    if new_indices:
        collection.add(
            ids=       [ids[i]        for i in new_indices],
            documents= [texts[i]      for i in new_indices],
            metadatas= [metadatas[i]  for i in new_indices],
            embeddings=[embeddings[i] for i in new_indices]
        )

    # ✅ return LangChain Chroma — same as load_vector_store
    return Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embedding_model
    )