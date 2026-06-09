from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss

# Initialize once, globally
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# fetch from mongoDB
def mongo_doc_to_text(doc):
    skip_fields = {"_id", "created_at", "updated_at", "__v"}  # fields that add no semantic value
    
    parts = []
    for key, value in doc.items():
        if key in skip_fields:
            continue
        if value is None or value == "" or value == []:
            continue
        # clean up the key name
        label = key.replace("_", " ").title()
        parts.append(f"{label}: {value}")
    
    return "\n".join(parts)

# Build vector store
def build_vector_store(mongo_docs):
    # turn each mongo doc into a flat string
    texts = [mongo_doc_to_text(doc) for doc in mongo_docs]
    metadatas = [{"patient_id": str(doc.get("_id", "unknown"))} for doc in mongo_docs]

    # Get embedding dimension
    sample_vector = embedding_model.encode([texts[0]])
    embedding_dim = sample_vector.shape[1]

    index = faiss.IndexFlatL2(embedding_dim) # is uses Euclidean distance (L2 distance)
    vector_store = FAISS(
        embedding_function=embedding_model.encode,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vector_store.add_texts(texts, metadatas=metadatas)
    return vector_store