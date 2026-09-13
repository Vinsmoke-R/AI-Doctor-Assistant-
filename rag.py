from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st 

from chromaDB import build_vector_store, chroma_client, embedding_model

from llm_service import llm

# ── Models (loaded once) ──────────────────────────────────────────
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)
# ── Config ───────────────────────────────────────────────────────
RECENT_COUNT = 6          # always send the last N raw messages
SUMMARY_THRESHOLD = 15

# ── Stage 1: Chunking ──────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    return splitter.split_text(text)


# ── Stage 2: Indexing ──────────────────────────────────────────────
# def _new_empty_store():
#     sample_vector = embedding_model.encode(["init"])
#     embedding_dim = sample_vector.shape[1]
#     index = faiss.IndexFlatL2(embedding_dim)
#     return FAISS(
#         embedding_function=embedding_model.encode,
#         index=index,
#         docstore=InMemoryDocstore(),
#         index_to_docstore_id={}
#     )

# chroma vector store automatically create a vector store if there is None 
# vector_store = client.get_or_create_collection(
#         name="patient_documents"
#     )

def load_vector_store(uid: str):
    """
    Loads the persisted ChromaDB collection for a patient.
    Returns None if patient has no stored vectors yet.
    """
    collection_name = f"patient_{uid}".replace("-", "_")
    existing = [c.name for c in chroma_client.list_collections()]

    if collection_name not in existing:
        return None

    return Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embedding_model
    )


# ── Stage 3: Query Routing ──────────────────────────────────────────
PROFILE_KEYWORDS = ["age", "gender", "blood group", "contact", "name", "how old"]

def needs_retrieval(query: str) -> bool:
    q = query.lower()
    return not any(k in q for k in PROFILE_KEYWORDS)


# ── Stage 4: Query Rewriting ─────────────────────────────────────────
def rewrite_query(query: str, chat_history: list) -> str:
    recent = chat_history[-3:] if chat_history else []
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)

    prompt = f"""Rewrite the user's question into a clear, standalone search query
for retrieving relevant medical report sections. Keep it short (under 20 words).
Return ONLY the rewritten query, nothing else.

Recent conversation:
{history_text}

Question: {query}

Rewritten query:"""

    try:
        response = llm.invoke(prompt)
        rewritten = response.content.strip()
        return rewritten if rewritten else query
    except Exception:
        return query  # fallback to original on any failure


# ── Stage 5: Retrieval ────────────────────────────────────────────
def retrieve(vector_store, query: str, k: int = 5) -> list:
    if vector_store is None:
        return []
    return vector_store.similarity_search(query, k=k)


# ── Stage 6: Reranking ────────────────────────────────────────────
def rerank(query: str, docs: list, top_n: int = 3) -> list:
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]


# ── Stage 7: Context Compression (optional, only if long) ─────────
def compress_context(docs: list, query: str) -> str:
    combined = "\n\n".join(d.page_content for d in docs)
    if len(combined) < 800:
        return combined

    prompt = f"""Extract only the information relevant to answering this question:
"{query}"

From the following text:
{combined}

Relevant extract (concise):"""

    try:
        return llm.invoke(prompt).content.strip()
    except Exception:
        return combined  # fallback to raw context on failure

# ── Stage 7.5: Rolling Chat Summary (cached) ─────────────────────
def summarize_history(old_messages: list) -> str:
    if not old_messages:
        return ""

    convo_text = "\n".join(f"{m['role']}: {m['content']}" for m in old_messages)

    prompt = f"""Summarize the key facts, names, and important details mentioned
in this conversation between a doctor and an AI assistant about a patient.
Keep it short (under 100 words). Only include information that would matter
for answering future questions.

Conversation:
{convo_text}

Summary:"""

    try:
        return llm.invoke(prompt).content.strip()
    except Exception:
        return ""

def get_history_context(chat_history: list) -> tuple[list, str]:
    """
    Returns (recent_messages_to_send_raw, summary_of_older_messages).
    Short conversations: no summarization at all, send everything raw.
    Long conversations: cache the summary in session_state, only
    re-summarize when the 'older' portion actually grows.
    """
    if len(chat_history) <= SUMMARY_THRESHOLD:
        return chat_history, ""

    recent = chat_history[-RECENT_COUNT:]
    older = chat_history[:-RECENT_COUNT]

    if (st.session_state.get('summary_covers') != len(older)):
        st.session_state['chat_summary'] = summarize_history(older)
        st.session_state['summary_covers'] = len(older)

    return recent, st.session_state.get('chat_summary', "")


# ── Stage 8: Prompt Assembly ────────────────────────────────────────
def build_prompt(patient: dict, rag_context: str, summary_note: str = "") -> str:
    report_summary = "\n".join(
        f"- {r['report_type']}: {r['file_name']}" for r in patient.get("reports", [])
    ) or "none"

    summary_block = f"\n\nSummary of earlier conversation: {summary_note}" if summary_note else ""

    return f"""You are an AI doctor assistant. Here is the patient's full profile:
Name: {patient['name']}
Age: {patient['age']}
Gender: {patient['gender']}
Blood Group: {patient.get('blood_group') or 'not specified'}
Medical History: {patient.get('medical_history') or 'none'}
Reports on file: {report_summary}

Relevant Report Data:
{rag_context}{summary_block}

Answer all questions in context of this patient."""


# ── Orchestrator — the ONLY function frontend.py calls for chat ──────
def answer_query(vector_store, patient: dict, query: str, chat_history: list):
    """
    Runs the full pipeline: route -> rewrite -> retrieve -> rerank ->
    compress -> build prompt -> stream generation.
    Returns a generator of response chunks (for st.write_stream).
    """
    if vector_store is not None and needs_retrieval(query):
        search_query = rewrite_query(query, chat_history)
        docs = retrieve(vector_store, search_query, k=5)
        top_docs = rerank(search_query, docs, top_n=3)
        rag_context = compress_context(top_docs, query) if top_docs else "No relevant report data found."
    else:
        rag_context = "No report lookup needed for this question." if vector_store else "No reports available."

    recent, summary_note = get_history_context(chat_history)

    system_prompt = build_prompt(patient, rag_context, summary_note)

    messages = [HumanMessage(content=system_prompt)]
    for msg in recent:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    return llm.stream(messages)