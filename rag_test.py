from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.messages import HumanMessage, AIMessage


from vectorDB import mongo_doc_to_text
from llm_service import llm

# chunking 
def chunk_text(text:str)-> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20
    )
    return splitter.split_text(text)

# indexing
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
def new_vector():
    pass 

def indexing(vector_store, report_doc: dict, file_name: str):
    text = mongo_doc_to_text(report_doc)
    chunks = chunk_text(text)
    if not vector_store:
        vector_store = new_vector()
    for chunk in chunks:
        vector_store.add_text(chunk)
    return vector_store

#query routing 
words = ["age", "gender", "blood group", "contact", "name", "how old"]
def query_routing(query:str)-> bool:
    q = query.lower()
    for k in words:
        if k in q:
            return False
    return True

# rewrite query 
def rewrite_query(query:str,chat_history:list[str])->str:
    prompt = f"""
Use {query} and {chat_history} to extract the relevant information and generate a query to ask the llm and length should be in between 20 and 30 words
"""
    response = llm.invoke(prompt)
    return response.content

# retrival 
def retrival(vector_store, query:str, k:int=3)->list:
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}  # Fetch the top 2 most relevant chunks
    )
    retrieved_docs = retriever.invoke(query)
    return retrieved_docs

# re-rank
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def re_rank(query:str, doc:list,)->list:
    pass

# query summarizier
def query_compress(query: str, doc: list) -> str:
    prompt = f"""
Extract only the relevant information from {doc} to answer this question {query}
"""
    response = llm.invoke(prompt)
    return response.content.strip()

# prompt generation
def prompt_generate(patient : dict, compressed_data : str):
    return f"""You are an AI doctor assistant here is the data of patient 
        Name : {patient['name']}
        Age : {patient['age']}
        Gender : {patient['gender']}
        Blood Group : {patient['Blood Group']}
        Contact : {patient['contact']}
and here is the relevant information of the patient {compressed_data}
    """

# output 
def answer(vector_store, patient : dict, query : str, chat_history : list):
    if vector_store is not None and query_routing(query):
        new_query = rewrite_query(query)
        retrived_docs = retrival(vector_store, new_query,3)
        reranked_list = re_rank(new_query,retrived_docs)
        rag_context = query_compress(new_query, reranked_list)

    else: 
        rag_context = "We cannot retrive the text from the documents"

    system_prompt = prompt_generate(patient,rag_context)

    message = [HumanMessage(content=system_prompt)]

    return llm.stream(message)