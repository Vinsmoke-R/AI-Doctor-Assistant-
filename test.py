import pytesseract
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from ocr_service import extract_text
from llm_service import llm_extraction
from vectorDB import mongo_doc_to_text, build_vector_store
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

vector_store = None
load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["ai_doctor"]
collection = db["patients"]

p = collection.find_one({"uid": "PAT-76E91482"})

for report in p['reports']:
    print(report['file_name'])
    text = extract_text(report['file_data'])
    # print(text)
    result = llm_extraction(text)
    # print(result)
    text1 = mongo_doc_to_text(result)
    print(text1)

    # we are using only one vector store here 
    if vector_store is None:
        # build on first iteration
        vector_store = build_vector_store([result])
    else:
        # add to existing store
        vector_store.add_texts([text1], metadatas=[{"file_name": report['file_name']}])

# print(vector_store.index.ntotal) 

# 1. How many vectors stored
print("Total vectors:", vector_store.index.ntotal)

# 2. See the actual text stored
for doc_id, doc in vector_store.docstore._dict.items():
    print("ID:", doc_id)
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("------------------------------------------------------------------------------------------------")

# 3. See raw numbers (vectors)
import numpy as np
vectors = vector_store.index.reconstruct_n(0, vector_store.index.ntotal)
print("Vector shape:", vectors.shape)  # e.g (1, 768)
print("First vector (first 10 numbers):", vectors[0][:10])  # just first 10 to not flood terminal