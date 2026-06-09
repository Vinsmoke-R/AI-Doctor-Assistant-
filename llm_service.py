from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import json

from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # current and reliable
    temperature=0,
    max_tokens = 2000,
)

def llm_extraction(text:str):
    prompt = f"""You are an assistant for a Neurology Doctor.
    You will receive raw OCR text from a medical document.
    Clean it up and convert it into a structured JSON format.
    Only include fields that are present in the text. 
    Return ONLY valid JSON, no markdown, no code fences, no explanation."""

    messages = messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=text)   # ← text is now actually passed in
    ]
    response = llm.invoke(messages)

    parsed = json.loads(response.content)  # ← convert string to dict
    
    return {
        "messages": response,
        "structured": parsed  # ← clean Python dict, ready for MongoDB
    }