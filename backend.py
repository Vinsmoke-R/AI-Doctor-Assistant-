from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pydantic import BaseModel
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional
import uuid
import os
from typing import List

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database
client = MongoClient(os.getenv("MONGO_URI"))
db = client["ai_doctor"]
collection = db["patients"]
chat_collection = db["chat_history"]


# --- Models --- 
class ChatMessage(BaseModel):
    role: str        # "user" or "assistant"
    content: str

class Report(BaseModel):
    report_type: str
    file_name: str
    file_data: str   # base64 str 

class PatientCreate(BaseModel):
    name: str
    age: int
    gender: str
    blood_group: Optional[str] = None
    contact: Optional[str] = None
    medical_history: Optional[str] = None
    reports: List[Report] = []
 
class PatientUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    blood_group: Optional[str] = None
    contact: Optional[str] = None
    medical_history: Optional[str] = None
    reports: Optional[List[Report]] = None
 

# --- Routes ---
@app.get("/")
def root():
    return {"message": "AI Doctor Backend Running!"}
 

 # --- Helper ---
def serialize_reports(reports: List[Report]) -> list:
    return [
        {
            "report_id": "RPT-" + str(uuid.uuid4())[:8].upper(),
            "report_type": r.report_type,
            "file_name": r.file_name,
            "file_data": r.file_data,  # base64 string
            "uploaded_at": datetime.now().isoformat(),
        }
        for r in reports
    ]

# chat_get_data
@app.get("/patients/{uid}/chat")
def load_chat(uid:str):
    p = collection.find_one({"uid": uid})
    if not p:
        raise HTTPException(status_code=404, detail=f"Patient {uid} not found")
    
    # Fetch all messages for this patient
    messages = list(chat_collection.find(
        {"uid": uid},
        {"_id": 0}          # exclude mongo's _id field
    ))
    return {"uid": uid, "messages": messages}

# chat_post_data
@app.post("/patients/{uid}/chat")
def ai_msg(uid:str,msg:ChatMessage):
    p = collection.find_one({"uid": uid})
    if not p:
        raise HTTPException(status_code=404, detail=f"Patient {uid} not found")
    
    chat_collection.insert_one({
        "uid": uid,
        "role": msg.role,
        "content": msg.content,
        "timestamp": datetime.now().isoformat()
    })
    return {"status": "saved"}

# CREATE
@app.post("/patients", status_code=201)
def create_patient(patient: PatientCreate):
    uid = "PAT-" + str(uuid.uuid4())[:8].upper()   # uuid is here 
 
    patient_data = {
        "uid": uid,
        "name": patient.name,
        "age": patient.age,
        "gender": patient.gender,
        "blood_group": patient.blood_group,
        "contact": patient.contact,
        "medical_history": patient.medical_history,
        "reports": serialize_reports(patient.reports),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
 
    collection.insert_one(patient_data)
    patient_data.pop("_id")  # remove MongoDB's internal _id before returning
    return {"message": "Patient created successfully", "patient": patient_data}
 
# READ ALL
@app.get("/patients")
def get_all_patients():
    patients = []
    for p in collection.find():
        p.pop("_id")
        patients.append(p)
    return {"total": len(patients), "patients": patients}
 
 
# READ ONE
@app.get("/patients/{uid}")
def get_patient(uid: str):
    p = collection.find_one({"uid": uid})
    if not p:
        raise HTTPException(status_code=404, detail=f"Patient {uid} not found")
    p.pop("_id")
    return p
 
 
# UPDATE
@app.put("/patients/{uid}")
def update_patient(uid: str, updates: PatientUpdate):
    p = collection.find_one({"uid": uid})
    if not p:
        raise HTTPException(status_code=404, detail=f"Patient {uid} not found")

    update_data = updates.model_dump(exclude_none=True)
    update_data["updated_at"] = datetime.now().isoformat()

    # ✅ Append new reports to existing ones instead of overwriting
    if "reports" in update_data:
        new_reports = serialize_reports(updates.reports)
        collection.update_one({"uid": uid}, {"$push": {"reports": {"$each": new_reports}}})
        update_data.pop("reports")  # already handled above

    if update_data:  # push remaining scalar fields if any
        collection.update_one({"uid": uid}, {"$set": update_data})

    updated = collection.find_one({"uid": uid})
    updated.pop("_id")
    return {"message": "Patient updated successfully", "patient": updated}
 
# DELETE
@app.delete("/patients/{uid}")
def delete_patient(uid: str):
    p = collection.find_one({"uid": uid})
    if not p:
        raise HTTPException(status_code=404, detail=f"Patient {uid} not found")
 
    collection.delete_one({"uid": uid})
    return {"message": f"Patient {p['name']} deleted successfully", "uid": uid}