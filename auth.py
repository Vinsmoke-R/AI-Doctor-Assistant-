from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel
import os, uuid
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# ── Load from environment (never hardcode!) ──────────────────
SECRET     = os.getenv("JWT_SECRET_KEY")
ALGORITHM  = "HS256"
TOKEN_TTL  = int(os.getenv("TOKEN_TTL_HOURS",8)) # here 8 is default value

# ── App & security ────────────────────────────────────────────
app   = FastAPI(title="Auth Demo")
pwd   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth = OAuth2PasswordBearer(tokenUrl="/login")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory user store (replace with a real DB) ─────────────
# Structure: { username: { id, hashed_password } }
# users_db: dict[str, dict] = {}

client = MongoClient(os.getenv("MONGO_URI"))
db = client["ai_doctor"]
users = db["users"]            # new, created on first register
patients= db["patients"]         # existing
chat_collection = db["chat_history"]     # existing

users.create_index("username", unique=True)


# ── Schemas ───────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str
    password: str

class UserOut(BaseModel):
    id: str
    username: str

# ── Helpers ───────────────────────────────────────────────────
def make_token(username: str, user_id: str) -> str:
    payload = {
        "sub": username,
        "uid": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_TTL),
    }
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth)) -> dict:
    try:
        data = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
        username: str = data.get("sub")
        if username is None or not users.find_one({"username": username}, {"_id": 1}):
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"username": username, "id": data.get("uid")}
    except JWTError:
        raise HTTPException(status_code=401, detail="Token expired or invalid")

# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Auth API is live"}

@app.post("/register", status_code=201)
def register(body: RegisterRequest):
    if users.find_one({"username": body.username}, {"_id": 1}):
        raise HTTPException(status_code=409, detail="Username already taken")
    if len(body.password) < 6:
        raise HTTPException(status_code=422, detail="Password must be at least 6 characters")

    user_id = str(uuid.uuid4())
    result = users.insert_one({
        "username":        body.username,
        "hashed_password": pwd.hash(body.password),
    })
    user_id = str(result.inserted_id)

    token = make_token(body.username, user_id)
    return {
        "msg":          "Account created",
        "access_token": token,
        "token_type":   "bearer",
        "user":         {"id": user_id, "username": body.username},
    }

@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    record = users.find_one({"username":form.username})
    if not record or not pwd.verify(form.password, record["hashed_password"]):
        raise HTTPException(status_code=401, detail="Wrong username or password")

    user_id = str(record["_id"])
    token = make_token(form.username, user_id)
    return {
        "access_token": token,
        "token_type":"bearer",
        "user":{"id": user_id, "username": form.username},
    }

@app.get("/me", response_model=UserOut)
def me(current_user: dict = Depends(get_current_user)):
    return {"id": current_user["id"], "username": current_user["username"]}

@app.get("/patients")
def get_patients(current_user: dict = Depends(get_current_user)):
    docs = list(patients.find({}, {"_id": 0}).sort("created_at", -1))
    return {"requested_by": current_user["username"], "count": len(docs), "patients": docs}