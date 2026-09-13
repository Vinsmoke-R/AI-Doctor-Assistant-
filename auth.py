from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import jwt
from datetime import datetime, timedelta

app    = FastAPI()
pwd    = CryptContext(schemes=["bcrypt"])
oauth2 = OAuth2PasswordBearer(tokenUrl="login")
SECRET = "keep-this-safe"

# ── fake db ──
users = {}

@app.get("/")
def root():
    return {"message":"Auth is working"}

# ── register ──
@app.post("/register")
def register(username: str, password: str):
    users[username] = pwd.hash(password)
    return {"msg": "registered"}

# ── login ──
@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    hashed = users.get(form.username)
    if not hashed or not pwd.verify(form.password, hashed):
        raise HTTPException(status_code=401, detail="Wrong credentials")
    
    token = jwt.encode({
        "sub": form.username,
        "exp": datetime.utcnow() + timedelta(hours=8)
    }, SECRET)
    return {"access_token": token, "token_type": "bearer"}

# ── protect route ──
def get_user(token = Depends(oauth2)):
    data = jwt.decode(token, SECRET, algorithms=["HS256"])
    return data["sub"]

@app.get("/patients")
def get_patients(user = Depends(get_user)):
    return {"user": user, "patients": []}