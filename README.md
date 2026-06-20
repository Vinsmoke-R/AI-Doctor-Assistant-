# AI Doctor Assistant

An AI-powered patient management system with OCR report processing and RAG-based chat, built for internship/portfolio purposes.

## Architecture

┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  Streamlit App   │ ──────> │  FastAPI Backend  │ ──────> │  MongoDB Atlas   │
│  (Frontend)       │ <────── │  (Render)          │ <────── │  (Database)       │
└─────────────────┘         └──────────────────┘         └─────────────────┘
        │
        │ (OCR + LLM, runs client-side on Streamlit Cloud)
        ▼
  Tesseract OCR → Groq LLM → FAISS Vector Store (in-memory, per session)

### Flow


Add Patient — patient details + report files submitted → backend generates a unique UID (PAT-XXXXXXXX) → stored in MongoDB (reports saved as base64).
<div align="center">
  <img src="assets/Screenshot (117).png" width="1000">
  <p><em></em></p>
</div>
<div align="center">
  <img src="assets/Screenshot (116).png" width="1000">
  <p><em>Add Patient screen</em></p>
</div>


Search by UID — backend returns the patient record. Frontend runs OCR (Tesseract) on each report, cleans the extracted text via an LLM (Groq, llama-3.3-70b-versatile), and embeds it into a FAISS vector store for semantic search.
<div align="center">
  <img src="assets/Screenshot (127).jpeg" width="1000">
  <p><em></em></p>
</div>

<div align="center">
  <img src="assets/Screenshot (120).png" width="1000">
  <p><em>Search patient</em></p>
</div>



Chat (RAG) — user questions trigger a similarity search over the vector store; relevant report chunks + patient profile are injected into the LLM prompt. Chat history is persisted in a separate MongoDB collection (chat_history), keyed by UID.
<div align="center">
  <img src="assets/Screenshot (126).jpeg" width="1000">
  <p><em>Chat (RAG) </em></p>
</div>


Update / Delete — standard CRUD via the backend API; new reports can be appended without overwriting existing ones.
<div align="center">
  <img src="assets/Screenshot (128).jpeg" width="1000">
  <p><em>Update</em></p>
</div>
<div align="center">
  <img src="assets/Screenshot (125).png" width="1000">
  <p><em>Delete</em></p>
</div>

view all patient - you can view all the patient with their uid number 
<div align="center">
  <img src="assets/Screenshot (118).png" width="1000">
  <p><em>View all patient</em></p>
</div>

Tech Stack


Frontend: Streamlit
Backend: FastAPI
Database: MongoDB Atlas
OCR: Tesseract (via pytesseract)
LLM: Groq (llama-3.3-70b-versatile via langchain-groq)
Embeddings/Vector Search: sentence-transformers (BAAI/bge-base-en-v1.5) + FAISS


Deployment

ComponentHostNotesFrontendStreamlit Community CloudFree tierBackendRenderFree tier — spins down after ~15 min idle, cold start ~30–50sDatabaseMongoDB AtlasFree tier (M0), 512MB storage

Environment Variables

Render (backend):

KeyValueMONGO_URIMongoDB Atlas connection stringGROQ_API_KEYGroq API key (optional here unless backend calls Groq directly)

Streamlit Cloud (frontend) — Secrets:

tomlAPI_URL = "https://your-backend-name.onrender.com"
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"

Required deploy files (repo root)


requirements.txt — all Python dependencies
packages.txt — system package for Tesseract:


  tesseract-ocr

⚠️ Auto-Deploy Behavior — Important

Both Render and Streamlit Cloud are connected directly to this GitHub repo with auto-deploy enabled by default.

This means: any git push to the connected branch (usually main) immediately triggers a rebuild and redeploys the live app — automatically, with no manual approval step.

What this means in practice


✅ Bug fixes go live within ~1–2 minutes of pushing — convenient during active debugging.
⚠️ There is no staging environment. An untested or broken change pushed to main goes straight to the live, public-facing app.
⚠️ If you want to experiment without affecting the live site, work on a separate git branch and only merge to main when confident.


Recommended safer workflow

bashgit checkout -b experiment
# make changes, test locally
git push origin experiment
# once confirmed working:
git checkout main
git merge experiment
git push origin main   # this triggers the real deploy

Known Limitations (be upfront about these — they show engineering awareness, not weakness)


Vector store is in-memory per session — rebuilt every time a patient is searched fresh; not persisted across restarts. A production version would use a persistent vector DB (Pinecone, Chroma, or MongoDB Atlas Vector Search) keyed by UID.
Reports stored as base64 in MongoDB — works for small files; large PDFs/images would be better handled via object storage (e.g. S3) with MongoDB just storing a reference.
CORS is open (allow_origins=["*"]) — fine for a demo, would be restricted in production.
Render free tier cold starts — first request after inactivity can take 30–50 seconds.
OCR/LLM extraction failures are isolated — wrapped in try/except so one bad report doesn't crash the whole patient view, but that report won't be searchable via chat.


Local Development

bash# Backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend (in a separate terminal)
streamlit run streamlit_app.py

Set environment variables locally via a .env file (never commit this — ensure it's in .gitignore):

MONGO_URI=your_atlas_connection_string
GROQ_API_KEY=your_groq_key