import os
import io
import ssl
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from pypdf import PdfReader
from bs4 import BeautifulSoup as bs

import cohere

# -----------------------------
# 1) Load environment variables
# -----------------------------
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "").strip()
COHERE_MODEL = os.getenv("COHERE_MODEL", "command-r").strip()

# For production, check environment variable
if not COHERE_API_KEY:
    COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "").strip()

if not COHERE_API_KEY:
    raise RuntimeError(
        "COHERE_API_KEY not set. Please set it as an environment variable."
    )

# -----------------------------
# 2) NLTK setup
# -----------------------------
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except Exception:
    pass

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

ps = PorterStemmer()
CHUNK_SIZE = 1000
NUMBER_OF_MATCHES = 3

# -----------------------------
# 3) FastAPI app
# -----------------------------
app = FastAPI(title="RAG Chatbot API")

# CORS - Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for CSS, JS, images if needed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Create static directory if it doesn't exist
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -----------------------------
# 4) In-memory store
# -----------------------------
STATE = {
    "original_docs": [],
    "documents": [],
    "vectorizer": TfidfVectorizer(),
    "vectors": None,
}

co = cohere.Client(api_key=COHERE_API_KEY)

# -----------------------------
# 5) Helper functions
# -----------------------------
def process_text(txt: str, chunk_size: int = CHUNK_SIZE):
    try:
        sentences = nltk.sent_tokenize(txt)
    except Exception:
        sentences = txt.split(". ")

    original_text = []
    processed_text = []
    chunk = ""

    for s in sentences:
        if len(chunk) + len(s) + 1 < chunk_size:
            chunk += " " + s
        else:
            c = chunk.strip()
            if c:
                original_text.append(c)
                processed_text.append(" ".join(ps.stem(w) for w in c.split()))
            chunk = s

    c = chunk.strip()
    if c:
        original_text.append(c)
        processed_text.append(" ".join(ps.stem(w) for w in c.split()))
    return original_text, processed_text

def read_pdf_bytes(data: bytes):
    reader = PdfReader(io.BytesIO(data))
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + " "
    return process_text(text)

def read_html_bytes(data: bytes):
    soup = bs(data, "html.parser")
    text = soup.get_text(separator=" ")
    return process_text(text)

def read_text_bytes(data: bytes):
    text = data.decode("utf-8", errors="ignore")
    return process_text(text)

def add_document(processed_chunks: List[str]):
    STATE["documents"].extend(processed_chunks)
    if STATE["documents"]:
        STATE["vectors"] = STATE["vectorizer"].fit_transform(STATE["documents"])

def find_best_matches(query: str, top_n: int = NUMBER_OF_MATCHES) -> List[str]:
    if STATE["vectors"] is None or STATE["vectors"].shape[0] == 0:
        return []
    processed_query = [" ".join(ps.stem(w) for w in query.split())]
    qv = STATE["vectorizer"].transform(processed_query)
    sim = (qv * STATE["vectors"].T).toarray()[0]
    idxs = sim.argsort()[::-1][:top_n]
    return [STATE["original_docs"][i] for i in idxs]

def cohere_answer(query: str, context_chunks: List[str]) -> str:
    if not context_chunks:
        return "No documents are indexed yet. Upload a document first."

    prompt_context = "\n".join([f"Context {i+1}: {c}" for i, c in enumerate(context_chunks)])
    prompt = (
        "You are an AI assistant. Use the provided context to answer the user's query accurately and concisely.\n\n"
        f"Available Context:\n{prompt_context}\n\n"
        f"User Query: {query}\n\n"
        "If the context doesn't contain relevant information, say: "
        "\"I don't have enough information in the provided context to answer this question.\"\n\n"
        "Answer:"
    )

    resp = co.chat(
        model=COHERE_MODEL,
        message=prompt,
        max_tokens=350,
        temperature=0.3,
    )
    return (resp.text or "").strip()

def reset_db():
    STATE["original_docs"].clear()
    STATE["documents"].clear()
    STATE["vectorizer"] = TfidfVectorizer()
    STATE["vectors"] = None

# -----------------------------
# 6) API Models
# -----------------------------
class ChatIn(BaseModel):
    message: str

# -----------------------------
# 7) API Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    """Serve the main HTML page"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>RAG Chatbot</h1><p>index.html not found in static folder</p>"

@app.get("/health")
def health_check():
    """Health check endpoint for deployment platforms"""
    return {"status": "healthy", "model": COHERE_MODEL}

@app.get("/stats")
def stats():
    """Get current statistics"""
    return {"chunks": len(STATE["original_docs"]), "model": COHERE_MODEL}

@app.post("/reset")
def api_reset():
    """Reset the database"""
    reset_db()
    return {"ok": True}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload and process a document"""
    name = file.filename or "uploaded"
    ext = name.split(".")[-1].lower() if "." in name else ""
    data = await file.read()

    if not data:
        return {"ok": False, "error": "Empty file."}

    try:
        if ext == "pdf":
            original, processed = read_pdf_bytes(data)
        elif ext in ("html", "htm"):
            original, processed = read_html_bytes(data)
        elif ext in ("txt", "md"):
            original, processed = read_text_bytes(data)
        else:
            return {"ok": False, "error": f"Unsupported format: .{ext}"}

        if not original:
            return {"ok": False, "error": "No content extracted from file."}

        STATE["original_docs"].extend(original)
        add_document(processed)

        return {"ok": True, "filename": name, "chunks_added": len(original)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/chat")
def chat(payload: ChatIn):
    """Chat with the RAG system"""
    q = payload.message.strip()
    if not q:
        return {"ok": False, "error": "Empty message."}

    try:
        ctx = find_best_matches(q)
        ans = cohere_answer(q, ctx)
        return {"ok": True, "answer": ans, "context": ctx}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# -----------------------------
# 8) Run server (for local development)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
