import os
import io
import ssl
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from pypdf import PdfReader
from bs4 import BeautifulSoup as bs
import cohere

# Load environment variables
load_dotenv()

# Get API credentials
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "").strip()
COHERE_MODEL = os.getenv("COHERE_MODEL", "command-r").strip()

if not COHERE_API_KEY:
    raise RuntimeError(
        "COHERE_API_KEY not set. "
        "Set it in Render environment variables."
    )

# NLTK setup with SSL workaround
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

# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot Pro", version="1.0.0")

# CORS middleware - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for index.html)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Create static directory if it doesn't exist
os.makedirs(STATIC_DIR, exist_ok=True)

try:
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
except Exception as e:
    print(f"Warning: Could not mount static files: {e}")

# In-memory document store
STATE = {
    "original_docs": [],
    "documents": [],
    "vectorizer": TfidfVectorizer(),
    "vectors": None,
}

# Initialize Co
