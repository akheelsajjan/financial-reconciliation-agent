
import json
import os
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import SentenceTransformer
from groq import Groq
from langfuse import get_client
from dotenv import load_dotenv



# These are loaded once at startup and shared across all requests
_index = None
_docstore = None
_bm25_encoder = None
_embedding_model = None
_groq_client = None
_langfuse = None




BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
load_dotenv(os.path.join(BASE_DIR, ".env"))

print(f"BASE_DIR: {BASE_DIR}")
print(f"ENV file exists: {os.path.exists(os.path.join(BASE_DIR, '.env'))}")
print(f"PINECONE_API_KEY loaded: {bool(os.getenv('PINECONE_API_KEY'))}")


def load_all_resources():
    """Load all resources at startup."""
    global _index, _docstore, _bm25_encoder, _embedding_model
    global _groq_client, _langfuse

    print("Loading resources...")

    # Groq
    _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    _index = pc.Index("financial-reconciliation-v4")

    # Docstore
    with open(os.path.join(DATA_DIR, "docstore.json"), "r") as f:
        _docstore = json.load(f)

    # BM25
    _bm25_encoder = BM25Encoder()
    _bm25_encoder.load(os.path.join(DATA_DIR, "bm25_encoder.json"))

    # SentenceTransformer
    _embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # Langfuse
    _langfuse = get_client()

    print(f"✓ All resources loaded | Docstore: {len(_docstore)} parents")


def get_index():
    return _index


def get_docstore():
    return _docstore


def get_bm25():
    return _bm25_encoder


def get_embedding_model():
    return _embedding_model


def get_groq():
    return _groq_client


def get_langfuse():
    return _langfuse