
import os
from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import BM25Encoder

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# Load once at module level — shared across all callers
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_dense_embedding(text: str) -> list:
    """Single text → dense vector."""
    return embedding_model.encode(
        text,
        normalize_embeddings=True
    ).tolist()


def get_dense_embeddings(texts: list) -> list:
    """Batch texts → list of dense vectors."""
    return embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()


def scale_dense(vector: list, alpha: float) -> list:
    return [v * alpha for v in vector]


def scale_sparse(vector: dict, alpha: float) -> dict:
    return {
        "indices": vector["indices"],
        "values": [v * (1 - alpha) for v in vector["values"]]
    }


def load_bm25(path: str) -> BM25Encoder:
    encoder = BM25Encoder()
    encoder.load(path)
    return encoder


def fit_and_save_bm25(texts: list, path: str) -> BM25Encoder:
    encoder = BM25Encoder()
    encoder.fit(texts)
    encoder.dump(path)
    print(f"✓ BM25 fitted on {len(texts)} texts and saved")
    return encoder