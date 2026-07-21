# src/retrieval/search.py

import json
import os
from pinecone import Pinecone
from src.retrieval.embeddings import (
    get_dense_embedding,
    scale_dense,
    scale_sparse,
    load_bm25
)
from src.retrieval.router import route_query
from dotenv import load_dotenv

load_dotenv()

NOISE_SECTIONS = ["General", "Signature", "Exhibits"]


def retrieve_with_parent_child(
    query: str,
    ticker: str,
    index,
    docstore: dict,
    bm25_encoder,
    section: str = None,
    chunk_type: str = None,
    top_k: int = 5,
    alpha: float = 0.5
) -> list:
    """
    Hybrid retrieval — BM25 + dense, parent-child pattern.
    Returns list of retrieved parents with metadata.
    """
    namespace = f"{ticker}_PC"

    dense_vector = get_dense_embedding(query)
    sparse_vector = bm25_encoder.encode_queries(query)

    filter_dict = {"section": {"$nin": NOISE_SECTIONS}}
    if section and section != "Any":
        filter_dict["section"] = {"$eq": section}
    if chunk_type:
        filter_dict["chunk_type"] = {"$eq": chunk_type}

    results = index.query(
        vector=scale_dense(dense_vector, alpha),
        sparse_vector=scale_sparse(sparse_vector, alpha),
        top_k=top_k * 3,
        namespace=namespace,
        include_metadata=True,
        filter=filter_dict
    )

    seen_parents = set()
    retrieved = []

    for match in results.matches:
        parent_id = match.metadata.get("parent_id")
        if not parent_id or parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)
        parent = docstore.get(parent_id)
        if parent:
            retrieved.append({
                "child_score": match.score,
                "child_content": match.metadata.get("content"),
                "parent_content": parent["content"],
                "metadata": {**match.metadata, "parent_id": parent_id}
            })
        if len(retrieved) >= top_k:
            break

    return retrieved


def retrieve_with_router(
    query: str,
    ticker: str,
    index,
    docstore: dict,
    bm25_encoder,
    top_k: int = 5,
    alpha: float = 0.5
) -> tuple:
    """
    Route query then retrieve with parent-child pattern.
    Returns: (retrieved_chunks, route)
    """
    route = route_query(query)

    retrieved = retrieve_with_parent_child(
        query=query,
        ticker=ticker,
        index=index,
        docstore=docstore,
        bm25_encoder=bm25_encoder,
        section=route["section"],
        chunk_type=route["chunk_type"],
        top_k=top_k,
        alpha=alpha
    )

    return retrieved, route