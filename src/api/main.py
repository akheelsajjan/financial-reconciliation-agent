import os
import json
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Literal
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langfuse import get_client

 # Get project root regardless of where uvicorn is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

# Globals
openai_client = None
index = None
docstore = None
bm25_encoder = None
langfuse = None

EMBEDDING_MODEL = "text-embedding-3-small"
NOISE_SECTIONS = ["General", "Signature", "Exhibits"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global openai_client, index, docstore, bm25_encoder, langfuse

    print("Loading resources...")
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("financial-reconciliation")



    with open(os.path.join(DATA_DIR, "docstore.json"), "r") as f:
        docstore = json.load(f)

    bm25_encoder = BM25Encoder()
    bm25_encoder.load(os.path.join(DATA_DIR, "bm25_encoder.json"))

    langfuse = get_client()

    print(f"Resources loaded | Docstore: {len(docstore)} parents")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Financial Reconciliation Agent",
    description="RAG system for SEC 10-Q financial filings",
    version="1.0.0",
    lifespan=lifespan
)


# Request/Response schemas
class QueryRequest(BaseModel):
    query: str
    ticker: str
    top_k: Optional[int] = 3
    alpha: Optional[float] = 0.5


class Source(BaseModel):
    section: str
    page: float
    chunk_type: str
    child_score: float


class QueryResponse(BaseModel):
    query: str
    ticker: str
    answer: str
    route: dict
    sources: list[Source]
    usage: dict
    latency_ms: int
    prompt_version: int


# Helper functions
def get_embeddings(texts: list) -> list:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [e.embedding for e in response.data]


def scale_dense(vector: list, alpha: float) -> list:
    return [v * alpha for v in vector]


def scale_sparse(vector: dict, alpha: float) -> dict:
    return {
        "indices": vector["indices"],
        "values": [v * (1 - alpha) for v in vector["values"]]
    }


class QRoute(BaseModel):
    chunk_type: Literal["table", "prose"]
    section: Literal[
        "Income Statement", "Balance Sheet", "Cash Flow",
        "MD&A", "Risk Factors", "Legal Proceedings",
        "Notes", "Shareholders Equity", "Any"
    ]


def route_query(query: str) -> dict:
    system_prompt = """You are a financial document query router for SEC 10-Q filings.

PRIORITY RULES:
- inventory/inventories → Balance Sheet, table
- dividends PAID → Cash Flow, table
- R&D as percentage → MD&A, prose
- EPS computation → Notes, table
- gross margin as dollar figure → Income Statement, table
- R&D dollar amount → Income Statement, table
- change/grew/increased about a metric → MD&A, prose (unless raw dollar with 'what was')

SECTION GUIDE:
- Income Statement: revenue, net sales, gross margin, operating income, net income, cost of sales
- Balance Sheet: total assets, cash, liabilities, equity, receivables, inventory
- Cash Flow: operating activities, investing, financing, dividends paid, capex
- MD&A: why metrics changed, percentage change, growth reasons, segment performance
- Risk Factors: risks, uncertainties, challenges
- Legal Proceedings: lawsuits, investigations
- Notes: RSUs, debt details, EPS computation, weighted average shares
- Shareholders Equity: retained earnings, AOCI
- Any: spans multiple sections

CHUNK TYPE: NUMBER/FIGURE → table | EXPLANATION/WHY/HOW → prose"""

    response = openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        response_format=QRoute,
        temperature=0
    )
    route = response.choices[0].message.parsed
    return {"chunk_type": route.chunk_type, "section": route.section}


def retrieve(query, ticker, section, chunk_type, top_k, alpha):
    namespace = f"{ticker}_PC"
    dense_vector = get_embeddings([query])[0]
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


# Endpoints
@app.get("/health")
def health():
    return {
        "status": "ok",
        "docstore_parents": len(docstore),
        "index": "financial-reconciliation"
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    start_time = time.time()

    with langfuse.start_as_current_observation(
        as_type="span",
        name="answer_question",
        input={"query": request.query, "ticker": request.ticker}
    ) as root_span:

        with langfuse.start_as_current_observation(
            as_type="span",
            name="route_query",
            input={"query": request.query}
        ) as route_span:
            route = route_query(request.query)
            route_span.update(output=route)

        with langfuse.start_as_current_observation(
            as_type="span",
            name="retrieve_chunks",
            input={"ticker": request.ticker, **route}
        ) as retrieve_span:
            retrieved = retrieve(
                request.query, request.ticker,
                route["section"], route["chunk_type"],
                request.top_k, request.alpha
            )
            retrieve_span.update(output={"chunks_retrieved": len(retrieved)})

        context = "\n\n---\n\n".join([
            f"[{r['metadata'].get('section')} | Page {r['metadata'].get('page')}]\n{r['parent_content']}"
            for r in retrieved
        ])

        try:
            prompt_obj = langfuse.get_prompt(
                "financial_analyst_system",
                label="production"
            )
            system_prompt = prompt_obj.prompt
            prompt_version = prompt_obj.version
        except Exception:
            system_prompt = """You are a financial analyst assistant.
Answer based ONLY on context provided.
Use Three Months Ended column for quarterly questions.
Append million to dollar figures. Never fabricate numbers."""
            prompt_version = 0

        user_prompt = f"Context:\n{context}\n\nQuestion: {request.query}\n\nAnswer:"

        with langfuse.start_as_current_observation(
            as_type="generation",
            name="llm_generation",
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        ) as gen_span:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )
            answer = response.choices[0].message.content
            gen_span.update(
                output=answer,
                usage_details={
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens
                }
            )

        root_span.update(output={"answer": answer})

    langfuse.flush()
    total_time = time.time() - start_time

    return QueryResponse(
        query=request.query,
        ticker=request.ticker,
        answer=answer,
        route=route,
        sources=[
            Source(
                section=r["metadata"].get("section", ""),
                page=r["metadata"].get("page", 0),
                chunk_type=r["metadata"].get("chunk_type", ""),
                child_score=r["child_score"]
            )
            for r in retrieved
        ],
        usage={
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        },
        latency_ms=round(total_time * 1000),
        prompt_version=prompt_version
    )