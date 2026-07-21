# src/api/main.py

import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv

from src.api.schemas import QueryRequest, QueryResponse, Source, HealthResponse
from src.api.dependencies import (
    load_all_resources,
    get_index,
    get_docstore,
    get_bm25,
    get_groq,
    get_langfuse
)
from src.retrieval.search import retrieve_with_router

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"))

SYSTEM_PROMPT = """You are a financial analyst assistant specializing in SEC filings.

Answer questions based ONLY on the provided financial document context.
Rules:
- State numbers exactly as they appear in the document
- Financial figures in 10-Q tables are in millions unless stated otherwise
- Always mention the exact period: "Three Months Ended March 28, 2026"
- 10-Q tables show BOTH quarterly (Three Months Ended) AND year-to-date (Six Months Ended)
- When asked about a specific quarter → use Three Months Ended column ONLY
- Always append "million" to dollar figures from financial statements
- If you cannot find the answer say: "Not found in the provided filing sections"
- Never fabricate numbers"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_resources()
    yield
    print("Shutting down...")


app = FastAPI(
    title="Financial Reconciliation Agent",
    description="RAG system for SEC 10-Q financial filings",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        docstore_parents=len(get_docstore()),
        index="financial-reconciliation-v4",
        embedding_model="BAAI/bge-small-en-v1.5",
        llm="llama-3.1-8b-instant"
    )


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    start_time = time.time()
    langfuse = get_langfuse()
    groq_client = get_groq()

    with langfuse.start_as_current_observation(
        as_type="span",
        name="answer_question",
        input={"query": request.query, "ticker": request.ticker}
    ) as root_span:

        # Route + Retrieve
        with langfuse.start_as_current_observation(
            as_type="span",
            name="retrieve_chunks",
            input={"query": request.query, "ticker": request.ticker}
        ) as retrieve_span:
            retrieved, route = retrieve_with_router(
                query=request.query,
                ticker=request.ticker,
                index=get_index(),
                docstore=get_docstore(),
                bm25_encoder=get_bm25(),
                top_k=request.top_k,
                alpha=request.alpha
            )
            retrieve_span.update(output={
                "chunks_retrieved": len(retrieved),
                "route": route
            })

        # Build context
        context = "\n\n---\n\n".join([
            f"[{r['metadata'].get('section')} | Page {r['metadata'].get('page')}]\n{r['parent_content']}"
            for r in retrieved
        ])

        # Fetch prompt from Langfuse
        try:
            prompt_obj = langfuse.get_prompt(
                "financial_analyst_system", label="production"
            )
            system_prompt = prompt_obj.prompt
            prompt_version = prompt_obj.version
        except Exception:
            system_prompt = SYSTEM_PROMPT
            prompt_version = 0

        user_prompt = f"Financial filing context:\n{context}\n\nQuestion: {request.query}\n\nAnswer:"

        # Generate
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="llm_generation",
            model="llama-3.1-8b-instant",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        ) as gen_span:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
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
                page=float(r["metadata"].get("page", 0) or 0),
                chunk_type=r["metadata"].get("chunk_type", ""),
                child_score=float(r["child_score"])
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