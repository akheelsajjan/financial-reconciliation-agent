# Financial Reconciliation Agent

> A multi-agent system that reads financial filings, independently computes key metrics, and flags any discrepancy between what the document *says* and what the numbers *prove* — pausing for human review when divergence exceeds a threshold.

---

## What we are building

A system that reads financial documents and checks its own math.
You give it a question like "Did Apple's revenue grow in Q3?"
It does three things most RAG apps don't:

Finds the stated answer in the filing — Apple says "revenue grew 8%"
Calculates the answer independently from the raw numbers in the same document — Q3 this year ÷ Q3 last year = 6.2% growth
Compares the two — if they differ by more than 5%, it stops and asks a human to review before giving a final answer

That's it. Retrieve → Verify → Flag.

'''
A self-auditing multi-agent RAG system that cross-references stated financial figures against independently calculated values — and escalates to a human reviewer when they don't match.
'''

---

## What makes this different

Most RAG applications retrieve information and answer questions. This system goes further — it **audits its own answers** by cross-referencing stated figures in financial filings against independently calculated values using live market data and document-extracted numbers.

If Apple's 10-Q states "revenue grew 8%" but the raw figures in the same document compute to 6.2%, the system flags it, explains the discrepancy, and escalates to a human reviewer before delivering a final answer.

---

## Architecture overview

```
User Query
    │
    ▼
Supervisor Agent (LangGraph)
    ├── Retrieval Agent        → finds stated figures in filings (Pinecone hybrid search)
    ├── Calculator Agent       → independently computes metrics (custom MCP server)
    └── Reconciliation Agent  → compares stated vs calculated, triggers HITL if Δ > 5%
            │
            ▼ (if discrepancy > 5%)
    HITL Escalation → pause → human review → resume or override
            │
            ▼
    Final Answer (with confidence score + audit trail)
```

---

## Companies & filings

| Company | Exchange | Filing type | Currency | Accounting standard |
|---|---|---|---|---|
| Apple (AAPL) | NASDAQ / SEC | 10-Q | USD | US GAAP |
| Microsoft (MSFT) | NASDAQ / SEC | 10-Q | USD | US GAAP |
| Reliance Industries | BSE / NSE | Annual Report | INR | Ind AS |

> Reliance introduces cross-market, cross-currency reconciliation — a real enterprise-grade challenge handled via a currency normalisation step before reconciliation.

---

## Metrics tracked (v1)

- YoY revenue growth (stated vs calculated)
- Operating margin (stated vs calculated)
- EPS (stated vs calculated)

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o / Claude Sonnet |
| Orchestration | LangGraph |
| Vector DB | Pinecone (serverless) |
| Retrieval | Hybrid BM25 + Dense + RRF |
| Reranking | Cohere rerank |
| Observability | Langfuse |
| Evaluation | RAGAS + DeepEval |
| Tool protocol | MCP (custom server) |
| Caching | Redis |
| Cost routing | LiteLLM |
| Fine-tuning | LoRA via Unsloth |
| Containerisation | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Cloud | AWS (ECS, Lambda, S3, Bedrock) |
| Package manager | uv |

---

## Build layers

| Layer | What gets built | Key addition |
|---|---|---|
| 1 | Core RAG pipeline — ingestion, hybrid retrieval, FastAPI | Foundation |
| 2 | Langfuse observability, prompt versioning, streaming | LLMOps starts |
| 3 | Component-level RAGAS eval, CI gate, Dockerfile | Eval + Docker |
| 4 | Multi-agent orchestration, MCP server, HITL, guardrails | Flagship |
| 5 | Cost routing, semantic cache, LiteLLM | Cost aware |
| 6 | LoRA fine-tuning on finance QA pairs | Fine-tuning |
| 7 | AWS deployment — ECS, Lambda, S3, Bedrock | Cloud |

> Each layer is a working, demonstrable system. Nothing above breaks what is below.

---

## Development approach

- **Jupyter first** — every layer is developed and tested in a notebook under `notebooks/`
- **Python structure after** — once a layer is stable, it moves to `src/` as a proper module
- **README-driven** — README is updated before code is written for each layer
- **Decisions logged** — every architectural choice is recorded in `DECISIONS.md`

---

## Repository structure

```
financial-reconciliation-agent/
├── README.md
├── DECISIONS.md
├── ERRORS.md
├── .env.example
├── notebooks/
│   ├── layer1_rag_pipeline.ipynb
│   ├── layer2_observability.ipynb
│   ├── layer3_eval_ci.ipynb
│   ├── layer4_multiagent.ipynb
│   ├── layer5_cost_routing.ipynb
│   ├── layer6_finetuning.ipynb
│   └── layer7_deployment.ipynb
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   ├── agents/
│   ├── mcp_server/
│   ├── evaluation/
│   └── api/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   ├── raw/          ← downloaded filings (gitignored)
│   └── processed/    ← chunked + embedded (gitignored)
└── .github/
    └── workflows/
        └── eval_ci.yml
```

---

## Setup (Layer 1)

```bash
# Clone
git clone https://github.com/akheelsajjan/financial-reconciliation-agent
cd financial-reconciliation-agent

# Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Fill in your keys in .env
```

---

## Author

**Akheel Sajjan** — Senior Software Engineer transitioning to Generative AI / Agentic AI
- GitHub: [@akheelsajjan](https://github.com/akheelsajjan)
- LinkedIn: [AkheelSajjan](https://linkedin.com/in/AkheelSajjan)