# Architectural Decisions Log

> Every significant design choice is recorded here with reasoning.
> Format: Decision → Why → Alternatives considered → Trade-offs accepted

---

## D-001 — Project domain: financial filings reconciliation

**Decision:** Build a reconciliation agent for financial filings rather than a general-purpose RAG assistant.

**Why:** Finance domain targets GCC roles (Goldman Sachs, JPMorgan). Reconciliation (stated vs calculated) creates a genuinely novel use case — most RAG demos only retrieve, this one audits. The HITL trigger is mathematically grounded (>5% discrepancy), not arbitrary.

**Alternatives considered:** General document QA, legal document analysis, HR policy assistant.

**Trade-offs accepted:** Finance documents are dense and require careful chunking. Tables + prose mixed layout is harder than plain text.

---

## D-002 — Companies: AAPL + MSFT + Reliance Industries

**Decision:** Use Apple (AAPL) and Microsoft (MSFT) SEC 10-Q filings, and Reliance Industries BSE Annual Report.

**Why:** AAPL and MSFT are widely understood, publicly available via SEC EDGAR, and have clean structured 10-Q filings. Reliance introduces cross-market, cross-currency reconciliation (INR vs USD, Ind AS vs US GAAP) — a genuine enterprise-grade challenge that differentiates this project.

**Alternatives considered:** All US companies only for simplicity.

**Trade-offs accepted:** Reliance requires a separate document schema and a currency normalisation step before reconciliation. Added complexity is worth the differentiation.

---

## D-003 — Ground truth definition for reconciliation

**Decision:** The independently *calculated* number is ground truth. The *stated* number in the filing is what is being audited.

**Why:** The purpose of this system is to detect hallucinations and errors in stated figures. If Apple's 10-Q states "revenue grew 8%" but the raw numbers in the same document compute to 6.2%, the calculated value (6.2%) is what we trust — it is derived directly from primary financial data.

**Alternatives considered:** Treating stated figures as ground truth and flagging calculation errors. Rejected — this defeats the purpose of reconciliation.

**Trade-offs accepted:** Requires reliable number extraction from PDFs. Errors in extraction propagate to false positives. Mitigated by component-level evaluation in Layer 3.

---

## D-004 — HITL trigger threshold: 5% discrepancy

**Decision:** Human-in-the-loop escalation fires when |stated_value - calculated_value| / calculated_value > 5%.

**Why:** 5% is a common materiality threshold in financial auditing. Below 5%, small rounding or reporting period differences explain the gap. Above 5%, the discrepancy is material and warrants review.

**Alternatives considered:** 1% (too sensitive, too many false positives), 10% (too loose, misses real errors), dynamic threshold per metric (too complex for v1).

**Trade-offs accepted:** Some genuine errors below 5% will not trigger HITL. Threshold is configurable via environment variable for easy tuning.

---

## D-005 — Retrieval strategy: Hybrid BM25 + Dense + RRF

**Decision:** Use hybrid retrieval combining BM25 (keyword) and dense (semantic) search, merged via Reciprocal Rank Fusion (RRF).

**Why:** Financial documents contain exact numerical terms ("$89.5 billion", "Q3 FY2024") that semantic search alone misses — exact keyword match is essential. Dense search handles semantic variations ("revenue" vs "net sales" vs "total income"). RRF merges ranked lists without requiring score normalisation.

**Alternatives considered:** Dense-only (misses exact numbers), BM25-only (misses semantic variations), cross-encoder reranking only (too slow for retrieval stage).

**Trade-offs accepted:** More complex ingestion pipeline. Requires maintaining both sparse and dense indices in Pinecone.

---

## D-006 — Chunking strategy: 800 tokens, 100 token overlap, section-aware

**Decision:** Chunk at 800 tokens with 100 token overlap. Respect section boundaries (do not split across "Revenue", "Operating Income" section headers). Tables are extracted separately as structured chunks.

**Why:** 800 tokens balances context richness (financial sentences are long) with retrieval precision (too large = irrelevant context retrieved). 100 token overlap preserves cross-sentence numerical context. Section-aware splitting ensures a revenue figure stays with its label. Tables extracted separately because mixed table+prose chunks confuse both BM25 and dense retrieval.

**Alternatives considered:** 512 tokens (too small for financial prose), 1024 tokens (retrieves too much irrelevant text), character-based splitting (ignores token budget of LLM context windows).

**Trade-offs accepted:** Table extraction adds complexity in Layer 1. Worth it — financial numbers live predominantly in tables.

---

## D-007 — Development approach: Jupyter first, Python module after each layer

**Decision:** Every layer is developed and validated in a Jupyter notebook. Once stable, code moves to `src/` as a Python module before the next layer begins.

**Why:** Notebooks allow fast iteration, inline output inspection, and easy debugging of intermediate steps (especially important for PDF parsing and retrieval quality checks). Moving to modules enforces clean API boundaries and makes the code interview-ready.

**Alternatives considered:** Python modules from day one (too slow for exploration), notebooks only (not production-ready, not demonstrable as a system).

**Trade-offs accepted:** Double work — code is written twice. Accepted because the notebook becomes documentation and the module becomes the product.

---

## D-008 — Package manager: uv

**Decision:** Use `uv` for all Python dependency management.

**Why:** uv is significantly faster than pip, supports lockfiles natively, and is becoming the standard in modern Python GenAI projects. Signals current tooling awareness to interviewers.

**Alternatives considered:** pip + venv (slower, no lockfile), conda (heavy, not standard in GenAI), poetry (good but slower than uv).

**Trade-offs accepted:** uv is newer — some edge cases less documented. Not a concern for this project's dependency set.

---

## D-009 — First metrics in scope for v1

**Decision:** Begin with YoY revenue growth only. Add operating margin and EPS in Layer 4 once the reconciliation loop is validated.

**Why:** YoY revenue growth is universally understood, always present in 10-Q filings, and easy to verify manually during development. Starting with one metric keeps Layer 1 and Layer 2 focused on infrastructure, not domain complexity.

**Alternatives considered:** All three metrics from day one (too much to debug simultaneously).

**Trade-offs accepted:** v1 demo covers only revenue. Acceptable — the architecture generalises to any metric once the loop works.

---

## D-010 — Multi-document, multi-currency handling

**Decision:** AAPL and MSFT filings are processed in USD/US GAAP. Reliance is processed in INR/Ind AS. A currency normalisation step (INR → USD at filing-date exchange rate) is applied before cross-company comparisons. Within-company reconciliation always uses native currency.

**Why:** Cross-company comparisons in different currencies require normalisation. Within-company reconciliation does not — comparing stated INR figure vs calculated INR figure is valid without conversion.

**Alternatives considered:** USD-only (excludes Reliance), no normalisation (produces misleading cross-company comparisons).

**Trade-offs accepted:** Exchange rate at filing date introduces a data dependency. Handled via MCP tool in Layer 4 (forex rate fetcher).

---

*New decisions are added here as each layer is built. Format: D-NNN — short title.*


BASELINE METRICS — Layer 1 hybrid retrieval, no Query Router
Hit Rate@5: 0.4667
MRR: 0.3222
Avg Precision@5: 0.2267
Key finding: Prose retrieval works well (Hit Rate 100%). Financial table retrieval fails (Hit Rate 0% for Income Statement queries). Root cause: dense embeddings favour semantic prose over number-heavy tables. Fix: Query Router with section + chunk_type filters.

After adding Query Router
Baseline → With Router → Delta
Hit Rate@5     :    0.4667  →   0.8000    → +0.33
MRR            :    0.3222  →   0.8000    → +0.48
Avg Precision@5:    0.2267  →   0.4533    → +0.23
retrieval Hit Rate@5 from 46.7% to 80%
and MRR from 0.32 to 0.80 on a 15-query golden evaluation set.

LAYER 1.5 METRICS — Parent-Child chunking + improved Query Router
Hit Rate@5: 0.7143 (+8.7% over Layer 1)
MRR: 0.7143 (+8.7% over Layer 1)
Avg Precision@5: 0.2552 (-33.4% — known trade-off of parent deduplication)
Key finding: Parent-Child improved retrieval recall. Precision drop is expected — deduplication reduces unique results in top-5. Fix planned: increase children top_k in Layer 3.