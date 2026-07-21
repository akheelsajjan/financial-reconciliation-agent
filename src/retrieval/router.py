
import json
import os
from pydantic import BaseModel
from typing import Literal
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

ROUTER_SYSTEM_PROMPT = """You are a financial document query router for SEC 10-Q filings.

PRIORITY RULES:
- inventory/inventories → Balance Sheet, table
- dividends PAID as cash outflow → Cash Flow, table
- R&D as percentage or ratio → MD&A, prose
- EPS computation or weighted average shares → Notes, table
- gross margin as dollar figure → Income Statement, table
- R&D dollar amount → Income Statement, table
- change/grew/increased about a metric → MD&A, prose
- share repurchase program details → Notes, prose

SECTION GUIDE:
- Income Statement: revenue, net sales, gross margin, operating income, net income, cost of sales
- Balance Sheet: total assets, cash, liabilities, equity, receivables, inventory
- Cash Flow: operating activities, investing, financing, dividends paid, capex
- MD&A: why metrics changed, percentage change, growth reasons, segment performance
- Risk Factors: risks, uncertainties, challenges
- Legal Proceedings: lawsuits, investigations
- Notes: RSUs, debt details, EPS computation, weighted average shares, share repurchase
- Shareholders Equity: retained earnings, AOCI
- Any: spans multiple sections

CHUNK TYPE: NUMBER/FIGURE → table | EXPLANATION/WHY/HOW → prose

Return ONLY valid JSON: {"chunk_type": "table", "section": "Income Statement"}"""


class QueryRoute(BaseModel):
    chunk_type: Literal["table", "prose"]
    section: Literal[
        "Income Statement", "Balance Sheet", "Cash Flow",
        "MD&A", "Risk Factors", "Legal Proceedings",
        "Notes", "Shareholders Equity", "Any"
    ]


def route_query(query: str, model: str = "llama-3.1-8b-instant") -> dict:
    """
    Route a query to the correct section and chunk type.
    Returns: {"chunk_type": "table", "section": "Income Statement"}
    """
    response = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    try:
        result = json.loads(response.choices[0].message.content)
        route = QueryRoute(**result)
        return {"chunk_type": route.chunk_type, "section": route.section}
    except Exception:
        return {"chunk_type": "prose", "section": "Any"}