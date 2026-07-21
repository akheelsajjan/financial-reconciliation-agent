# src/ingestion/section_detector.py

import json
import os
from pydantic import BaseModel
from typing import Literal
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SECTION_PATTERNS = {
    "Income Statement": [
        "statements of operations", "statements of income",
        "income statements", "income statement", "comprehensive income"
    ],
    "Balance Sheet": [
        "balance sheet", "financial position", "balance sheets"
    ],
    "Cash Flow": [
        "cash flows", "cash flow statement", "cash flows statements"
    ],
    "Shareholders Equity": [
        "shareholders equity", "stockholders equity",
        "stockholders' equity statements", "equity statements"
    ],
    "MD&A": [
        "management's discussion", "results of operations",
        "summary results of operations", "item 2. management",
        "item 2.    management"
    ],
    "Risk Factors": [
        "risk factors", "item 1a", "item 1a.",
        "strategic and competitive risks"
    ],
    "Legal Proceedings": [
        "legal proceedings", "item 1. legal", "legal proceeding"
    ],
    "Notes": [
        "notes to condensed", "notes to consolidated",
        "notes to financial statements"
    ],
    "Signature": ["pursuant to the requirements", "duly authorized"],
    "Exhibits": ["exhibit 31", "exhibit 32", "incorporated by reference"]
}

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "processed", "section_cache.json"
)


class SectionClassification(BaseModel):
    section: Literal[
        "Income Statement", "Balance Sheet", "Cash Flow",
        "Shareholders Equity", "MD&A", "Risk Factors",
        "Legal Proceedings", "Notes", "Signature", "Exhibits", "General"
    ]
    confidence: Literal["high", "low"]


def load_section_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_section_cache(cache: dict):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


section_cache = load_section_cache()


def detect_section_regex(text: str) -> str:
    text_lower = text.lower()
    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if pattern in text_lower:
                return section
    return "General"


def detect_section_llm(header_text: str) -> str:
    if header_text in section_cache:
        return section_cache[header_text]

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a financial document section classifier for SEC 10-Q filings.
Classify ONLY top-level major section headers. Sub-headers → General.

High confidence examples:
  "INCOME STATEMENTS" → Income Statement
  "BALANCE SHEETS" → Balance Sheet
  "STOCKHOLDERS' EQUITY STATEMENTS" → Shareholders Equity
  "NOTES TO FINANCIAL STATEMENTS" → Notes
  "CASH FLOWS STATEMENTS" → Cash Flow
  "MANAGEMENT'S DISCUSSION AND ANALYSIS" → MD&A
  "ITEM 1A. RISK FACTORS" → Risk Factors
  "ITEM 1. LEGAL PROCEEDINGS" → Legal Proceedings

Low confidence → General:
  Individual notes, sub-topics, navigation text, date headers

Return ONLY valid JSON: {"section": "...", "confidence": "high|low"}"""
            },
            {"role": "user", "content": f"Classify: '{header_text}'"}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    try:
        result = json.loads(response.choices[0].message.content)
        section = result.get("section", "General")
        confidence = result.get("confidence", "low")
        final = section if confidence == "high" else "General"
        section_cache[header_text] = final
        return final
    except:
        section_cache[header_text] = "General"
        return "General"


def detect_section(header_text: str) -> str:
    """Hybrid: regex fast path → Groq LLM fallback."""
    regex_result = detect_section_regex(header_text)
    if regex_result != "General":
        return regex_result
    return detect_section_llm(header_text)