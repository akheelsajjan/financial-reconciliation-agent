# tests/test_section_detector.py

import pytest
from src.ingestion.section_detector import detect_section_regex


class TestDetectSectionRegex:

    def test_income_statement_variations(self):
        assert detect_section_regex("INCOME STATEMENTS") == "Income Statement"
        assert detect_section_regex("STATEMENTS OF OPERATIONS") == "Income Statement"
        assert detect_section_regex("CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS") == "Income Statement"
        assert detect_section_regex("STATEMENTS OF INCOME") == "Income Statement"

    def test_balance_sheet_variations(self):
        assert detect_section_regex("BALANCE SHEETS") == "Balance Sheet"
        assert detect_section_regex("CONDENSED CONSOLIDATED BALANCE SHEETS") == "Balance Sheet"
        assert detect_section_regex("FINANCIAL POSITION") == "Balance Sheet"

    def test_cash_flow_variations(self):
        assert detect_section_regex("CASH FLOWS STATEMENTS") == "Cash Flow"
        assert detect_section_regex("STATEMENTS OF CASH FLOWS") == "Cash Flow"
        assert detect_section_regex("CASH FLOW STATEMENT") == "Cash Flow"

    def test_mda_variations(self):
        assert detect_section_regex("MANAGEMENT'S DISCUSSION") == "MD&A"
        assert detect_section_regex("RESULTS OF OPERATIONS") == "MD&A"
        assert detect_section_regex("ITEM 2. MANAGEMENT") == "MD&A"

    def test_risk_factors(self):
        assert detect_section_regex("RISK FACTORS") == "Risk Factors"
        assert detect_section_regex("ITEM 1A. RISK FACTORS") == "Risk Factors"
        assert detect_section_regex("ITEM 1A") == "Risk Factors"

    def test_legal_proceedings(self):
        assert detect_section_regex("LEGAL PROCEEDINGS") == "Legal Proceedings"
        assert detect_section_regex("ITEM 1. LEGAL PROCEEDINGS") == "Legal Proceedings"

    def test_notes(self):
        assert detect_section_regex("NOTES TO CONDENSED CONSOLIDATED") == "Notes"
        assert detect_section_regex("NOTES TO FINANCIAL STATEMENTS") == "Notes"

    def test_shareholders_equity(self):
        assert detect_section_regex("STOCKHOLDERS EQUITY") == "Shareholders Equity"
        assert detect_section_regex("SHAREHOLDERS EQUITY") == "Shareholders Equity"

    def test_unknown_returns_general(self):
        assert detect_section_regex("PART I Item 1") == "General"
        assert detect_section_regex("Effective Tax Rate") == "General"
        assert detect_section_regex("Note 2 - Revenue") == "General"
        assert detect_section_regex("Three Months Ended December 31") == "General"

    def test_case_insensitive(self):
        assert detect_section_regex("income statements") == "Income Statement"
        assert detect_section_regex("Balance Sheet") == "Balance Sheet"
        assert detect_section_regex("cash flows") == "Cash Flow"