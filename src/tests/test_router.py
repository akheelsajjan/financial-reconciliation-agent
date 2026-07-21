
import pytest
from src.retrieval.router import route_query

# Mark all router tests as "integration" (slow, needs API)
pytestmark = pytest.mark.integration

class TestRouteQuery:

    def test_net_sales_routes_to_income_statement(self):
        route = route_query("What was Apple total net sales in Q2 2026?")
        assert route["section"] == "Income Statement"
        assert route["chunk_type"] == "table"

    def test_net_income_routes_to_income_statement(self):
        route = route_query("What was Apple net income in Q2 2026?")
        assert route["section"] == "Income Statement"
        assert route["chunk_type"] == "table"

    def test_gross_margin_dollar_routes_to_income_statement(self):
        route = route_query("What was Apple gross margin in Q2 2026?")
        assert route["section"] == "Income Statement"
        assert route["chunk_type"] == "table"

    def test_total_assets_routes_to_balance_sheet(self):
        route = route_query("What was Apple total assets?")
        assert route["section"] == "Balance Sheet"
        assert route["chunk_type"] == "table"

    def test_cash_routes_to_balance_sheet(self):
        route = route_query("What was Apple cash and cash equivalents?")
        assert route["section"] == "Balance Sheet"
        assert route["chunk_type"] == "table"

    def test_dividends_paid_routes_to_cash_flow(self):
        route = route_query("How much did Apple pay in dividends?")
        assert route["section"] == "Cash Flow"
        assert route["chunk_type"] == "table"

    def test_operating_activities_routes_to_cash_flow(self):
        route = route_query("What was Apple cash from operating activities?")
        assert route["section"] == "Cash Flow"
        assert route["chunk_type"] == "table"

    def test_why_revenue_increased_routes_to_mda(self):
        route = route_query("Why did Apple iPhone revenue increase in Q2 2026?")
        assert route["section"] == "MD&A"
        assert route["chunk_type"] == "prose"

    def test_percentage_growth_routes_to_mda(self):
        route = route_query("By what percentage did Apple net sales grow?")
        assert route["section"] == "MD&A"
        assert route["chunk_type"] == "prose"

    def test_risk_factors_routes_correctly(self):
        route = route_query("What are Apple risk factors related to AI?")
        assert route["section"] == "Risk Factors"
        assert route["chunk_type"] == "prose"

    def test_eps_routes_to_notes(self):
        route = route_query("What was Apple diluted earnings per share?")
        assert route["section"] == "Notes"
        assert route["chunk_type"] == "table"

    def test_share_repurchase_routes_to_notes(self):
        route = route_query("What was Apple share repurchase activity?")
        assert route["section"] == "Notes"
        assert route["chunk_type"] == "prose"

    def test_route_always_returns_valid_section(self):
        """Router should never return an invalid section."""
        valid_sections = [
            "Income Statement", "Balance Sheet", "Cash Flow",
            "MD&A", "Risk Factors", "Legal Proceedings",
            "Notes", "Shareholders Equity", "Any"
        ]
        route = route_query("What was Apple revenue?")
        assert route["section"] in valid_sections

    def test_route_always_returns_valid_chunk_type(self):
        """Router should never return an invalid chunk type."""
        route = route_query("What was Apple revenue?")
        assert route["chunk_type"] in ["table", "prose"]