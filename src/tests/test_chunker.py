
import pytest
from src.ingestion.chunker import (
    split_table_into_rows,
    is_meaningful_row,
    split_into_sentences,
    clean_prose,
    get_page_no,
)


# ── is_meaningful_row ─────────────────────────────────────────────

class TestIsMeaningfulRow:

    def test_dollar_sign_row_is_meaningful(self):
        row = "| Net sales | $ 111,184 | $ 95,359 |"
        assert is_meaningful_row(row) is True

    def test_comma_number_row_is_meaningful(self):
        row = "| Total assets | 111,184 | 95,359 |"
        assert is_meaningful_row(row) is True

    def test_date_row_is_not_meaningful(self):
        row = "| March 28, 2026 | March 29, 2025 |"
        assert is_meaningful_row(row) is False

    def test_december_date_is_not_meaningful(self):
        row = "| December 31, 2025 | December 31, 2024 |"
        assert is_meaningful_row(row) is False

    def test_empty_row_is_not_meaningful(self):
        row = "| | | |"
        assert is_meaningful_row(row) is False

    def test_single_cell_row_is_not_meaningful(self):
        row = "| Net sales: |"
        assert is_meaningful_row(row) is False

    def test_row_without_numbers_is_not_meaningful(self):
        row = "| Products | Services |"
        assert is_meaningful_row(row) is False


# ── split_table_into_rows ─────────────────────────────────────────

class TestSplitTableIntoRows:

    def test_basic_table_returns_one_row(self):
        table = """| Header A | Header B |
|----------|----------|
| Net sales | $ 111,184 |"""
        rows = split_table_into_rows(table)
        assert len(rows) == 1

    def test_row_contains_header(self):
        table = """| Header A | Header B |
|----------|----------|
| Net sales | $ 111,184 |"""
        rows = split_table_into_rows(table)
        assert "Header A" in rows[0]

    def test_row_contains_data(self):
        table = """| Header A | Header B |
|----------|----------|
| Net sales | $ 111,184 |"""
        rows = split_table_into_rows(table)
        assert "111,184" in rows[0]

    def test_date_rows_filtered_out(self):
        table = """| Header A | Header B |
|----------|----------|
| March 28, 2026 | March 29, 2025 |
| Net sales | $ 111,184 |"""
        rows = split_table_into_rows(table)
        assert len(rows) == 1
        assert "111,184" in rows[0]

    def test_multiple_data_rows(self):
        table = """| Header |
|--------|
| Net sales | $ 111,184 |
| Net income | $ 29,578 |
| Gross margin | $ 54,781 |"""
        rows = split_table_into_rows(table)
        assert len(rows) == 3

    def test_empty_table_returns_empty_list(self):
        table = """| Header |
|--------|"""
        rows = split_table_into_rows(table)
        assert rows == []


# ── split_into_sentences ──────────────────────────────────────────

class TestSplitIntoSentences:

    def test_single_sentence(self):
        text = "Apple revenue increased in Q2 2026."
        sentences = split_into_sentences(text)
        assert len(sentences) == 1

    def test_multiple_sentences(self):
        text = "Revenue increased. This was due to iPhone growth. Services also grew."
        sentences = split_into_sentences(text)
        assert len(sentences) == 3

    def test_empty_string(self):
        sentences = split_into_sentences("")
        assert sentences == []

    def test_sentences_stripped(self):
        text = "  Apple grew.  Microsoft also grew.  "
        sentences = split_into_sentences(text)
        for s in sentences:
            assert s == s.strip()


# ── clean_prose ───────────────────────────────────────────────────

class TestCleanProse:

    def test_removes_excessive_newlines(self):
        text = "Line one.\n\n\n\nLine two."
        result = clean_prose(text)
        assert "\n\n\n" not in result

    def test_removes_excessive_spaces(self):
        text = "Word  word   word"
        result = clean_prose(text)
        assert "  " not in result

    def test_fixes_broken_lines(self):
        text = "This sentence was broken\nacross two lines."
        result = clean_prose(text)
        assert "\n" not in result

    def test_strips_whitespace(self):
        text = "  Hello world.  "
        result = clean_prose(text)
        assert result == result.strip()

    def test_empty_string(self):
        result = clean_prose("")
        assert result == ""