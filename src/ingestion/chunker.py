# src/ingestion/chunker.py

import re
import uuid
from docling.datamodel.base_models import DocItemLabel
from src.ingestion.section_detector import detect_section, save_section_cache, section_cache  


SECTION_BOUNDARY_PHRASES = [
    "item 1a", "risk factor",
    "item 1. legal", "legal proceeding",
    "the company is subject to various legal",
    "strategic and competitive risks",
    "this item and other sections of this quarterly report",
    "part ii. other", "part ii other"
]


def get_page_no(element) -> int:
    try:
        return element.prov[0].page_no
    except:
        return None


def get_part_and_item(text: str) -> dict:
    text_lower = text.lower()
    part = None
    item = None
    if "part i" in text_lower and "part ii" not in text_lower:
        part = "Part I"
    elif "part ii" in text_lower:
        part = "Part II"
    for i in range(1, 5):
        if f"item {i}." in text_lower or f"item {i} " in text_lower:
            item = f"Item {i}"
            break
    return {"part": part, "item": item}


def clean_prose(text: str) -> str:
    text = re.sub(r'(?<=[a-z,;])\n(?=[a-z])', ' ', text)
    text = re.sub(r'\n\d{1,3}\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'(Apple Inc\.|Microsoft Corporation)\s*\n', '', text)
    text = re.sub(r'Form 10-Q\s*\n', '', text)
    return text.strip()


def split_into_sentences(text: str) -> list:
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def is_meaningful_row(row: str) -> bool:
    cells = [c.strip() for c in row.split('|') if c.strip()]
    if not cells:
        return False
    non_empty = [c for c in cells if c and c != '-']
    if len(non_empty) < 2:
        return False
    date_patterns = [
        'march', 'june', 'september', 'december',
        'january', 'february', 'april', 'july',
        'august', 'october', 'november'
    ]
    if any(month in row.lower() for month in date_patterns):
        return False
    has_financial = any(
        '$' in cell or
        (any(char.isdigit() for char in cell) and ',' in cell)
        for cell in cells
    )
    return has_financial


def split_table_into_rows(table_markdown: str) -> list:
    lines = table_markdown.strip().split('\n')
    header = None
    separator = None
    data_rows = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^\|[-|\s:]+\|$', line):
            separator = line
            continue
        if header is None:
            header = line
            continue
        if line.startswith('|'):
            data_rows.append(line)

    children = []
    for row in data_rows:
        if not is_meaningful_row(row):
            continue
        child_content = (
            f"{header}\n{separator}\n{row}"
            if separator else f"{header}\n{row}"
        )
        children.append(child_content)
    return children


def create_parent_child_chunks(ticker: str, result, filing_metadata: dict):
    parents = {}
    children = []
    current_section = "General"
    current_part = None
    current_item = None
    doc = result.document
    child_index = 0

    for element, level in doc.iterate_items():
        page_no = get_page_no(element)

        if hasattr(element, 'label') and element.label in [
            DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER
        ]:
            continue

        if hasattr(element, 'label') and element.label in [
            DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE
        ]:
            if hasattr(element, 'text'):
                detected = detect_section(element.text)
                if detected != "General":
                    current_section = detected
                pi = get_part_and_item(element.text)
                if pi["part"]:
                    current_part = pi["part"]
                if pi["item"]:
                    current_item = pi["item"]
            continue

        base_metadata = {
            **filing_metadata,
            "section": current_section,
            "part": current_part,
            "item": current_item,
            "page": page_no,
        }

        if hasattr(element, 'data'):
            table_text = element.export_to_markdown(doc=doc)
            if not table_text.strip():
                continue

            parent_id = str(uuid.uuid4())
            parents[parent_id] = {
                "content": table_text,
                "metadata": {**base_metadata, "chunk_type": "table"}
            }

            for row in split_table_into_rows(table_text):
                children.append({
                    "content": row,
                    "metadata": {
                        **base_metadata,
                        "chunk_type": "table",
                        "parent_id": parent_id,
                        "child_index": child_index
                    }
                })
                child_index += 1
            continue

        if hasattr(element, 'text') and element.text.strip():
            prose_text = clean_prose(element.text.strip())
            if not prose_text:
                continue

            text_lower = prose_text.lower()
            is_boundary = any(
                phrase in text_lower
                for phrase in SECTION_BOUNDARY_PHRASES
            )
            if is_boundary:
                detected = detect_section(prose_text[:200])
                if detected != "General":
                    current_section = detected
                    continue

            parent_id = str(uuid.uuid4())
            parents[parent_id] = {
                "content": prose_text,
                "metadata": {**base_metadata, "chunk_type": "prose"}
            }

            sentences = split_into_sentences(prose_text)
            for i in range(0, len(sentences), 2):
                pair = sentences[i:i + 2]
                child_text = " ".join(pair)
                if child_text.strip():
                    children.append({
                        "content": child_text,
                        "metadata": {
                            **base_metadata,
                            "chunk_type": "prose",
                            "parent_id": parent_id,
                            "child_index": child_index
                        }
                    })
                    child_index += 1

    save_section_cache(section_cache)
    return parents, children