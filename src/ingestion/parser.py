
import json
import os
import pickle
from docling.document_converter import DocumentConverter

converter = DocumentConverter()


def convert_filings(filings: list) -> dict:
    """
    Convert PDF filings to Docling result objects.
    Returns dict: {ticker: {result, metadata}}
    """
    all_results = {}

    for filing in filings:
        print(f"Converting {filing['company']}...")
        result = converter.convert(filing["local_path"])
        all_results[filing["ticker"]] = {
            "result": result,
            "metadata": {
                "company": filing["company"],
                "ticker": filing["ticker"],
                "period": filing["period"],
                "filing_date": filing["filing_date"],
                "filing_type": filing["filing_type"],
                "currency": filing["currency"],
                "source": filing["local_path"].split("/")[-1]
            }
        }
        print(f"  ✓ {filing['ticker']} converted")

    return all_results


def save_chunks(all_children: dict, all_parents: dict, data_dir: str):
    """Save chunks to disk for fast reload."""
    os.makedirs(data_dir, exist_ok=True)

    with open(os.path.join(data_dir, "all_children.pkl"), "wb") as f:
        pickle.dump(all_children, f)

    with open(os.path.join(data_dir, "all_parents.pkl"), "wb") as f:
        pickle.dump(all_parents, f)

    print(f"✓ Chunks saved to {data_dir}")


def load_chunks(data_dir: str) -> tuple:
    """Load chunks from disk."""
    with open(os.path.join(data_dir, "all_children.pkl"), "rb") as f:
        all_children = pickle.load(f)

    with open(os.path.join(data_dir, "all_parents.pkl"), "rb") as f:
        all_parents = pickle.load(f)

    return all_children, all_parents


def save_docstore(docstore: dict, data_dir: str):
    """Save docstore to JSON."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "docstore.json")
    with open(path, "w") as f:
        json.dump(docstore, f)
    print(f"✓ Docstore saved: {len(docstore)} parents")


def load_docstore(data_dir: str) -> dict:
    """Load docstore from JSON."""
    path = os.path.join(data_dir, "docstore.json")
    with open(path, "r") as f:
        return json.load(f)