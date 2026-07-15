"""
One-time utility to inspect indexed chunks in ChromaDB
so we can manually assign ground-truth chunk IDs for the eval dataset.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.vectordb.database import VectorDatabase


def inspect_all_chunks(snippet_length: int = 160):
    db = VectorDatabase()
    results = db.collection.get(include=["documents", "metadatas"])

    ids = results["ids"]
    documents = results["documents"]

    print(f"Total chunks indexed: {len(ids)}\n")
    print("=" * 80)

    for chunk_id, doc in zip(ids, documents):
        snippet = doc.replace("\n", " ").strip()
        snippet = snippet[:snippet_length] + ("..." if len(snippet) > snippet_length else "")
        print(f"ID: {chunk_id}")
        print(f"Text: {snippet}")
        print("-" * 80)


if __name__ == "__main__":
    inspect_all_chunks()