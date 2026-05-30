#!/usr/bin/env python3
"""
Document ingestion script.

Loads Markdown documents from the docs/ directory,
splits them into chunks, generates embeddings,
and stores them in Qdrant.

Usage:
    python -m scripts.ingest_docs          # Ingest all docs
    python -m scripts.ingest_docs --clear  # Clear and re-ingest

Auto re-ingest:
    Set env var AUTO_REINGEST_INTERVAL=3600 for hourly re-ingest.
    Or use a cron job: 0 * * * * cd /app && python -m scripts.ingest_docs
"""

import argparse
import asyncio
import os
from pathlib import Path
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Allow running as script or module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore


DOCS_DIR = Path(__file__).parent.parent / "docs"


def discover_files() -> List[Dict[str, str]]:
    """Find all Markdown files in docs/ with their category."""
    files = []
    if not DOCS_DIR.exists():
        print(f"⚠️  Docs directory not found: {DOCS_DIR}")
        return files

    for category_dir in sorted(DOCS_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for md_file in sorted(category_dir.glob("*.md")):
            files.append({
                "path": str(md_file),
                "category": category,
                "name": md_file.stem,
            })
    return files


async def ingest_all(clear: bool = False) -> Dict:
    """Ingest all documents into Qdrant."""
    embedder = EmbeddingService()
    vector_store = VectorStore(vector_size=embedder.dimension)

    if clear:
        print("🗑️  Clearing existing documents...")
        vector_store.clear_all()

    files = discover_files()
    if not files:
        return {"files": 0, "chunks": 0}

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )

    total_chunks = 0
    processed_files = 0

    for file_info in files:
        path = Path(file_info["path"])
        print(f"📄 Processing: {path.name} [{file_info['category']}]")

        # Read file
        content = path.read_text(encoding="utf-8")

        # Split into chunks
        chunks = text_splitter.split_text(content)

        if not chunks:
            continue

        # Generate embeddings
        embeddings = await embedder.embed(chunks)

        # Build metadata
        metadata = [
            {
                "title": file_info["name"].replace("_", " ").title(),
                "category": file_info["category"],
                "source_file": str(path),
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        # Store in Qdrant
        vector_store.upsert(chunks, embeddings, metadata)

        total_chunks += len(chunks)
        processed_files += 1
        print(f"   → {len(chunks)} chunks indexed")

    result = {"files": processed_files, "chunks": total_chunks}
    print(f"\n✅ Done: {result['files']} files, {result['chunks']} chunks total")
    print(f"   Collection: {vector_store.collection}")
    print(f"   Qdrant: {vector_store.client}")
    return result


async def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument("--clear", action="store_true", help="Clear existing documents first")
    args = parser.parse_args()

    await ingest_all(clear=args.clear)


if __name__ == "__main__":
    asyncio.run(main())
