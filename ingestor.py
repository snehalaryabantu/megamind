"""
Intelli-Credit Phase 2 — PDF + Web Ingestor
"""

import sys
import pymupdf
import chromadb
from sentence_transformers import SentenceTransformer
import asyncio
from crawl4ai import AsyncWebCrawler

# ── Setup ─────────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./credit_db")
collection = chroma_client.get_or_create_collection("intelli_credit")

def chunk_text(text, size=400):
    words = text.split()
    chunks = [" ".join(words[i:i+size]) for i in range(0, len(words), size)]
    return [c for c in chunks if len(c.strip()) > 80]

def store_chunks(chunks, source):
    if not chunks:
        print("⚠️  No content extracted.")
        return
    print(f"📦 Embedding {len(chunks)} chunks from: {source}")
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{source[:50]}_{i}" for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))]
    )
    print(f"✅ Stored {len(chunks)} chunks. Total in DB: {collection.count()}")

def ingest_pdf(path):
    print(f"\n📄 Reading PDF: {path}")
    doc = pymupdf.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    print(f"   Extracted {len(text)} characters from {len(doc)} pages")
    chunks = chunk_text(text)
    store_chunks(chunks, path)

async def ingest_url_async(url):
    print(f"\n🕷️  Crawling: {url}")
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        if not result.success:
            print(f"❌ Failed to crawl {url}")
            return
        chunks = chunk_text(result.markdown)
        store_chunks(chunks, url)

def ingest_url(url):
    asyncio.run(ingest_url_async(url))

def search_db(query, n=4):
    if collection.count() == 0:
        print("⚠️  Database is empty. Ingest some docs first.")
        return []
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=min(n, collection.count()))
    docs = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    print(f"\n🔍 Top {len(docs)} results for: '{query}'")
    for i, (doc, src) in enumerate(zip(docs, sources)):
        print(f"\n  [{i+1}] Source: {src[:60]}")
        print(f"       {doc[:200]}...")
    return list(zip(docs, sources))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python ingestor.py pdf <path>")
        print("  python ingestor.py url <url>")
        print("  python ingestor.py search <query>")
        print("  python ingestor.py status")
        sys.exit(0)

    cmd = sys.argv[1].lower()
    arg = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""

    if cmd == "pdf":
        ingest_pdf(arg)
    elif cmd == "url":
        ingest_url(arg)
    elif cmd == "search":
        search_db(arg)
    elif cmd == "status":
        print(f"📚 Total chunks in DB: {collection.count()}")
    else:
        print(f"❌ Unknown command: {cmd}")
