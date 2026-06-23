import sys, json, requests, asyncio, chromadb
from sentence_transformers import SentenceTransformer
from crawl4ai import AsyncWebCrawler

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./credit_db")
collection = chroma_client.get_or_create_collection("intelli_credit")

def retrieve(query, n=5):
    if collection.count() == 0:
        return []
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=min(n, collection.count()))
    return list(zip(results["documents"][0], [m["source"] for m in results["metadatas"][0]]))

async def crawl(url):
    print("Crawling: " + url)
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        if not result.success:
            print("Failed to crawl " + url)
            return
        words = result.markdown.split()
        chunks = [" ".join(words[i:i+400]) for i in range(0, len(words), 400)]
        chunks = [c for c in chunks if len(c.strip()) > 80]
        embeddings = embedder.encode(chunks).tolist()
        collection.add(documents=chunks, embeddings=embeddings,
            ids=[f"{url[:50]}_{i}" for i in range(len(chunks))],
            metadatas=[{"source": url, "chunk": i} for i in range(len(chunks))])
        print("Stored " + str(len(chunks)) + " chunks")

def ask_llm(query, chunks):
    context = "\n\n---\n\n".join(["[Source: " + src + "]\n" + doc for doc, src in chunks]) if chunks else ""
    system = ("You are an expert Indian corporate credit analyst.\nAnalyze research and answer the credit officer.\nHighlight RED FLAGS clearly.\nCite sources.\n\nRESEARCH:\n" + context) if context else "You are an expert Indian corporate credit analyst."
    print("\nCredit Agent: ", end="", flush=True)
    full = ""
    with requests.post(OLLAMA_URL + "/api/chat", json={"model": MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": query}], "stream": True}, stream=True, timeout=120) as r:
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                print(token, end="", flush=True)
                full += token
                if chunk.get("done"): break
    print("\n")
    return full

print("=" * 50)
print("  Intelli-Credit - Research Agent")
print("  crawl <url> | ask <question> | exit")
print("=" * 50)
print("Docs in memory: " + str(collection.count()) + " chunks\n")
while True:
    try:
        user = input("Credit Officer: ").strip()
        if not user: continue
        if user.lower() == "exit": print("Goodbye!"); break
        elif user.lower() == "status": print("Chunks: " + str(collection.count()))
        elif user.lower().startswith("crawl "): asyncio.run(crawl(user[6:].strip()))
        else:
            query = user[4:].strip() if user.lower().startswith("ask ") else user
            print("Searching...")
            ask_llm(query, retrieve(query))
    except KeyboardInterrupt: print("\nGoodbye!"); break
    except Exception as e: print("Error: " + str(e))
