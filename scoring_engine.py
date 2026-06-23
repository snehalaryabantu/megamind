import json
import requests
import chromadb
from sentence_transformers import SentenceTransformer

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./credit_db")
collection = chroma_client.get_or_create_collection("intelli_credit")

WEIGHTS = {"Character": 25, "Capacity": 25, "Capital": 20, "Collateral": 15, "Conditions": 15}

def retrieve(query, n=5):
    if collection.count() == 0:
        return ""
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=min(n, collection.count()))
    return "\n\n".join(["[" + m["source"] + "]\n" + d for d, m in zip(results["documents"][0], results["metadatas"][0])])

def ask_ollama(prompt):
    r = requests.post(OLLAMA_URL + "/api/chat", json={"model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False}, timeout=120)
    return r.json()["message"]["content"]

def score_one_c(c_name, company, officer_notes):
    queries = {"Character": company + " promoter litigation fraud", "Capacity": company + " revenue GST cash flow", "Capital": company + " net worth debt equity", "Collateral": company + " assets property security", "Conditions": company + " sector RBI regulation outlook"}
    research = retrieve(queries[c_name])
    prompt = "You are a senior Indian credit analyst.\nCompany: " + company + "\nFactor: " + c_name + "\nNotes: " + (officer_notes or "None") + "\nResearch: " + (research or "Use general knowledge") + "\n\nScore " + c_name + " 0-100. Reply ONLY in JSON: {\"score\": <0-100>, \"rating\": \"<Poor/Fair/Good/Excellent>\", \"key_findings\": [\"f1\", \"f2\"], \"red_flags\": [], \"reasoning\": \"2-3 sentences\"}"
    print("  Analyzing " + c_name + "...", end="", flush=True)
    raw = ask_ollama(prompt)
    try:
        result = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
        print(" " + str(result["score"]) + "/100 (" + result["rating"] + ")")
        return result
    except:
        print(" 50/100")
        return {"score": 50, "rating": "Fair", "key_findings": [], "red_flags": [], "reasoning": raw[:200]}

def run_scoring():
    print("=" * 60)
    print("  INTELLI-CREDIT - Five Cs Scoring Engine")
    print("=" * 60)
    company = input("\nCompany name: ").strip() or "Adani Group"
    officer_notes = input("Officer notes (or Enter to skip): ").strip()
    print("\nAnalyzing " + company + "...\n")
    scores = {}
    for c in WEIGHTS:
        scores[c] = score_one_c(c, company, officer_notes)
    final_score = round(sum(scores[c]["score"] * WEIGHTS[c] / 100 for c in WEIGHTS), 1)
    red_flags = [f for c in scores for f in scores[c].get("red_flags", [])]
    if final_score >= 75 and len(red_flags) == 0:
        decision, rate, multiple = "APPROVE", "9.5%", 3.0
    elif final_score >= 65 and len(red_flags) <= 1:
        decision, rate, multiple = "APPROVE WITH CONDITIONS", "11.5%", 2.0
    elif final_score >= 50:
        decision, rate, multiple = "REFER TO CREDIT COMMITTEE", "13.5%", 1.0
    else:
        decision, rate, multiple = "REJECT", "N/A", 0
    print("\n" + "=" * 60)
    print("  FIVE Cs BREAKDOWN:")
    for c in WEIGHTS:
        s = scores[c]
        print("  " + c.ljust(12) + " [" + "#"*(s["score"]//5) + "."*(20-s["score"]//5) + "] " + str(s["score"]) + "/100 " + s["rating"])
        for f in s.get("red_flags", []):
            print("               RED FLAG: " + f)
    print("\n  FINAL SCORE: " + str(final_score) + "/100")
    print("  DECISION:    " + decision)
    if multiple > 0:
        print("  INTEREST:    " + rate)
        print("  LOAN LIMIT:  Up to " + str(multiple) + "x annual turnover")
    if red_flags:
        print("\n  RED FLAGS:")
        for i, f in enumerate(red_flags, 1):
            print("    " + str(i) + ". " + f)
    print("\n  REASONING:")
    for c in WEIGHTS:
        print("\n  [" + c + "] " + scores[c]["reasoning"])
    print("\n" + "=" * 60)
    with open("last_score.json", "w") as f:
        json.dump({"company": company, "final_score": final_score, "decision": decision, "rate": rate, "scores": scores, "red_flags": red_flags, "officer_notes": officer_notes}, f, indent=2)
    print("  Saved to last_score.json\n")

if __name__ == "__main__":
    run_scoring()
