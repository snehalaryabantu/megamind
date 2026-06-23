import requests, json, sys

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2"
SYSTEM_PROMPT = "You are MegaMind, a precise and helpful AI research analyst. Give clear, structured answers. Cite your reasoning."

def chat(messages):
    full = ""
    print("\n🤖 MegaMind: ", end="", flush=True)
    with requests.post(f"{OLLAMA_URL}/api/chat", json={"model": MODEL, "messages": messages, "stream": True}, stream=True, timeout=120) as r:
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                print(token, end="", flush=True)
                full += token
                if chunk.get("done"): break
    print("\n")
    return full

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
print("="*50)
print("  🧠  MegaMind — Local AI Chat")
print("  Type 'exit' to quit")
print("="*50)
while True:
    try:
        user = input("\nYou: ").strip()
        if not user: continue
        if user.lower() in ("exit","quit"): print("Goodbye! 👋"); break
        messages.append({"role": "user", "content": user})
        reply = chat(messages)
        messages.append({"role": "assistant", "content": reply})
    except KeyboardInterrupt:
        print("\nGoodbye! 👋"); break
