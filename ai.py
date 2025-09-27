import json
import os
from llama_cpp import Llama
import random

# ----------------- LLaMA Model -----------------
llm = Llama(
    model_path="llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,
)

# ----------------- Memory -----------------
MEMORY_FILE = "memory.json"
MAX_MEMORY = 50  # number of messages to keep in full detail

# Load memory file
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # normalize old formats: if entries use 'who'/'text', convert into paired user/assistant
            conv = data.get("conversations") or data.get("history") or []
            # if any entry uses 'who', attempt to pair them into user/assistant
            if conv and isinstance(conv[0], dict) and 'who' in conv[0]:
                paired = []
                # pair sequential turns (father -> mimic) into user/assistant
                i = 0
                while i < len(conv) - 1:
                    a = conv[i]
                    b = conv[i+1]
                    if a.get('who') and b.get('who'):
                        # map who to user/assistant
                        if a.get('who') in ('father', 'user') and b.get('who') in ('mimic', 'jimmy', 'assistant'):
                            paired.append({"user": a.get('text',''), "assistant": b.get('text','')})
                            i += 2
                            continue
                    i += 1
                data = {"conversations": paired}
            return data
    return {"conversations": []}

# Save memory file
def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

# Store a new interaction
def remember(user, assistant):
    memory = load_memory()
    # ensure conversations list exists and consists of dicts with user/assistant
    memory.setdefault("conversations", [])
    memory["conversations"].append({"user": user, "assistant": assistant})

    # Trim old stuff if too long
    if len(memory["conversations"]) > MAX_MEMORY:
        memory["conversations"] = memory["conversations"][-MAX_MEMORY:]

    save_memory(memory)

# Recall memory for context
def recall():
    memory = load_memory()
    history = memory.get("conversations", [])
    context = ""
    for turn in history[-10:]:  # last 10 exchanges
        user = turn.get('user') if isinstance(turn, dict) else None
        assistant = turn.get('assistant') if isinstance(turn, dict) else None
        if user is None or assistant is None:
            continue
        context += f"User: {user}\nJimmy: {assistant}\n"
    return context

# ----------------- AI Wrapper -----------------
def jimmyAI(prompt):
    context = recall()
    full_prompt = f"""
You are Jimmy, a living AI assistant with memory and personality.
Here is your past memory of conversations:
{context}

Now continue with the new message:
User: {prompt}
Jimmy:
"""

    output = llm(full_prompt, max_tokens=256, stop=["User:", "Jimmy:"])
    text = output["choices"][0]["text"].strip()

    # Save this exchange to memory
    remember(prompt, text)

    return text
