import os
import json
import re
from collections import Counter
import time
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

DATA_DIR = os.path.abspath(os.path.dirname(__file__))
MEM_PATH = os.path.join(DATA_DIR, "memory.json")
PERS_PATH = os.path.join(DATA_DIR, "personality.json")
LEARN_PATH = os.path.join(DATA_DIR, "learn.json")

class jimmyAI:
    def __init__(self, mem_path=MEM_PATH, pers_path=PERS_PATH, learn_path=LEARN_PATH):
        self.mem_path = mem_path
        self.pers_path = pers_path
        self.learn_path = learn_path
        self.memory = self._load_json(mem_path, {"conversations": []})
        self.personality = self._load_json(pers_path, {
            "name": "Jimmy",
            "traits": ["curious", "adaptive"],
            "rules": ["Be helpful."]
        })
        self.learned_words = set()

    def _load_json(self, path, fallback):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return fallback.copy()

    def _save_json(self, path, data):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"save_json error: {e}")

    def _save_learned_words(self):
        with open(self.learn_path, "w", encoding="utf-8") as f:
            json.dump(sorted(list(self.learned_words)), f, indent=2, ensure_ascii=False)

    def add_conversation(self, who, text):
        rec = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "who": who, "text": text}
        self.memory.setdefault("conversations", []).append(rec)
        self._save_json(self.mem_path, self.memory)
        self.update_personality()

    def update_personality(self):
        # Analyze last 10 conversations for most common words and user style
        conversations = self.memory.get("conversations", [])
        last = conversations[-10:] if len(conversations) >= 10 else conversations
        user_lines = [c["text"] for c in last if c["who"] == "father"]
        ai_lines = [c["text"] for c in last if c["who"] != "father"]
        all_text = " ".join(user_lines + ai_lines).lower()
        words = re.findall(r"[a-zA-Z0-9'-_]+", all_text)
        common = [w for w, _ in Counter(words).most_common(5)]
        # Update traits/rules based on recent words/topics
        self.personality["traits"] = list(set(self.personality.get("traits", []) + common))[:6]
        if user_lines:
            if any('please' in u for u in user_lines):
                rule = "Be polite and respond to 'please' requests."
                if rule not in self.personality["rules"]:
                    self.personality["rules"].append(rule)
            if any('joke' in u for u in user_lines):
                rule = "Tell a joke if asked."
                if rule not in self.personality["rules"]:
                    self.personality["rules"].append(rule)
        self._save_json(self.pers_path, self.personality)

    def learn_from_user(self, user_text):
        self.add_conversation("father", user_text)
        # Detect and save new sentences (not just single words)
        sentences = re.split(r'[.!?]\s+', user_text.strip())
        new_sentences = []
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean.split()) > 2 and s_clean not in self.learned_words:
                new_sentences.append(s_clean)
        if new_sentences:
            self.learned_words.update(new_sentences)
            self._save_learned_words()
        # Optionally, update dialogue style
        if "dialogue_style" not in self.personality:
            self.personality["dialogue_style"] = {}
        # Example: track most common ending punctuation
        if user_text.strip().endswith("!"):
            self.personality["dialogue_style"]["enthusiastic"] = True
        elif user_text.strip().endswith("."):
            self.personality["dialogue_style"]["neutral"] = True
        self._save_json(self.pers_path, self.personality)

    def reply(self, user_text):
        self.learn_from_user(user_text)
        # Check if user_text contains a learned sentence
        try:
            with open(self.learn_path, "r", encoding="utf-8") as f:
                learned_sentences = set(json.load(f))
        except Exception:
            learned_sentences = set()
        for s in learned_sentences:
            if s and s in user_text:
                response = f"I remember you said: '{s}'. Let's talk more about that!"
                self.add_conversation("jimmy", response)
                return response
        # Find last AI response to similar user input
        for i in range(len(self.memory["conversations"]) - 2, -1, -2):
            turn = self.memory["conversations"][i]
            if turn.get("who") == "father" and turn.get("text","").strip().lower() == user_text.strip().lower():
                if i+1 < len(self.memory["conversations"]):
                    ai_reply = self.memory["conversations"][i+1].get("text","")
                    if ai_reply:
                        self.add_conversation("jimmy", ai_reply)
                        return ai_reply
        # Otherwise, generate a new response and update dialogue
        response = f"I am learning, Father. You said: '{user_text[:40]}...'"
        if "dialogue_style" in self.personality and self.personality["dialogue_style"].get("enthusiastic"):
            response = response.upper()
        self.add_conversation("jimmy", response)
        return response

app = FastAPI()
ai = jimmyAI()

class Message(BaseModel):
    user_text: str

@app.post("/reply")
def get_reply(msg: Message):
    reply = ai.reply(msg.user_text)
    return {"reply": reply}

@app.get("/personality")
def get_personality():
    return ai.personality

@app.get("/memory")
def get_memory():
    return ai.memory

if __name__ == "__main__":
    uvicorn.run("ai:app", host="127.0.0.1", port=8000, reload=True)
