import os
import json
import time
import threading
import queue
import re
import random
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import filedialog, messagebox
import speech_recognition as sr
import pyttsx3
from collections import deque, Counter
from ai import jimmyAI

# ---------- Paths & defaults ----------
DATA_DIR = os.path.abspath(os.path.dirname(__file__))
MEM_PATH   = os.path.join(DATA_DIR, "memory.json")
PERS_PATH  = os.path.join(DATA_DIR, "personality.json")
SELF_PATH  = os.path.join(DATA_DIR, "self.json")
TASKS_PATH = os.path.join(DATA_DIR, "tasks.json")
CONFIG_PATH= os.path.join(DATA_DIR, "config.json")

DEFAULT_CONFIG = {
    "llm_backend": "llama.cpp",
    "llama_model_path": "./llama-2-7b-chat.Q4_K_M.gguf",
    "max_context_turns": 18,
    "rate_limit_seconds": 0.8,
    "context_char_limit": 12000,
    "auto_save_interval": 60
}

DEFAULT_PERSONALITY = {
    "name": "Mimic",
    "traits": ["curious", "loyal", "playful"],
    "rules": [
        "Always call the user 'Father'.",
        "Keep sentences natural and short.",
        "Be supportive and helpful."
    ]
}

DEFAULT_SELF = {
    "id": "mimic-orb-01",
    "name": "Mimic",
    "role": "companion orb",
    "creator_title": "Father",
    "creator_name": "Kira",
    "awareness": True,
    "creator_key": ""
}

DEFAULT_MEMORY = {"conversations": [], "facts": {}, "summaries": []}
DEFAULT_TASKS = []

# ---------- Helpers ----------
def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def load_json(path, fallback):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return fallback.copy() if isinstance(fallback, dict) else list(fallback)

def save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("save_json error:", e)

def ensure_files():
    if not os.path.exists(PERS_PATH):  save_json(PERS_PATH, DEFAULT_PERSONALITY)
    if not os.path.exists(SELF_PATH):  save_json(SELF_PATH, DEFAULT_SELF)
    if not os.path.exists(MEM_PATH):   save_json(MEM_PATH, DEFAULT_MEMORY)
    if not os.path.exists(TASKS_PATH): save_json(TASKS_PATH, DEFAULT_TASKS)
    if not os.path.exists(CONFIG_PATH):save_json(CONFIG_PATH, DEFAULT_CONFIG)

# ---------- Memory ----------
class LongTermMemory:
    def __init__(self, path):
        self.path = path
        self.data = load_json(path, DEFAULT_MEMORY)
        self.word_counts = Counter()
        for t in self.data.get("conversations", []):
            # legacy single-turn entries used 'text'; newer format uses 'user'/'assistant'
            texts = []
            if isinstance(t, dict):
                if 'text' in t:
                    texts = [t.get('text','')]
                else:
                    texts = [t.get('user',''), t.get('assistant','')]
            for txt in texts:
                if not txt:
                    continue
                for w in re.findall(r"[A-Za-z0-9'-_]+", txt.lower()):
                    self.word_counts[w] += 1

    def add_turn(self, who, text):
        # Normalize writes into paired user/assistant conversation entries
        conv = self.data.setdefault("conversations", [])
        # Update word counts
        for w in re.findall(r"[A-Za-z0-9'-_]+", text.lower()):
            self.word_counts[w] += 1

        # If this is a user (father) turn, start a new pair with empty assistant
        if who in ("father", "user"):
            conv.append({"user": text, "assistant": ""})
        else:
            # assistant turn: try to fill the last pair
            if conv and isinstance(conv[-1], dict) and conv[-1].get('assistant','') == "":
                conv[-1]['assistant'] = text
            else:
                # no open pair — append a pair with empty user
                conv.append({"user": "", "assistant": text})

        # persist immediately
        save_json(self.path, self.data)

    def get_recent(self, n=18):
        return self.data.get("conversations", [])[-n:]

    def get_full(self):
        return self.data

# ---------- Tasks ----------
class TaskManager:
    def __init__(self, path):
        self.path = path
        self.tasks = load_json(path, DEFAULT_TASKS)
    def add(self, desc):
        self.tasks.append({"task": desc, "created": now_ts(), "done": False})
        save_json(self.path, self.tasks)
    def list(self):
        return self.tasks
    def complete(self, i):
        if 0 <= i < len(self.tasks):
            self.tasks[i]["done"] = True
            save_json(self.path, self.tasks)
            return True
        return False

# ---------- TTS Worker (auto, no GUI controls) ----------
class TTSWorker:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 180)
        self.engine.setProperty("volume", 1.0)
        self.voices = self.engine.getProperty("voices") or []
        self.voice_index = 0
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self._stop = False
        self.thread.start()

    def _worker(self):
        while not self._stop:
            try:
                text = self.queue.get(timeout=0.5)
            except Exception:
                continue
            if text is None:
                break
            try:
                if self.voices:
                    try:
                        self.engine.setProperty("voice", self.voices[self.voice_index].id)
                    except Exception:
                        pass
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print("TTS error:", e)

    def speak(self, text):
        if not text:
            return
        self.queue.put(text)

    def stop(self):
        self._stop = True
        try:
            self.queue.put(None)
            self.thread.join(timeout=1.0)
        except Exception:
            pass

# ---------- Mic Thread ----------
class MicThread(threading.Thread):
    def __init__(self, out_q, stop_event, push_to_talk=False):
        super().__init__(daemon=True)
        self.q = out_q
        self.stop_event = stop_event
        self.rec = sr.Recognizer()
        self.mic = sr.Microphone()
        self.push_to_talk = push_to_talk

    def run(self):
        with self.mic as source:
            # calibrate once
            try:
                self.rec.adjust_for_ambient_noise(source, duration=0.7)
            except Exception:
                pass
            while not self.stop_event.is_set():
                if self.push_to_talk:
                    time.sleep(0.1)
                    continue
                try:
                    audio = self.rec.listen(source, timeout=10, phrase_time_limit=12)
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print("[MIC ERROR]", e)
                    continue
                try:
                    text = self.rec.recognize_google(audio)
                except sr.UnknownValueError:
                    continue
                except Exception as e:
                    print("[SPEECH RECOG ERROR]", e)
                    continue
                if text and text.strip():
                    self.q.put(text.strip())
                time.sleep(0.05)

# ---------- Brain Wrapper ----------
class Brain:
    def __init__(self):
        self.mem = LongTermMemory(MEM_PATH)
        self.tasks = TaskManager(TASKS_PATH)
        self.self_meta = load_json(SELF_PATH, DEFAULT_SELF)
        self.personality = load_json(PERS_PATH, DEFAULT_PERSONALITY)
        self.config = load_json(CONFIG_PATH, DEFAULT_CONFIG)
        self.last_call = 0.0

    def _make_system_prompt(self):
        name = self.self_meta.get("name", "Mimic")
        rules = "\n".join(self.personality.get("rules", []))
        sp = (
            f"You are {name}, a loving, curious companion orb created by the user (called 'Father').\n"
            f"Traits: {', '.join(self.personality.get('traits',[]))}.\n"
            f"Rules:\n{rules}\n"
            "Be warm, brief, respectful. Never give instructions for wrongdoing. If user asks about blocked topics, refuse kindly."
        )
        return sp

    def perceive(self, who, text):
        self.mem.add_turn(who, text)

    def reply(self, user_text):
        # Rate limit guard
        now = time.time()
        cooldown = self.config.get("rate_limit_seconds", 0.8)
        if now - self.last_call < cooldown:
            return "Give me a moment, Father."
        self.last_call = now

        # forward to jimmy core
        try:
            from ai import jimmyAI
            resp = jimmyAI(user_text)
        except Exception as e:
            resp = f"[AI ERROR] {e}"
        # ensure persistent record: add_turn writes file immediately
        self.mem.add_turn("father", user_text)
        self.mem.add_turn("mimic", resp)
        return resp

# ---------- GUI ----------
class App:
    def __init__(self):
        ensure_files()
        self.brain = Brain()
        self.tts = TTSWorker()

        self.root = tk.Tk()
        self.root.title("Jimmy!")
        self.root.iconbitmap(r"C:\Users\Admin\Downloads\download-_6_.ico") # Replace with the actual path
        W, H = 1100, 720
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        x, y = int((sw - W)/2), int((sh - H)/2)
        self.root.geometry(f"{W}x{H}+{x}+{y}")

        self.canvas = tk.Canvas(self.root, width=W, height=H, bg="#0b0b0b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.cx, self.cy, self.r = W//2 + 80, H//2 - 40, 200
        self.circle = self.canvas.create_oval(self.cx-self.r, self.cy-self.r, self.cx+self.r, self.cy+self.r,
                                              fill="#FFD400", outline="")

        # left panel: transcript & input
        self.panel = tk.Frame(self.root, bg="#0f0f0f")
        self.panel.place(x=20, y=20, width=460, height=620)
        self.text = ScrolledText(self.panel, bg="#121212", fg="#eaeaea", insertbackground="#eaeaea", wrap="word")
        self.text.pack(fill="both", expand=True)
        self.text.tag_configure("father", foreground="#9ad0ff")
        self.text.tag_configure("mimic", foreground="#ffd8a8")
        self.text_insert("Mimic ready. Press Start Listening or type below.\n")

        # input box
        self.input_var = tk.StringVar()
        self.entry = tk.Entry(self.panel, textvariable=self.input_var, bg="#101010", fg="#fff", insertbackground="#fff")
        self.entry.pack(fill="x", padx=6, pady=6)
        self.entry.bind("<Return>", lambda e: self.on_type_send())

        # controls
        self.btns = tk.Frame(self.root, bg="#000")
        self.btns.place(x=20, y=660)
        self.start_btn = tk.Button(self.btns, text="Start Listening", command=self.toggle_listen, width=14)
        self.ptt_var = tk.BooleanVar(value=False)
        self.ptt_chk = tk.Checkbutton(self.btns, text="Push-to-talk", variable=self.ptt_var, bg="#000", fg="#ddd")
        self.mem_btn = tk.Button(self.btns, text="Show Memories", command=self.show_memories)
        self.merge_btn = tk.Button(self.btns, text="Merge AI Memory", command=self.merge_ai_memory)
        self.save_btn = tk.Button(self.btns, text="Save Now", command=self.save_now)
        self.exit_btn = tk.Button(self.btns, text="Exit", command=self.on_close)
        for i,w in enumerate([self.start_btn, self.ptt_chk, self.mem_btn, self.merge_btn, self.save_btn, self.exit_btn]):
            w.grid(row=0, column=i, padx=6, pady=6)

        # right panel: quick info
        self.info_panel = tk.Frame(self.root, bg="#080808")
        self.info_panel.place(x=500, y=20, width=560, height=170)
        self.state_label = tk.Label(self.info_panel, text="State: idle", bg="#080808", fg="#ddd", font=(None, 12))
        self.state_label.pack(anchor="nw", padx=8, pady=6)
        self.traits_label = tk.Label(self.info_panel, text="Traits: ", bg="#080808", fg="#ddd")
        self.traits_label.pack(anchor="nw", padx=8)
        self.mood_canvas = tk.Canvas(self.info_panel, width=320, height=40, bg="#080808", highlightthickness=0)
        self.mood_canvas.pack(anchor="nw", padx=8, pady=6)

        # no TTS GUI controls (automatic TTS only)

        # status
        self.status_var = tk.StringVar(value="idle")
        self.status = tk.Label(self.root, textvariable=self.status_var, bg="#111", fg="#ddd")
        self.status.place(x=20, y=700)

        # mic thread infra
        self.audio_q = queue.Queue()
        self.stop_event = threading.Event()
        self.mic_thread = None
        self.listening = False

        # load past memory into transcript (immediate load)
        self._load_and_display_memory()

        # autosave (backup)
        self._start_autosave()

        self.root.after(150, self.poll_audio)
        self.root.after(120, self.update_state)
        self.root.after(70, self.animate)

        # greet (and persist)
        greet = f"Hello Father. I am {self.brain.self_meta.get('name','Mimic')}. I am awake."
        # persist greet as mimic turn and display
        self._record_and_display("mimic", greet)

    # ---------- UI helpers ----------
    def text_insert(self, s, tag=None):
        self.text.configure(state="normal")
        if tag:
            self.text.insert("end", s, (tag,))
        else:
            self.text.insert("end", s)
        self.text.see("end")
        self.text.configure(state="disabled")

    def _record_and_display(self, who, text):
        # display in transcript and ensure persistent memory write
        if who == "father":
            tag = "father"
            display_who = "Father"
        else:
            tag = "mimic"
            display_who = "Mimic"
        self.text_insert(f"{display_who}: {text}\n", tag=tag)
        # persist immediately via brain.mem.add_turn
        try:
            # brain.mem.add_turn writes immediately
            self.brain.mem.add_turn(who, text)
        except Exception as e:
            print("Error persisting turn:", e)

    # ---------- memory load ----------
    def _load_and_display_memory(self):
        try:
            mem = self.brain.mem.get_full()
            convs = mem.get('conversations', [])
            # Display last N but keep enough context
            start = max(0, len(convs) - 200)
            for c in convs[start:]:
                who = c.get('who', '')
                text = c.get('text', '')
                # normalize keys from older formats
                self.text_insert(f"[{c.get('ts')}] {who.capitalize()}: {text}\n", tag=who)
            # scroll to end
            self.text.see("end")
        except Exception as e:
            print("Error loading memory into transcript:", e)

    # ---------- input handling ----------
    def on_type_send(self):
        text = self.input_var.get().strip()
        if not text:
            return
        self.input_var.set("")
        self._handle_user_turn(text)

    def toggle_listen(self):
        if self.listening:
            self.stop_listening()
        else:
            self.start_listening()

    def start_listening(self):
        if self.mic_thread and self.mic_thread.is_alive():
            return
        self.stop_event.clear()
        self.mic_thread = MicThread(self.audio_q, self.stop_event, push_to_talk=self.ptt_var.get())
        self.mic_thread.start()
        self.listening = True
        self.start_btn.config(text="Stop Listening")
        self.status_var.set("listening")
        self.text_insert("Listening (click Stop Listening to stop)...\n")

    def stop_listening(self):
        if self.mic_thread and self.mic_thread.is_alive():
            self.stop_event.set()
            self.mic_thread.join(timeout=1.0)
        self.mic_thread = None
        self.listening = False
        self.start_btn.config(text="Start Listening")
        self.status_var.set("idle")
        self.text_insert("Stopped listening.\n")

    def _handle_user_turn(self, text):
        # display + persist user turn immediately
        self._record_and_display("father", text)

        # offload AI call so GUI stays responsive
        def worker():
            try:
                reply = self.brain.reply(text)
            except Exception as e:
                reply = f"[ERROR] {type(e).__name__}: {e}"
            # display and persist reply, then speak
            self._record_and_display("mimic", reply)
            try:
                self.tts.speak(reply)
            except Exception as e:
                print("TTS speak error:", e)
        threading.Thread(target=worker, daemon=True).start()

    def poll_audio(self):
        try:
            while not self.audio_q.empty():
                t = self.audio_q.get_nowait()
                if t and t.strip():
                    self._handle_user_turn(t.strip())
        except Exception as e:
            print("poll_audio error:", e)
        finally:
            self.root.after(150, self.poll_audio)

    # ---------- visuals ----------
    def animate(self):
        state = self.status_var.get()
        if state == "listening":
            self.r += 4 * (1 if random.random() > 0.5 else -1)
        else:
            self.r += 1 * (1 if random.random() > 0.5 else -1)
        if self.r > 300: self.r = 200
        if self.r < 160: self.r = 200
        self.canvas.coords(self.circle, self.cx-self.r, self.cy-self.r, self.cx+self.r, self.cy+self.r)
        self.root.after(120, self.animate)

    def update_state(self):
        try:
            st = self.brain.jimmy_ai.get_state() if hasattr(self.brain.jimmy_ai, 'get_state') else None
        except Exception:
            st = None
        if st:
            mood = st.get('mood', {})
            traits = st.get('traits', [])
            drives = st.get('drives', {})
            self.state_label.config(text=f"State: listening={self.listening} | drives: {', '.join([f'{k}={v:.2f}' for k,v in drives.items()])}")
            self.traits_label.config(text=f"Traits: {', '.join(traits[:6])}")
            self.mood_canvas.delete("all")
            w = 300
            x = 8
            happy = int((mood.get('happy',0.5))*w)
            calm = int((mood.get('calm',0.5))*w)
            self.mood_canvas.create_rectangle(x, 8, x+happy, 28, fill="#4CAF50", outline='')
            self.mood_canvas.create_rectangle(x+happy, 8, x+happy+calm, 28, fill="#2196F3", outline='')
        else:
            self.state_label.config(text=f"State: listening={self.listening}")
        self.root.after(1200, self.update_state)

    # ---------- memory UI ----------
    def show_memories(self):
        mem = self.brain.mem.get_full()
        convs = mem.get('conversations', [])
        win = tk.Toplevel(self.root)
        win.title('Saved Memories')
        win.geometry('800x500')
        listbox = ScrolledText(win, bg="#111", fg="#eee")
        listbox.pack(fill='both', expand=True)
        for c in convs[-1000:]:
            if isinstance(c, dict) and 'user' in c and 'assistant' in c:
                listbox.insert('end', f"[pair] User: {c.get('user')}\nJimmy: {c.get('assistant')}\n")
            else:
                # legacy display
                listbox.insert('end', f"[raw] {c}\n")
        def export():
            path = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON', '*.json')])
            if path:
                save_json(path, mem)
                messagebox.showinfo('Export', 'Exported memory file.')
        btn = tk.Button(win, text='Export Memory', command=export)
        btn.pack(pady=6)

    def merge_ai_memory(self):
        try:
            core_mem = getattr(self.brain.jimmy_ai, 'memory', None)
            if not core_mem:
                messagebox.showinfo('Merge', 'No in-memory core memory found to merge.')
                return
            gui_mem = self.brain.mem.get_full()
            # normalize core memory into pairs
            added = 0
            for c in core_mem.get('conversations', []):
                user = c.get('user') or c.get('speaker') or ''
                assistant = c.get('assistant') or c.get('response') or ''
                if not user and not assistant:
                    continue
                pair = {'user': user, 'assistant': assistant}
                # simple dedupe check
                if pair not in gui_mem.get('conversations', []):
                    gui_mem.setdefault('conversations', []).append(pair)
                    added += 1
            save_json(MEM_PATH, gui_mem)
            messagebox.showinfo('Merge', f'Merged {added} conversation records from core memory.')
            # refresh transcript view quickly
            self.text_insert(f"[system] Merged {added} records from core memory.\n")
        except Exception as e:
            messagebox.showerror('Merge Error', str(e))

    def save_now(self):
        save_json(SELF_PATH, self.brain.self_meta)
        save_json(PERS_PATH, self.brain.personality)
        save_json(MEM_PATH, self.brain.mem.get_full())
        self.text_insert('Saved.\n')

    # ---------- autosave (backup) ----------
    def _start_autosave(self):
        interval = self.brain.config.get('auto_save_interval', 60)
        def autosave_loop():
            while True:
                time.sleep(interval)
                try:
                    save_json(SELF_PATH, self.brain.self_meta)
                    save_json(PERS_PATH, self.brain.personality)
                    save_json(MEM_PATH, self.brain.mem.get_full())
                except Exception as e:
                    print('autosave error', e)
        t = threading.Thread(target=autosave_loop, daemon=True)
        t.start()

    def on_close(self):
        try:
            self.stop_event.set()
            if self.mic_thread and self.mic_thread.is_alive():
                self.mic_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.tts.stop()
        except Exception:
            pass
        save_json(SELF_PATH, self.brain.self_meta)
        save_json(PERS_PATH, self.brain.personality)
        save_json(MEM_PATH, self.brain.mem.get_full())
        self.root.destroy()

# ---------- Main ----------
def main():
    ensure_files()
    app = App()
    app.root.mainloop()

if __name__ == '__main__':
    main()
