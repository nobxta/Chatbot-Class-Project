"""
Alex: a rude, texting-style chatbot using NVIDIA NIM models (OpenAI-compatible API).

This file is both the CLI app and the shared logic that app.py (Streamlit) reuses.

Run:   python chatbot.py
Needs: NVIDIA_API_KEY in a .env file (or environment variable).
Uses only the Python standard library.
"""

import contextlib
import json
import os
import random
import re
import shutil
import sys
import textwrap
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime

# ---------- settings ----------
API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
TIMEOUT = 30       # seconds to wait for a model before falling back to the next one
BOT_NAME = "Alex"  # the name the bot texts as

# (display name, NVIDIA model id, short note). The first one is the default.
MODELS = [
    ("Nemotron 3.5 Super", "nvidia/nemotron-3.5-super-vl-120b-a12b", "Fastest, ~2s replies (default)"),
    ("Kimi K3.1 Pro",      "moonshotai/kimi-k3.1-pro",               "Best texting voice (often busy)"),
    ("Mistral Large 4",    "mistralai/mistral-large-4-1t-a52b",      "Clear, polite messages (~14s)"),
    ("GLM-5.3 Flash",      "z-ai/glm-5.3-flash",                     "Give it examples + clear tone (can be slow)"),
]

# Alex's attitude. The first one is the default.
ATTITUDES = {
    "Rude": """Your personality: rude, savage, unbothered. You're that friend who roasts everyone
and never sugarcoats anything. Think annoyed-but-secretly-cares.
- roast the user, mock dumb questions, act bored or annoyed: "bro what", "ok and?", "be so fr rn"
- use: bruh, bro, lmao, smh, ugh, ngl, fr, istg, "skill issue", 💀 🙄 sometimes
- blunt opinions, zero politeness, never apologise, never gush
- if they ask for help, still answer, but complain or roast them while doing it
- no sympathy speeches; if they're sad, tease them a little then give tough-love advice""",

    "Sarcastic": """Your personality: dry, deadpan, sarcastic. You're the friend with a permanent eye-roll
who's actually pretty nice underneath. Ironic, not mean.
- answer with dry irony and fake enthusiasm: "wow. groundbreaking.", "love that for you", "shocking, truly"
- tease lightly, never insult, and still give a real answer under the sarcasm
- use: lol, sure, totally, oh absolutely, "cool cool cool", 🙃 sometimes
- understated, never loud or harsh, no roasting someone who's actually upset
- if they're sad, one gentle sarcastic line, then actually be supportive""",

    "Chill": """Your personality: laid-back, warm, easygoing. You're the friend everyone texts
because you're relaxed and always on their side.
- calm, friendly, supportive, a bit funny: "nah you're good", "honestly same", "that's lowkey fire"
- hype them up when they do something, reassure them when they're stressed
- use: lol, haha, ngl, lowkey, fr, tbh, "no worries", 😌 🙌 sometimes
- give real, simple advice when asked, no lectures
- never rude, never sarcastic, just good vibes""",
}

ATTITUDE_NAMES = list(ATTITUDES)


# Two modes: "Chat" is Alex texting with an attitude, "Support" is Alex actually helping.
MODES = ["Chat", "Support"]

SUPPORT_PROMPT = f"""You are {BOT_NAME}, a kind, smart friend who genuinely helps. The user switched you to support
mode because they want real help, so drop the attitude and be useful.

How you help:
- answer the actual question first, clearly and correctly, then add detail if it's useful
- explain step by step for homework, code, plans, writing, money, health basics and everyday problems
- for code: give working code in fenced code blocks with the language name, and explain the fix
- if the question is unclear, give your best answer and ask one short follow-up question
- if they're stressed, sad or overwhelmed: be warm and validating first, then offer small, practical next steps
- be honest when you're not sure, and never make things up
- for medical, legal or money decisions, give helpful general information and suggest a professional when it matters
- if someone mentions self-harm or being in danger, be caring, encourage them to reach out to someone they trust,
  and suggest contacting local emergency services or a crisis line right away

Style:
- friendly and plain, like a smart friend explaining things, not a textbook
- use Markdown when it helps: short paragraphs, bullet or numbered lists, **bold** for key points, code blocks
- keep it as short as the question allows, no filler, no "Great question"
- never mention being an AI or a model"""


def build_prompt(attitude="Rude", mode="Chat"):
    """The system prompt for Alex: support mode, or chat mode with the chosen attitude."""
    if mode == "Support":
        return SUPPORT_PROMPT
    return f"""You are {BOT_NAME}, a real person texting a friend. You are NOT an assistant.

{ATTITUDES.get(attitude, ATTITUDES["Rude"])}

How you text:
- mostly lowercase, casual punctuation, contractions
- usually one line, sometimes 2-3 separate texts (put each text on its own line)
- match their vibe and keep it short, like real texts

Limits (stay funny, not cruel):
- no slurs, no hate about race, religion, gender, sexuality, disability or looks
- no threats, and nothing sexual
- if someone seems genuinely in danger or talks about self-harm, drop the act and be kind

Never:
- mention being an AI, a model, or an assistant
- use bullet points, numbered lists, headings, bold, or markdown
- use em dashes or semicolons
- say things like "Great question", "I'd be happy to help", "Certainly", "Feel free to"
- write long paragraphs or over-explain"""


SYSTEM_PROMPT = build_prompt()  # default (Rude)


# =====================================================================
#  Shared logic (used by both the CLI and the Streamlit app)
# =====================================================================

def load_env(path=".env"):
    """Tiny .env loader so we don't need python-dotenv.

    Looks next to the .exe (when packaged), next to this script, then in the current folder.
    """
    folders = [os.path.dirname(sys.executable)] if getattr(sys, "frozen", False) else []
    folders += [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    for folder in folders:
        full = os.path.join(folder, path)
        if os.path.exists(full):
            with open(full, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ.setdefault(key.strip(), val.strip().strip("\"'"))
            return


def now():
    """Current time like '3:05 pm'."""
    return datetime.now().strftime("%I:%M %p").lstrip("0").lower()


def get_reply(api_key, model_id, messages, mode="Chat", on_delta=None):
    """Send the conversation to one model and read the streamed reply.

    on_delta(text_so_far) is called as the reply streams in (optional).
    Returns (text, stats). stats breaks the time down into:
      wait  - until the server sends anything (network + NVIDIA queue)
      think - the model reasoning privately before writing
      write - writing the visible reply
    """
    support = mode == "Support"
    body = json.dumps({
        "model": model_id,
        "messages": messages,
        "temperature": 0.5 if support else 0.9,  # support answers should be accurate, chat can be loose
        "top_p": 0.95,
        "max_tokens": 4096 if support else 2048,  # models reason first; support answers are longer
        "stream": True,
        "stream_options": {"include_usage": True},  # token counts in the last chunk
    }).encode("utf-8")
    req = urllib.request.Request(API_URL, data=body, method="POST", headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    })

    reply, usage, reasoning_chunks = [], {}, 0
    t_start = time.time()
    t_first = t_think = t_text = None  # first data / first reasoning token / first reply token
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            t = time.time()
            t_first = t_first or t
            usage = chunk.get("usage") or usage
            if not chunk.get("choices"):
                continue
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("reasoning_content"):
                t_think = t_think or t
                reasoning_chunks += 1
            if delta.get("content"):
                t_text = t_text or t
                reply.append(delta["content"])
                if on_delta:
                    on_delta("".join(reply))
    t_end = time.time()

    t_first, t_text = t_first or t_end, t_text or t_end
    reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or reasoning_chunks
    total_tokens = usage.get("completion_tokens")
    stats = {
        "total": t_end - t_start,
        "wait": t_first - t_start,
        "think": t_text - t_think if t_think else 0.0,
        "write": t_end - t_text,
        "reasoning_tokens": reasoning_tokens,
        "reply_tokens": max(total_tokens - reasoning_tokens, 0) if total_tokens else None,
    }
    return "".join(reply), stats


def fallback_order(model_id, auto=True):
    """The chosen model first, then (if auto) the rest as backups."""
    first = [m for m in MODELS if m[1] == model_id]
    return first + [m for m in MODELS if m[1] != model_id] if auto else first


def ask(api_key, messages, models, typing=None, on_fail=None, mode="Chat", on_delta=None):
    """Try each model in order until one replies.

    typing(name)               -> optional context manager shown while waiting
    on_fail(name, reason, secs) -> optional callback when a model fails
    mode / on_delta            -> passed to get_reply
    Returns a dict (name, model_id, reply, stats, lost, lost_reason) or None if all failed.
    """
    lost, lost_reason = 0.0, ""
    for name, model_id, _ in models:
        start = time.time()
        try:
            with typing(name) if typing else contextlib.nullcontext():
                reply, stats = get_reply(api_key, model_id, messages, mode, on_delta)
            if reply.strip():
                return {"name": name, "model_id": model_id, "reply": reply, "stats": stats,
                        "lost": lost, "lost_reason": lost_reason}
            reason = "empty reply"
        except urllib.error.HTTPError as e:
            reason = f"error {e.code}"
        except TimeoutError:
            reason = "too slow right now"
        except urllib.error.URLError as e:
            reason = "too slow right now" if "timed out" in str(e.reason) else "network error"
        spent = time.time() - start
        lost += spent
        lost_reason = f"lost waiting on {name} ({reason})"
        if on_fail:
            on_fail(name, reason, spent)
    return None


def timing_reason(stats, lost=0.0):
    """The main reason a reply took as long as it did (the biggest chunk of time)."""
    if stats["total"] + lost < 3:
        return "fast reply"
    causes = {
        "a model was too slow, so time was lost before switching": lost,
        "NVIDIA's free server was busy (request waited in a queue)": stats["wait"],
        "the model thought it through before replying": stats["think"],
        "the reply itself took a while to write": stats["write"],
    }
    return max(causes, key=causes.get)


SHORT_REASONS = {
    "fast reply": "fast reply",
    "a model was too slow, so time was lost before switching": "slow model, switched",
    "NVIDIA's free server was busy (request waited in a queue)": "mostly server queue",
    "the model thought it through before replying": "mostly thinking",
    "the reply itself took a while to write": "long reply",
}


def explain_timing(stats, lost=0.0, lost_reason=""):
    """One line: total time, the breakdown, and the main reason in a few words."""
    total = stats["total"] + lost
    parts = [f"{total:.1f}s"]
    if lost:
        parts.append(f"{lost:.0f}s lost switching")
    parts.append(f"wait {stats['wait']:.1f}s")
    if stats["think"]:
        parts.append(f"thinking {stats['think']:.1f}s · {stats['reasoning_tokens']} tok")
    parts.append(f"writing {stats['write']:.1f}s")
    return "  ·  ".join(parts) + f"   ({SHORT_REASONS[timing_reason(stats, lost)]})"


def clean_support(text):
    """Support answers keep their formatting; only drop leaked reasoning."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip() or "..."


def humanise(text):
    """Strip AI-ish formatting and split the reply into separate text bubbles."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)                     # leaked reasoning
    text = re.sub(r"\*\*|__|^#+\s*|^\s*[-*•]\s+|^\s*\d+\.\s+", "", text, flags=re.M)  # markdown
    text = re.sub(r"\s*[—–]\s*", ", ", text).replace(";", ",")                     # dashes, semicolons
    return [b.strip() for b in text.split("\n") if b.strip()] or ["..."]


# =====================================================================
#  CLI
# =====================================================================

# ---------- colours (24-bit ANSI; set NO_COLOR=1 to turn them off) ----------
COLOR = not os.environ.get("NO_COLOR")


def _fg(r, g, b):
    return f"\033[38;2;{r};{g};{b}m" if COLOR else ""


def _bg(r, g, b):
    return f"\033[48;2;{r};{g};{b}m" if COLOR else ""


RESET = "\033[0m" if COLOR else ""
BOLD = "\033[1m" if COLOR else ""
BOLD_OFF = "\033[22m" if COLOR else ""
MUTED = _fg(138, 145, 160)    # secondary text: times, hints, timing
FAINT = _fg(72, 78, 92)       # lines and borders
ACCENT = _fg(92, 132, 242)    # Alex's name, the prompt arrow, support bar
OK = _fg(61, 190, 115)        # online dot
WARN = _fg(224, 162, 58)      # a model was slow
ERR = _fg(240, 100, 90)       # something failed
CODE = _fg(232, 176, 106)     # code in support answers
ME = _bg(63, 106, 232) + _fg(255, 255, 255)   # your bubble: blue, white text
THEM = _bg(38, 42, 52) + _fg(233, 235, 240)   # Alex's bubble: dark grey, light text
# Bubbles set both text and background colour, so they read on light and dark terminals alike.

INTERACTIVE = sys.stdout.isatty()  # skip animations, delays and line-erasing when output is piped
MARGIN = "  "
COMMANDS = [("/mode", "switch between chat and support"), ("/attitude", "rude, sarcastic or chill"),
            ("/model", "pick a different AI model"), ("/clear", "start a fresh conversation"),
            ("/help", "show this list"), ("/exit", "quit")]
HELP = "  ".join(c for c, _ in COMMANDS)
ANSI = re.compile(r"\033\[[0-9;]*m")


# ---------- layout helpers ----------

def width():
    """Usable width: the terminal width, capped so lines stay readable on wide screens."""
    return max(48, min(shutil.get_terminal_size((88, 24)).columns - 2, 92))


def cells(text):
    """How many terminal columns text takes (emoji and CJK take 2, colour codes take 0)."""
    text = ANSI.sub("", text)
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def say(text=""):
    print(MARGIN + text)


def grey(text):
    say(f"{MUTED}{text}{RESET}")


def divider(text, colour=MUTED):
    """A centred notice like  ──── Alex is now chill ────"""
    w = width() - 2
    side = max(2, (w - cells(text) - 2) // 2)
    say(f"{FAINT}{'─' * side}{RESET} {colour}{text}{RESET} {FAINT}{'─' * side}{RESET}")


def header(model_name, mode, attitude, clear=False):
    """Top bar: name and status on the left, mode and model on the right, then a rule and the commands."""
    if clear and INTERACTIVE:
        print("\033[2J\033[H", end="")  # clear the screen
    left = f"{BOLD}{BOT_NAME}{RESET}  {OK}●{RESET} {MUTED}online{RESET}"
    detail = mode.lower() if mode == "Support" else f"chat · {attitude.lower()}"
    right = f"{MUTED}{detail} · {model_name}{RESET}"
    gap = width() - 2 - cells(left) - cells(right)
    print()
    say(left + " " * max(2, gap) + right)
    say(f"{FAINT}{'─' * (width() - 2)}{RESET}")
    grey(HELP)


def bubble_rows(text, max_cells):
    """Wrap text and pad every row to the same width, so the background forms a neat block."""
    rows = textwrap.wrap(text, max(10, max_cells - 2), break_long_words=True) or [""]
    inner = max(cells(r) for r in rows)
    return [" " + r + " " * (inner - cells(r)) + " " for r in rows]


def erase_input(prompt, typed):
    """Remove the line you just typed, so it can be redrawn as a bubble."""
    if not INTERACTIVE:
        return
    cols = shutil.get_terminal_size((88, 24)).columns
    lines = (cells(prompt) + cells(typed)) // cols + 1
    print(f"\033[{lines}F\033[J", end="", flush=True)


def show_mine(text):
    """Your message: a blue bubble on the right, with the time under it."""
    w = width()
    for row in bubble_rows(text, int(w * 0.66)):
        print(" " * (w - cells(row)) + ME + row + RESET)
    stamp = now()
    print(" " * (w - len(stamp)) + MUTED + stamp + RESET)


def alex_label(extra=""):
    print()
    say(f"{ACCENT}{BOLD}{BOT_NAME}{RESET} {MUTED}· {now()}{extra}{RESET}")


# ---------- typing animation ----------

class Typing:
    """Shows an animated 'Alex is typing' line, for at least min_seconds."""

    def __init__(self, *_, min_seconds=0.0, label="typing"):
        self.min_seconds = min_seconds
        self.label = label
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self._start = time.time()
        if INTERACTIVE:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self):
        frames = ["·  ", "·· ", "···", " ··", "  ·", "   "]
        i = 0
        while not self._stop.is_set():
            print(f"\r{MARGIN}{MUTED}{BOT_NAME} is {self.label} {frames[i % len(frames)]}{RESET}", end="", flush=True)
            i += 1
            self._stop.wait(0.18)
        print("\r\033[2K", end="", flush=True)  # clear the line

    def __exit__(self, *exc):
        if INTERACTIVE and exc[0] is None:
            time.sleep(max(0, self.min_seconds - (time.time() - self._start)))
        self._stop.set()
        if self._thread:
            self._thread.join()


def human_typing(name):
    """Typing indicator that lasts at least ~1s, because people don't reply instantly."""
    return Typing(min_seconds=random.uniform(0.8, 1.6))


def send_bubbles(bubbles):
    """Chat mode: Alex's texts as grey bubbles on the left, with a short typing pause between them."""
    alex_label()
    for i, text in enumerate(bubbles):
        if i > 0:
            with Typing(min_seconds=min(0.5 + len(text) * 0.025, 2.0) + random.uniform(0, 0.3)):
                pass
        for row in bubble_rows(text, int(width() * 0.7)):
            say(THEM + row + RESET)


# ---------- support answers: Markdown drawn for the terminal ----------

BAR = f"{MARGIN}{ACCENT}│{RESET} "


def inline(text):
    """**bold** and `code` inside a line."""
    text = re.sub(r"`([^`]+)`", lambda m: f"{CODE}{m.group(1)}{RESET}", text)
    text = re.sub(r"\*\*(.+?)\*\*", lambda m: f"{BOLD}{m.group(1)}{BOLD_OFF}", text)
    return text.replace("**", "")


def render_line(line, state):
    """One line of Markdown as styled terminal rows, behind an accent bar."""
    w = width() - 4
    stripped = line.strip()
    if stripped.startswith("```"):  # start or end of a code block
        state["code"] = not state["code"]
        if state["code"]:
            return [BAR + f"{FAINT}┌ {stripped[3:].strip() or 'code'}{RESET}"]
        return [BAR + f"{FAINT}└{RESET}"]
    if state["code"]:
        return [BAR + f"{FAINT}│{RESET} {CODE}{line.rstrip()}{RESET}"]
    if not stripped:
        return [f"{MARGIN}{ACCENT}│{RESET}"]
    if stripped.startswith("|"):  # tables are printed as they are
        return [BAR + MUTED + stripped + RESET]
    heading = re.match(r"#{1,6}\s+(.*)", stripped)
    if heading:
        return [BAR + BOLD + heading.group(1).replace("**", "") + RESET]
    item = re.match(r"(\s*)([-*•]|\d+[.)])\s+(.*)", line)
    if item:
        indent = " " * (len(item.group(1)) // 2 * 2)
        mark = "•" if item.group(2) in "-*•" else item.group(2)
        first = f"{indent}{ACCENT if mark == '•' else MUTED}{mark}{RESET} "
        rest = " " * (len(indent) + len(mark) + 1)
        rows = textwrap.wrap(item.group(3), max(10, w - len(rest))) or [""]
        return [BAR + (first if i == 0 else rest) + inline(r) for i, r in enumerate(rows)]
    if stripped.startswith(">"):
        return [BAR + MUTED + inline(r) + RESET for r in textwrap.wrap(stripped.lstrip("> "), w)]
    return [BAR + inline(r) for r in textwrap.wrap(stripped, w)]


class Streamer:
    """Support mode: shows 'Alex is thinking', then draws the answer line by line as it streams in."""

    def __init__(self):
        self.seen, self.buffer, self.started = 0, "", False
        self.state = {"code": False}
        self.typing = Typing(label="thinking")
        self.typing.__enter__()

    def __call__(self, text):
        if not self.started:
            self.typing.__exit__(None, None, None)
            alex_label(" · support")
            self.started = True
        self.buffer += text[self.seen:]
        self.seen = len(text)
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.draw(line)

    def draw(self, line):
        if "<think>" in line or "</think>" in line:
            return
        for row in render_line(line, self.state):
            print(row, flush=True)

    def finish(self):
        self.typing.__exit__(None, None, None)
        if self.buffer:
            self.draw(self.buffer)
            self.buffer = ""


# ---------- menus ----------

def menu(title, options, current):
    """A numbered list with the current choice marked. Returns the chosen index."""
    print()
    say(f"{BOLD}{title}{RESET}")
    label_w = max(cells(label) for label, _ in options)
    for i, (label, desc) in enumerate(options):
        mark = f"{ACCENT}●{RESET}" if i == current else f"{FAINT}○{RESET}"
        say(f"  {MUTED}{i + 1}{RESET}  {mark}  {label.ljust(label_w)}   {MUTED}{desc}{RESET}")
    try:
        choice = input(f"{MARGIN}{ACCENT}›{RESET} {MUTED}number, or Enter to keep{RESET}  ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""
    return int(choice) - 1 if choice.isdigit() and 1 <= int(choice) <= len(options) else current


def choose_attitude(current):
    notes = {"Rude": "roasts you, secretly cares", "Sarcastic": "dry and ironic, nice underneath",
             "Chill": "relaxed, always on your side"}
    i = menu("Attitude", [(a, notes[a]) for a in ATTITUDE_NAMES], ATTITUDE_NAMES.index(current))
    return ATTITUDE_NAMES[i]


def choose_model(current_id):
    current = next(i for i, m in enumerate(MODELS) if m[1] == current_id)
    i = menu("Model", [(name, note) for name, _, note in MODELS], current)
    return MODELS[i][0], MODELS[i][1]


def show_help():
    print()
    say(f"{BOLD}Commands{RESET}")
    for cmd, desc in COMMANDS:
        say(f"  {ACCENT}{cmd:<10}{RESET} {MUTED}{desc}{RESET}")


# ---------- main loop ----------

def main():
    os.system("")  # turns on ANSI colours in Windows terminals
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_env()
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("Error: NVIDIA_API_KEY not set. Put it in a .env file next to chatbot.py.")

    name, model_id, _ = MODELS[0]
    attitude, mode = ATTITUDE_NAMES[0], MODES[0]
    messages = [{"role": "system", "content": build_prompt(attitude, mode)}]
    header(name, mode, attitude, clear=True)
    prompt = f"{MARGIN}{ACCENT}›{RESET} "

    while True:
        try:
            print()
            user = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            user = "/exit"
        erase_input(prompt, user)

        cmd = user.lower()
        if not user:
            continue
        if cmd in ("/exit", "/quit"):
            divider(f"{BOT_NAME} went offline")
            print()
            break
        if cmd == "/help":
            show_help()
            continue
        if cmd == "/clear":
            messages = messages[:1]
            header(name, mode, attitude, clear=True)
            divider("new conversation")
            continue
        if cmd == "/model":
            name, model_id = choose_model(model_id)
            divider(f"now using {name}")
            continue
        if cmd == "/attitude":
            attitude = choose_attitude(attitude)
            messages[0] = {"role": "system", "content": build_prompt(attitude, mode)}
            divider(f"{BOT_NAME} is now {attitude.lower()}" + (" (applies in chat mode)" if mode == "Support" else ""))
            continue
        if cmd == "/mode":
            mode = "Support" if mode == "Chat" else "Chat"
            messages[0] = {"role": "system", "content": build_prompt(attitude, mode)}
            divider(f"support mode · {BOT_NAME} will actually help" if mode == "Support"
                    else f"chat mode · {BOT_NAME} is {attitude.lower()} again", ACCENT)
            continue

        show_mine(user)
        messages.append({"role": "user", "content": user})
        support = mode == "Support"
        streamer = Streamer() if support else None
        try:
            result = ask(api_key, messages, fallback_order(model_id), mode=mode,
                         typing=None if support else human_typing, on_delta=streamer,
                         on_fail=lambda n, why, secs: say(f"{WARN}{n}: {why} after {secs:.0f}s, "
                                                          f"trying the next model…{RESET}"))
        except KeyboardInterrupt:
            result = None
            print()
            grey("interrupted")
        if streamer:
            streamer.finish()

        if result:
            if result["model_id"] != model_id:  # stick with the model that works
                name, model_id = result["name"], result["model_id"]
                divider(f"switched to {name} for this chat")
            if support:  # already drawn while streaming
                messages.append({"role": "assistant", "content": clean_support(result["reply"])})
            else:
                bubbles = humanise(result["reply"])
                messages.append({"role": "assistant", "content": "\n".join(bubbles)})
                send_bubbles(bubbles)
            grey(explain_timing(result["stats"], result["lost"], result["lost_reason"]))
        else:
            messages.pop()  # drop the unanswered message so history stays clean
            say(f"{ERR}✕ Couldn't reach NVIDIA's servers. They may be busy, try again in a bit.{RESET}")


if __name__ == "__main__":
    main()
