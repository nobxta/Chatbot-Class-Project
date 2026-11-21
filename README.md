# Alex: a CLI chatbot on NVIDIA models

Alex is a chatbot you talk to in the terminal. It runs on large language models hosted by NVIDIA
(build.nvidia.com) and has two modes:

- **Chat mode**: Alex texts like a friend, with an attitude you choose (Rude, Sarcastic or Chill).
- **Support mode**: Alex actually helps with homework, code, plans or a rough day, with full,
  formatted answers.

Everything is in one file, `chatbot.py`, and it uses **only the Python standard library**, so there
is nothing to install.

---

## 1. Run it

1. Install Python 3.10 or newer.
2. Put your NVIDIA key in a file named `.env` next to `chatbot.py`:
   ```
   NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxx
   ```
3. Start it:
   ```
   python chatbot.py
   ```

### Commands

| Command | What it does |
|---|---|
| `/mode` | Switch between Chat mode and Support mode |
| `/attitude` | Pick Rude, Sarcastic or Chill (Chat mode) |
| `/model` | Pick a different AI model |
| `/clear` | Start a fresh conversation |
| `/help` | Show the commands |
| `/exit` | Quit |

### What it looks like

```
  Alex  ● online                                  chat · rude · Nemotron 3.5 Super
  ────────────────────────────────────────────────────────────────────────────────
  /mode  /attitude  /model  /clear  /help  /exit

                                                ugh my exam got moved to tomorrow
                                                                          3:02 pm
  Alex · 3:02 pm
   skill issue
   you had all week lmao
   go study instead of texting me
  2.1s  ·  wait 0.5s  ·  thinking 1.5s · 180 tok  ·  writing 0.1s   (mostly thinking)

  ─────────────────── support mode · Alex will actually help ───────────────────

                                              how do i reverse a string in python?
                                                                          7:41 am
  Alex · 7:41 am · support
  │ Use slicing:
  │ ┌ python
  │ │ s = "hello"
  │ │ print(s[::-1])  # olleh
  │ └
  1.6s  ·  wait 0.3s  ·  thinking 0.9s · 138 tok  ·  writing 0.4s   (fast reply)
```

In the terminal your messages are blue bubbles on the right and Alex's are grey bubbles on the left.

---

## 2. How a message travels (the big picture)

```
 you type a message
        │
        ▼
 main()  ── is it a /command? ──yes──► change mode / attitude / model, loop again
        │ no
        ▼
 add it to `messages` (the whole conversation so far)
        │
        ▼
 ask()  ── tries each model in order ──► get_reply() ── HTTPS ──► NVIDIA API
        │                                     ▲                       │
        │                                     └──── streamed reply ◄──┘
        ▼
 Chat mode:    humanise() → send_bubbles()   (short texts with "typing…")
 Support mode: Streamer prints the answer live as it arrives
        │
        ▼
 explain_timing()  → shows how long it took and why
```

---

## 3. The file, section by section

`chatbot.py` has three parts: **settings**, **shared logic** (talking to the AI) and the **CLI**
(the terminal interface). Line numbers are approximate.

### Part A: Settings (lines 1–116)

| Name | What it is | Why it's there |
|---|---|---|
| `API_URL` | NVIDIA's chat endpoint | All models share one URL. The model is chosen by name in the request. |
| `TIMEOUT = 30` | Seconds to wait for a model | Free servers are sometimes overloaded. After 30 s we give up on that model and try the next. |
| `BOT_NAME = "Alex"` | The bot's name | Used in the prompt and on screen, so you can rename the bot in one place. |
| `MODELS` | 4 models: name, id, note | The first one (Nemotron 3.5 Super) is the default because it was fastest in testing, about 2 s per reply. |
| `ATTITUDES` | Rude / Sarcastic / Chill text | Each attitude is a block of instructions that changes Alex's personality. |
| `SUPPORT_PROMPT` | Instructions for Support mode | Tells the model to be accurate, explain step by step, use formatting and handle sensitive topics safely. |
| `build_prompt()` | Builds the **system prompt** | Combines the attitude or support instructions with the texting rules and safety limits. |

**What's a system prompt?** It's the first, hidden message in the conversation. The model reads it
before anything you type, and it decides the bot's personality and rules. Changing just this text
is how one model behaves like three different people.

### Part B: Shared logic (lines 119–290)

| Function | What it does | Why it works this way |
|---|---|---|
| `load_env()` | Reads `NVIDIA_API_KEY` from `.env` | Keeps the secret key out of the code. A tiny version written by hand, so no extra library is needed. |
| `now()` | Current time like `3:05 pm` | Timestamps on each message, like a real chat app. |
| `get_reply()` | **Sends the conversation to one model and reads the answer** | The core of the app. Explained in detail below. |
| `fallback_order()` | Puts the chosen model first, then the rest | So there's always a backup model. |
| `ask()` | Tries each model until one answers | If a model times out or errors, it moves on and remembers how much time was lost. |
| `timing_reason()` / `explain_timing()` | Turns the timing numbers into a sentence | Answers "why did that take so long?" (busy server, long thinking, or a slow model). |
| `humanise()` | Cleans a Chat-mode reply | Removes AI-looking formatting (bullets, bold, long dashes) and splits the reply into separate texts. |
| `clean_support()` | Cleans a Support-mode reply | Keeps the formatting (lists, code), only removes leaked "thinking" text. |

#### Inside `get_reply()` (the most important function)

1. **Builds the request**: a JSON body with the model name, the whole conversation (`messages`), and
   settings:
   - `temperature`: 0.9 in Chat mode (more creative) and 0.5 in Support mode (more accurate)
   - `max_tokens`: 2048 in Chat mode, 4096 in Support mode
   - `stream: True`: send the answer piece by piece
2. **Sends it** with `urllib.request` and the header `Authorization: Bearer <key>`.
3. **Reads the stream.** The server sends lines that start with `data:` (Server-Sent Events). Each
   line holds a small piece of the answer. A piece is either:
   - `reasoning_content`: the model thinking privately (we time it but don't show it), or
   - `content`: the actual reply (we collect it).
4. **Records the time** of the first byte, the first thinking token and the first reply token.
   From those it works out *server wait*, *thinking* and *writing* time.
5. **Returns** the full text and the timing stats.

**Why send the whole conversation every time?** The API doesn't remember anything between
requests. Sending the history is what lets Alex remember your name or what you said earlier.

### Part C: The CLI (lines 293–472)

| Name | What it does | Why |
|---|---|---|
| Colours (`MUTED`, `ACCENT`, `ME`, `THEM`…) | 24-bit ANSI colours from one small palette | One blue accent, greys for secondary text. Bubbles set both text and background colour, so they read on light and dark terminals. `NO_COLOR=1` turns colours off. |
| `width()` / `cells()` | Terminal width (capped at 92) and the on-screen width of text | Keeps lines readable on wide screens. `cells()` counts emoji as 2 columns, so bubbles line up. |
| `header()` | Top bar: name, online dot, mode, attitude, model, then the commands | Everything about the current state in one line. Redrawn on `/clear`. |
| `bubble_rows()` | Wraps text and pads every row to the same width | The coloured background then forms a neat block, like a real chat bubble. |
| `erase_input()` + `show_mine()` | Removes the line you typed and redraws it as a blue bubble on the right | Your messages on the right and Alex's on the left, like a messaging app. |
| `divider()` | Centred notices like `──── Alex is now chill ────` | Mode, attitude and model changes stand out without cluttering the chat. |
| `Typing` (class) | Shows an animated `Alex is typing ···` line | Runs in a separate **thread** so the dots keep moving while the main thread waits for the network. |
| `human_typing()` | Typing indicator that lasts at least ~1 s | Real people don't reply instantly, so very fast replies get a short pause. |
| `send_bubbles()` | Prints Alex's grey bubbles on the left, with a small pause between them | Feels like someone sending several texts. |
| `render_line()` / `inline()` | Draws Markdown for the terminal: bold, bullets, headings, code blocks | Support answers look formatted instead of showing raw `**` and backticks. |
| `Streamer` (class) | Prints a Support answer **line by line as it streams in**, behind a blue bar | Long answers start appearing straight away instead of after 10+ seconds. |
| `menu()`, `choose_model()`, `choose_attitude()` | Numbered menus with the current choice marked `●` | Press a number, or Enter to keep the current one. |
| `main()` | **The main loop** | Loads the key, then repeats: read input, handle commands, ask the AI, show the reply. |

#### Inside `main()`

```
load the key  →  messages = [system prompt]
loop forever:
    read what you typed
    /exit      → quit
    /clear     → keep only the system prompt
    /model     → choose a model
    /attitude  → rebuild the system prompt with the new attitude
    /mode      → switch Chat ↔ Support (rebuilds the system prompt)
    otherwise:
        add your message to `messages`
        ask() the models
        if a backup model answered → keep using it for the rest of the chat
        show the reply (bubbles or streamed) and the timing line
        add the reply to `messages` so Alex remembers it
        if every model failed → remove your message and say so
```

---

## 4. The models

| Model | NVIDIA id | Notes from testing |
|---|---|---|
| Nemotron 3.5 Super (default) | `nvidia/nemotron-3.5-super-vl-120b-a12b` | Fastest: about 2 s |
| Kimi K3.1 Pro | `moonshotai/kimi-k3.1-pro` | Best texting voice, but often overloaded |
| Mistral Large 4 | `mistralai/mistral-large-4-1t-a52b` | About 14 s |
| GLM-5.3 Flash | `z-ai/glm-5.3-flash` | Often slow on the free tier |

DeepSeek V4.1 Flash and Kimi K3 were removed because NVIDIA retired them.

**Why are replies a few seconds slow?** These are *reasoning* models: they "think" (write hidden
notes) before answering. The timing line shows this. In testing, thinking was usually most of the
time, not the network.

---

## 5. Key ideas to remember

- **API**: we don't run the AI ourselves. We send text to NVIDIA's servers over HTTPS and get text back.
- **System prompt**: hidden instructions that set the personality. Different prompt, different Alex.
- **Conversation history**: the full message list is sent every time, which is how the bot "remembers".
- **Streaming**: the answer arrives in small pieces, so we can show it live and time each stage.
- **Fallback**: if one model is down or slow, the next one answers automatically.
- **Threads**: the typing animation runs alongside the network request without freezing.
- **No dependencies**: only Python's standard library (`json`, `urllib`, `threading`, `re`, ...).

---

## 6. Safety and limits

- The API key lives in `.env`, never in the code. Don't share `.env` or upload it.
- Every attitude has limits: no slurs or hate, no threats, nothing sexual. If someone seems to be
  in danger, Alex drops the act and is kind.
- Support mode suggests a professional for serious medical, legal or money decisions, and points
  to a crisis line or emergency services if someone mentions self-harm.
- The free NVIDIA tier can be slow or busy, which is why the timeout and fallback exist.
