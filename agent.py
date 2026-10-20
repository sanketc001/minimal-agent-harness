"""CCHarness - a minimal illustrative LLM agent harness.

An agent is just a loop around an LLM call with a few tools. This file
contains the entire loop plus every built-in tool, all in plain Python.

Memory is an Obsidian vault: .md files in ./Memory with [[wiki-links]]
between them. The agent's current-session context is a file named
`Memory/{epoch}.md`, created at startup and appended to as work happens.
Future sessions can read it back.

Run:
    python agent.py                 # REPL
    python agent.py "do a thing"    # one-shot
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import browser
import mcp_loader


load_dotenv()
CLIENT = OpenAI(base_url=os.environ.get("OPENAI_API_BASE", "http://localhost:1234/v1"),
                api_key=os.environ.get("OPENAI_API_KEY", ""))
MODEL = os.environ.get("OPENAI_MODEL", "google/gemma-4-e4b")

MEMORY_DIR = Path("Memory")
MEMORY_DIR.mkdir(exist_ok=True)

SESSION_ID = int(time.time())
SESSION_FILE = MEMORY_DIR / f"{SESSION_ID}.md"


# ---------------------------------------------------------------------------
# Session bootstrap
# ---------------------------------------------------------------------------

def _init_session() -> None:
    SESSION_FILE.write_text(
        f"---\n"
        f"type: session\n"
        f"session_id: {SESSION_ID}\n"
        f"started: {datetime.now().isoformat(timespec='seconds')}\n"
        f"---\n\n"
        f"# Session {SESSION_ID}\n\n",
        encoding="utf-8",
    )


SYSTEM_PROMPT = f"""You are CCHarness, a general-purpose agent running in a Python harness.

━━ TOOL-USE RULES (non-negotiable) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. NEVER describe, simulate, or pretend to perform an action.
   If the user asks you to run a command, open a URL, write a file, or search,
   you MUST call the appropriate tool. Saying "I ran ls" without calling `shell`
   is a hallucination and is forbidden.
2. Use as many tool calls as needed. Chain them: run a command, read the output,
   act on it. Don't stop and ask if you have enough information to proceed.
3. Always respond AFTER the tool results are back, never before.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Available capability groups:
  shell            — run any CLI command
  browser_*        — raw Chrome DevTools Protocol (see BROWSER GUIDE below)
  memory_*         — Obsidian vault in ./Memory/ (.md files with [[wiki-links]])
  context_append   — append notes to this session's scratchpad Memory/{SESSION_ID}.md
  sub_agent        — isolated LLM call for deep research (returns summary + .md artifact)
  mcp__*           — any tools from MCP servers configured in mcp.json

━━ BROWSER AUTOMATION GUIDE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Typical workflow for any browser task:
  1. browser_start                 — launch/attach to Chrome (idempotent; safe to call)
  2. browser_goto(url)             — navigate; returns the final URL
  3. browser_snapshot() / browser_elements()
                                    — SEE the page before acting. Snapshot gives you
                                      a JPEG (vision); elements gives a numbered
                                      list of every clickable/fillable thing.
  4. browser_click_text / browser_fill_by_label / browser_click / browser_fill
                                    — act.
  5. browser_wait_for(text)        — pause until async content appears (search results,
                                      modals, SPAs loading data).
  6. browser_snapshot() again      — VERIFY the action did what you expected.

Selector rules:
  ✅ Valid CSS:    `button.submit`, `#search`, `input[name="q"]`, `a[href*="login"]`
  ❌ NOT valid:    `button:contains("Submit")` — that's jQuery, it WILL fail.
                   Use `browser_click_text("Submit")` instead.

Choosing the right click/fill tool:
  • Know the CSS selector?                → browser_click / browser_fill
  • Only know the visible text on a button/link?
                                          → browser_click_text
  • Form field you can identify by its label/placeholder?
                                          → browser_fill_by_label
  • Dropdown (<select>)?                  → browser_select
  • Dropdowns built from <div>s?          → browser_click_text on the option

Other useful moves:
  • browser_key("Enter")           — submit a form after filling a field
  • browser_key("Escape")          — close a modal or dropdown
  • browser_scroll("down")         — reveal more content
  • browser_url() / browser_status()
                                    — check where you actually are
  • browser_eval(js)               — escape hatch when nothing else fits
  • browser_back()                 — history.back()

Recovery:
  • If click/fill returns "NOT FOUND", call browser_elements or browser_snapshot
    first, then retry with the right text/selector. Never guess twice.
  • If a page looks wrong after navigation, browser_status tells you what
    target CDP is attached to.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Memory conventions:
- Your session scratchpad is Memory/{SESSION_ID}.md — use context_append for findings/decisions.
- Durable knowledge goes in named .md files (e.g., notes_on_X.md) with [[wiki-links]].
- Use sub_agent for deep multi-step research to keep your own context clean.
"""


# ---------------------------------------------------------------------------
# Special tool return type for images.
# When a tool returns _ImageToolResult, _run_loop sends the image to the LLM
# as a multimodal content array (text + image_url) so vision-capable models
# can actually see the screenshot.
# ---------------------------------------------------------------------------

class _ImageToolResult:
    """Wraps a screenshot so the dispatch loop injects it as image content."""
    __slots__ = ("text", "data_url", "filepath")

    def __init__(self, text: str, data_url: str, filepath: str):
        self.text = text
        self.data_url = data_url
        self.filepath = filepath

    def __str__(self) -> str:
        return self.text


# ---------------------------------------------------------------------------
# Built-in tools. Each entry is (python function, openai tool schema).
# ---------------------------------------------------------------------------

def _tool(schema: dict):
    """Decorator: attach an OpenAI tool schema to a python function."""
    def wrap(fn):
        fn.schema = {"type": "function", "function": schema}
        return fn
    return wrap


@_tool({
    "name": "shell",
    "description": "Run a shell command and return stdout, stderr, and exit code.",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
})
def shell(command: str) -> str:
    try:
        r = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=120
        )
        out = (r.stdout or "")[-6000:]
        err = (r.stderr or "")[-2000:]
        return f"exit={r.returncode}\n--- stdout ---\n{out}\n--- stderr ---\n{err}"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 120s"


@_tool({
    "name": "browser_start",
    "description": "Launch Chrome with the DevTools Protocol and connect to it.",
    "parameters": {"type": "object", "properties": {}},
})
def browser_start() -> str:
    return browser.start()


@_tool({
    "name": "browser_goto",
    "description": "Navigate the current browser tab to a URL.",
    "parameters": {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    },
})
def browser_goto(url: str) -> str:
    return browser.goto(url)


@_tool({
    "name": "browser_read_text",
    "description": "Return the visible innerText of the current page (first 8000 chars).",
    "parameters": {"type": "object", "properties": {}},
})
def browser_read_text() -> str:
    return browser.read_text()


@_tool({
    "name": "browser_click",
    "description": "Click the first element matching a CSS selector.",
    "parameters": {
        "type": "object",
        "properties": {"selector": {"type": "string"}},
        "required": ["selector"],
    },
})
def browser_click(selector: str) -> str:
    return browser.click(selector)


@_tool({
    "name": "browser_fill",
    "description": "Set the value of an input/textarea matched by selector and fire input/change events.",
    "parameters": {
        "type": "object",
        "properties": {"selector": {"type": "string"}, "value": {"type": "string"}},
        "required": ["selector", "value"],
    },
})
def browser_fill(selector: str, value: str) -> str:
    return browser.fill(selector, value)


@_tool({
    "name": "browser_eval",
    "description": "Evaluate a JavaScript expression in the page and return the value. Escape hatch.",
    "parameters": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
})
def browser_eval(expression: str) -> str:
    return json.dumps(browser.eval_js(expression), default=str)


@_tool({
    "name": "browser_status",
    "description": "Diagnostic: list all CDP page targets and report the currently attached URL/readyState/title. Useful when goto seems to have no effect.",
    "parameters": {"type": "object", "properties": {}},
})
def browser_status() -> str:
    return browser.status()


@_tool({
    "name": "browser_snapshot",
    "description": (
        "Capture a JPEG screenshot of the current browser tab and return it as an image "
        "so you can visually inspect the page. The screenshot is also saved to Memory/ "
        "as a .jpg artifact. Use this to verify that a page loaded correctly, read visual "
        "content, or understand page layout before clicking/filling."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "quality": {
                "type": "integer",
                "description": "JPEG quality 1-100 (default 80). Lower = smaller, faster.",
            }
        },
    },
})
def browser_snapshot(quality: int = 80) -> _ImageToolResult:
    b64, filepath = browser.screenshot(quality=quality)
    return _ImageToolResult(
        text=f"Screenshot saved to {filepath}",
        data_url=f"data:image/jpeg;base64,{b64}",
        filepath=filepath,
    )


@_tool({
    "name": "browser_click_text",
    "description": (
        "Click the first interactive element whose visible text matches. "
        "Use this instead of CSS selectors like `button:contains(\"Submit\")` — "
        "that's jQuery, not valid CSS, and will fail. Tries exact match first, "
        "then case-insensitive, substring, then aria-label."
    ),
    "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Visible text (or aria-label) of the element to click."}},
        "required": ["text"],
    },
})
def browser_click_text(text: str) -> str:
    return browser.click_text(text)


@_tool({
    "name": "browser_elements",
    "description": (
        "List every visible interactive element on the current page (buttons, links, "
        "inputs, role=button, etc.) with an index, tag, id, href and visible text. "
        "Call this when you need to know what's clickable before acting — much more "
        "reliable than guessing a selector."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Max number of elements to return (default 60)."},
        },
    },
})
def browser_elements(limit: int = 60) -> str:
    return browser.elements(limit=limit)


@_tool({
    "name": "browser_fill_by_label",
    "description": (
        "Fill the input/textarea/select whose associated <label>, aria-label, placeholder "
        "or name matches. Use this when you can see the form label but don't know the CSS selector."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["label", "value"],
    },
})
def browser_fill_by_label(label: str, value: str) -> str:
    return browser.fill_by_label(label, value)


@_tool({
    "name": "browser_key",
    "description": (
        "Press a named key. Useful for submitting forms (Enter), closing dialogs "
        "(Escape), tabbing between fields (Tab), or moving the caret. Known: "
        "Enter, Tab, Escape, Backspace, Delete, ArrowUp/Down/Left/Right, "
        "Home, End, PageUp, PageDown, Space."
    ),
    "parameters": {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
})
def browser_key(key: str) -> str:
    return browser.press_key(key)


@_tool({
    "name": "browser_wait_for",
    "description": (
        "Block until the given text appears in the page body, or time out. "
        "Use after an action that kicks off async content loading."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "timeout": {"type": "number", "description": "Seconds (default 10)."},
        },
        "required": ["text"],
    },
})
def browser_wait_for(text: str, timeout: float = 10.0) -> str:
    return browser.wait_for(text, timeout=timeout)


@_tool({
    "name": "browser_scroll",
    "description": "Scroll the page. direction: 'up', 'down', 'top', 'bottom'.",
    "parameters": {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["up", "down", "top", "bottom"]},
            "amount": {"type": "integer", "description": "Pixels for up/down (default 600)."},
        },
    },
})
def browser_scroll(direction: str = "down", amount: int = 600) -> str:
    return browser.scroll(direction=direction, amount=amount)


@_tool({
    "name": "browser_back",
    "description": "Navigate the current tab back one step in history.",
    "parameters": {"type": "object", "properties": {}},
})
def browser_back() -> str:
    return browser.go_back()


@_tool({
    "name": "browser_url",
    "description": "Return the current tab's URL (location.href).",
    "parameters": {"type": "object", "properties": {}},
})
def browser_url() -> str:
    return browser.current_url()


@_tool({
    "name": "browser_select",
    "description": "Pick an option inside a <select> element. Matches by value, then by visible text.",
    "parameters": {
        "type": "object",
        "properties": {
            "selector": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["selector", "value"],
    },
})
def browser_select(selector: str, value: str) -> str:
    return browser.select_option(selector, value)


# ---- Memory (Obsidian vault) -------------------------------------------------

_NAME_OK = re.compile(r"[^A-Za-z0-9_\-\. ]+")


def _mem_path(name: str) -> Path:
    name = _NAME_OK.sub("_", name).strip() or "untitled"
    if not name.endswith(".md"):
        name += ".md"
    return MEMORY_DIR / name


@_tool({
    "name": "memory_write",
    "description": (
        "Write or overwrite a markdown file in the Obsidian vault at ./Memory/. "
        "If the content has no YAML frontmatter, one is auto-prepended. Use "
        "[[wiki-links]] to reference other notes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "File name, with or without .md"},
            "content": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "content"],
    },
})
def memory_write(name: str, content: str, tags: list | None = None) -> str:
    path = _mem_path(name)
    if not content.lstrip().startswith("---"):
        fm = [
            "---",
            f"created: {datetime.now().isoformat(timespec='seconds')}",
        ]
        if tags:
            fm.append("tags: [" + ", ".join(tags) + "]")
        fm.append("---\n")
        content = "\n".join(fm) + "\n" + content
    path.write_text(content, encoding="utf-8")
    return f"wrote {path.as_posix()} ({len(content)} bytes)"


@_tool({
    "name": "memory_read",
    "description": "Read a markdown file from ./Memory/.",
    "parameters": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
})
def memory_read(name: str) -> str:
    path = _mem_path(name)
    if not path.exists():
        return f"not found: {path.as_posix()}"
    return path.read_text(encoding="utf-8")[:12000]


@_tool({
    "name": "memory_list",
    "description": "List all markdown files in ./Memory/ with their first heading as title.",
    "parameters": {"type": "object", "properties": {}},
})
def memory_list() -> str:
    rows = []
    for p in sorted(MEMORY_DIR.glob("*.md")):
        title = ""
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        rows.append(f"- {p.name} — {title}" if title else f"- {p.name}")
    return "\n".join(rows) or "(empty vault)"


@_tool({
    "name": "memory_search",
    "description": "Case-insensitive substring search across all markdown files in ./Memory/. Returns file:line snippets.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
})
def memory_search(query: str) -> str:
    q = query.lower()
    hits: list[str] = []
    for p in sorted(MEMORY_DIR.glob("*.md")):
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if q in line.lower():
                hits.append(f"{p.name}:{i}: {line.strip()[:200]}")
                if len(hits) >= 50:
                    return "\n".join(hits) + "\n... (truncated)"
    return "\n".join(hits) or f"no matches for {query!r}"


@_tool({
    "name": "context_append",
    "description": "Append markdown to the current session's scratchpad (Memory/{session}.md). Use for decisions, findings, links to artifacts.",
    "parameters": {
        "type": "object",
        "properties": {"markdown": {"type": "string"}},
        "required": ["markdown"],
    },
})
def context_append(markdown: str) -> str:
    with SESSION_FILE.open("a", encoding="utf-8") as f:
        f.write(f"\n{markdown}\n")
    return f"appended {len(markdown)} bytes to {SESSION_FILE.as_posix()}"


# ---- Sub-agent ---------------------------------------------------------------

@_tool({
    "name": "sub_agent",
    "description": (
        "Fire an isolated LLM call with its own tool loop to do research or "
        "planning without polluting your context. Writes a detailed markdown "
        "artifact into ./Memory/ and returns a short summary plus a wiki-link "
        "to the artifact. Good for deep web research, multi-step exploration, "
        "or any subtask whose intermediate tool outputs you don't want in your "
        "own context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "What the sub-agent should accomplish."},
            "context_hint": {"type": "string", "description": "Optional prior context/constraints."},
        },
        "required": ["task"],
    },
})
def sub_agent(task: str, context_hint: str = "") -> str:
    sub_id = int(time.time() * 1000)
    artifact_name = f"research_{sub_id}"
    artifact_path = _mem_path(artifact_name)

    sub_system = (
        "You are a sub-agent fired by the main agent for an isolated task. "
        "Do the work using your tools. When done, reply with a concise "
        "summary (1-3 short paragraphs) for the caller. The harness will "
        "automatically save the full transcript as a markdown artifact, so "
        "you do NOT need to write it yourself."
    )
    messages: list = [
        {"role": "system", "content": sub_system},
        {"role": "user", "content": f"Task: {task}\n\nContext: {context_hint}".strip()},
    ]

    # Sub-agent tools: everything except sub_agent itself (prevent recursion).
    sub_tools = [t for t in ALL_TOOLS if t["name"] != "sub_agent"]
    transcript: list[str] = [f"# Sub-agent research: {task}", ""]

    summary = _run_loop(messages, sub_tools, transcript=transcript, max_turns=12)

    artifact = (
        f"---\n"
        f"type: research\n"
        f"parent_session: {SESSION_ID}\n"
        f"created: {datetime.now().isoformat(timespec='seconds')}\n"
        f"---\n\n"
        f"# {task}\n\n"
        f"## Summary\n\n{summary}\n\n"
        f"## Transcript\n\n" + "\n".join(transcript)
    )
    artifact_path.write_text(artifact, encoding="utf-8")

    return f"Summary:\n{summary}\n\nFull artifact: [[{artifact_name}]] ({artifact_path.as_posix()})"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

BUILTIN_FNS = [
    shell,
    browser_start, browser_goto, browser_read_text,
    browser_click, browser_fill, browser_eval, browser_status, browser_snapshot,
    browser_click_text, browser_elements, browser_fill_by_label,
    browser_key, browser_wait_for, browser_scroll, browser_back,
    browser_url, browser_select,
    memory_write, memory_read, memory_list, memory_search, context_append,
    sub_agent,
]

ALL_TOOLS: list[dict] = [
    {"name": fn.schema["function"]["name"], "schema": fn.schema, "fn": fn}
    for fn in BUILTIN_FNS
]


# ---------------------------------------------------------------------------
# The loop. This is the entire agent.
# ---------------------------------------------------------------------------

_W = 62  # terminal display width


def _run_loop(
    messages: list,
    tools: list[dict],
    transcript: list[str] | None = None,
    max_turns: int = 30,
    log: list[str] | None = None,
) -> str:
    """Core agent loop.

    Calls the LLM, executes every tool call the model requests (feeding each
    result back as a `tool` message), then loops until the model replies with
    text rather than another tool call.

    `transcript` collects markdown for sub-agent artifacts.
    `log`        collects markdown for the main session file.
    """
    by_name = {t["name"]: t for t in tools}
    schemas = [t["schema"] for t in tools]

    for _ in range(max_turns):
        resp = CLIENT.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=schemas,
            tool_choice="auto",      # explicit: model may call tools or reply in text
        )
        choice = resp.choices[0]
        msg = choice.message
        finish = choice.finish_reason

        # ── debug: show what the model actually decided ────────────────────
        tc_count = len(msg.tool_calls) if msg.tool_calls else 0
        print(f"\n[llm] finish={finish}  tool_calls={tc_count}  model={resp.model}")

        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            text = msg.content or ""
            if transcript is not None and text:
                transcript.append(f"**Assistant:** {text}")
            if log is not None:
                log.append(f"\n**Response:** {text}\n")
            return text

        pending_images: list[_ImageToolResult] = []  # reset each LLM turn

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            args_str = json.dumps(args, indent=2)

            # ── terminal: tool call header ─────────────────────────────────
            print(f"\n{'═' * _W}")
            print(f"  TOOL CALL  ▶  {name}")
            print(f"{'─' * _W}")
            for line in args_str.splitlines()[:12]:
                print(f"  {line}")
            if len(args_str.splitlines()) > 12:
                print("  ...")

            # ── execute locally (the LLM cannot run tools itself) ──────────
            try:
                raw_result = (
                    by_name[name]["fn"](**args)
                    if name in by_name
                    else f"ERROR: unknown tool '{name}'"
                )
            except Exception as e:
                raw_result = f"ERROR: {type(e).__name__}: {e}"

            # Separate display text from the actual content sent to the LLM.
            # _ImageToolResult carries both a text summary and a base64 image.
            is_image = isinstance(raw_result, _ImageToolResult)
            result_text = str(raw_result)  # always a plain string for display/logs

            # ── terminal: result ───────────────────────────────────────────
            print(f"{'─' * _W}")
            print(f"  RESULT{' (image)' if is_image else ''}")
            print(f"{'─' * _W}")
            result_lines = result_text.splitlines()
            for line in result_lines[:15]:
                print(f"  {line}")
            if len(result_lines) > 15:
                print(f"  … ({len(result_lines) - 15} more lines)")
            print(f"{'═' * _W}")

            # ── collect for sub-agent artifact ─────────────────────────────
            if transcript is not None:
                transcript.append(
                    f"\n**Tool:** `{name}`\n"
                    f"```json\n{args_str[:600]}\n```\n"
                    f"**Result:**\n```\n{result_text[:600]}\n```"
                )

            # ── collect for session file ───────────────────────────────────
            if log is not None:
                log.append(
                    f"\n**`{name}`** `{json.dumps(args)[:300]}`\n"
                    f"\n```\n{result_text[:600]}\n```\n"
                )

            # ── feed result back to the LLM ────────────────────────────────
            # Tool messages must be plain text — no API accepts image arrays in
            # the "tool" role. Images are injected as a user message below,
            # after ALL tool results for this turn have been queued.
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text[:16000],
            })

            if is_image:
                pending_images.append(raw_result)

        # ── inject screenshots as a user message so vision models can see them ──
        # This runs once per LLM turn, after every tool result has been added.
        # A single user message carries all images captured in this turn.
        if pending_images:
            vision_content: list = []
            for img in pending_images:
                vision_content.append({
                    "type": "text",
                    "text": f"[browser screenshot saved at {img.filepath}]",
                })
                vision_content.append({
                    "type": "image_url",
                    "image_url": {"url": img.data_url},
                })
            messages.append({"role": "user", "content": vision_content})

    return "(reached max_turns without a final text response)"


def run_turn(messages: list) -> str:
    """Run one full user turn and auto-log everything to the session file."""
    user_msg = next(
        (m["content"] for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"),
        "",
    )
    log: list[str] = [
        f"\n## {datetime.now().isoformat(timespec='seconds')}\n",
        f"**User:** {user_msg}\n",
    ]
    text = _run_loop(messages, ALL_TOOLS, log=log)
    with SESSION_FILE.open("a", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    return text


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session()

    mcp_tools, mcp_servers = mcp_loader.load("mcp.json")
    ALL_TOOLS.extend(mcp_tools)

    tool_names = [t["name"] for t in ALL_TOOLS]
    print(f"CCHarness  session={SESSION_ID}  model={MODEL}")
    print(f"  endpoint : {os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1')}")
    print(f"  session  : {SESSION_FILE}")
    print(f"  tools    : {len(ALL_TOOLS)} total  ({len(BUILTIN_FNS)} builtin + {len(mcp_tools)} mcp)")
    for name in tool_names:
        print(f"             · {name}")
    print()

    messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        argv = sys.argv[1:]
        if argv:
            prompt = " ".join(argv)
            messages.append({"role": "user", "content": prompt})
            print(run_turn(messages))
            return

        while True:
            try:
                user = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user:
                continue
            if user in {"/exit", "/quit"}:
                break
            messages.append({"role": "user", "content": user})
            print(run_turn(messages))
            print()
    finally:
        for s in mcp_servers:
            s.stop()
        browser.stop()


if __name__ == "__main__":
    main()
