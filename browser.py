"""Raw Chrome DevTools Protocol client with a tiny helper layer.

An agent is just a loop around an LLM call with a few tools. This file is
one of those tools: it launches Chrome with the DevTools Protocol enabled,
opens a websocket to the first tab, and exposes a handful of synchronous
helpers (goto, read_text, click, fill, eval_js).

No Playwright, no Selenium. ~130 lines of raw protocol so you can see
exactly how browser automation works.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from typing import Any

import requests
from websockets.sync.client import connect as ws_connect


CDP_PORT = int(os.environ.get("CDP_PORT", 9222))
_chrome_proc: subprocess.Popen | None = None
_ws = None
_msg_id = 0


def _find_chrome() -> str:
    env = os.environ.get("CHROME_PATH")
    if env and os.path.exists(env):
        return env
    system = platform.system()
    if system == "Windows":
        candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe"),
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]
    elif system == "Darwin":
        candidates = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
    else:
        candidates = [
            shutil.which("google-chrome") or "",
            shutil.which("chromium") or "",
            shutil.which("chromium-browser") or "",
        ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise RuntimeError(
        "Chrome not found. Set CHROME_PATH env var or install Chrome/Edge."
    )


def _list_page_targets() -> list[dict]:
    """Return only 'page' type CDP targets — no workers, extensions, or browser."""
    try:
        targets = requests.get(f"http://localhost:{CDP_PORT}/json", timeout=2).json()
    except Exception:
        return []
    return [t for t in targets if t.get("type") == "page"]


def _pick_target(pages: list[dict]) -> dict | None:
    """Prefer about:blank, then any non-chrome:// URL, else the first page."""
    if not pages:
        return None
    for p in pages:
        if p.get("url") == "about:blank":
            return p
    for p in pages:
        url = p.get("url", "")
        if not url.startswith(("chrome://", "edge://", "devtools://")):
            return p
    return pages[0]


def _connect_to_port() -> bool:
    """Try to connect _ws to a 'page' target on the CDP port. Returns True on success."""
    global _ws
    target = _pick_target(_list_page_targets())
    if target is None:
        return False
    try:
        ws_url = target["webSocketDebuggerUrl"]
        _ws = ws_connect(ws_url, max_size=32 * 1024 * 1024)
        # Activate the tab so the visible window matches our CDP session,
        # then enable the domains we'll use.
        try:
            requests.get(
                f"http://localhost:{CDP_PORT}/json/activate/{target['id']}",
                timeout=2,
            )
        except Exception:
            pass
        send("Page.enable")
        send("Runtime.enable")
        print(f"[browser] attached to target id={target['id'][:8]}… url={target.get('url','?')}", flush=True)
        return True
    except Exception as e:
        print(f"[browser] connect failed: {e}", flush=True)
        _ws = None
        return False


def start() -> str:
    """Connect to Chrome (reuse existing CDP endpoint if available, else launch fresh)."""
    global _chrome_proc, _ws
    if _ws is not None:
        return "already connected"

    # Reuse an existing Chrome with CDP open on the port.
    if _connect_to_port():
        return f"connected to existing Chrome on port {CDP_PORT}"

    # Launch a fresh isolated Chrome instance.
    chrome = _find_chrome()
    profile = tempfile.mkdtemp(prefix="ccharness-chrome-")
    _chrome_proc = subprocess.Popen(
        [
            chrome,
            f"--remote-debugging-port={CDP_PORT}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Poll until Chrome's CDP endpoint is reachable.
    deadline = time.time() + 20
    while time.time() < deadline:
        if _connect_to_port():
            return f"chrome launched, connected on port {CDP_PORT}"
        time.sleep(0.3)

    raise RuntimeError(
        f"Chrome did not start within 20 s. "
        f"Check that port {CDP_PORT} is free and Chrome is installed."
    )


def send(method: str, params: dict | None = None, timeout: float = 20.0) -> Any:
    """Send one CDP command and return its result dict.

    Events that arrive while waiting for the response are discarded — they're
    not needed for command/response pairs. Use goto() for navigation, which
    reads events directly so it can wait for Page.loadEventFired.
    """
    global _msg_id, _ws
    if _ws is None:
        start()
    _msg_id += 1
    mid = _msg_id
    _ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            raw = _ws.recv(timeout=max(0.05, deadline - time.time()))
        except TimeoutError:
            break  # recv timed out; outer loop will also exit
        msg = json.loads(raw)
        if msg.get("id") == mid:
            if "error" in msg:
                raise RuntimeError(f"CDP error [{method}]: {msg['error']}")
            return msg.get("result", {})
        # It's an event — drain it and keep waiting for our response.
    raise TimeoutError(f"CDP timeout waiting for response to '{method}'")


def goto(url: str, timeout: float = 20.0) -> str:
    """Navigate to url and wait until document.readyState reaches 'complete'.

    Sequence:
      1. Bring this tab to the front (so the visible window is the one we navigate).
      2. Page.navigate — Chrome acks immediately, real load is async.
      3. Poll document.readyState + location.href via Runtime.evaluate.
         Polling is more robust than waiting for Page.loadEventFired because
         send() drains CDP events while waiting for command responses.
    """
    # Make sure we're actually driving the visible tab.
    try:
        send("Page.bringToFront")
    except Exception:
        pass

    before = ""
    try:
        before = eval_js("location.href") or ""
    except Exception:
        pass

    nav_result = send("Page.navigate", {"url": url})
    print(f"[browser] Page.navigate({url}) → {nav_result}", flush=True)
    if nav_result.get("errorText"):
        return f"ERROR: Page.navigate: {nav_result['errorText']}"

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.4)
        try:
            state = eval_js("document.readyState")
            href  = eval_js("location.href") or ""
            # Success = readyState complete AND URL actually changed from before.
            if state == "complete" and href != before and not href.startswith("about:"):
                return f"navigated → {href}"
        except Exception:
            # Runtime briefly unavailable during cross-process nav — retry.
            pass

    try:
        href = eval_js("location.href") or url
        state = eval_js("document.readyState")
    except Exception:
        href, state = url, "unknown"
    return f"WARNING: load timed out — href={href}, readyState={state}"


def eval_js(expr: str) -> Any:
    result = send(
        "Runtime.evaluate",
        {"expression": expr, "returnByValue": True, "awaitPromise": True},
    )
    r = result.get("result", {})
    if r.get("subtype") == "error":
        raise RuntimeError(r.get("description", "JS error"))
    return r.get("value")


def read_text(max_chars: int = 8000) -> str:
    text = eval_js("document.body && document.body.innerText || ''") or ""
    return text[:max_chars]


def click(selector: str) -> str:
    """Click an element matched by a standard CSS selector."""
    js = f"""
    (() => {{
        const el = document.querySelector({json.dumps(selector)});
        if (!el) return 'NOT FOUND: {selector}';
        el.scrollIntoView({{block: 'center'}});
        el.click();
        const t = (el.innerText || el.value || el.getAttribute('aria-label') || '').trim().slice(0, 80);
        return 'clicked: ' + t;
    }})()
    """
    return str(eval_js(js) or "clicked")


def click_text(text: str) -> str:
    """Click the first interactive element whose visible text matches `text`.

    Tries: exact → case-insensitive exact → substring → aria-label.
    This is what you want when the LLM would be tempted to write
    `button:contains("…")` — that's jQuery, not valid CSS.
    """
    js = f"""
    (() => {{
        const needle = {json.dumps(text)};
        const low = needle.toLowerCase();
        const sel = 'button, a, [role="button"], [role="link"], [role="tab"], '
                  + '[role="menuitem"], [onclick], input[type="submit"], '
                  + 'input[type="button"], summary, label';
        const all = [...document.querySelectorAll(sel)].filter(el => {{
            const r = el.getBoundingClientRect();
            return r.width > 0 && r.height > 0;
        }});
        const txt = el => (el.innerText || el.value || '').trim();
        const aria = el => (el.getAttribute('aria-label') || '').trim();

        let m = all.find(el => txt(el) === needle);
        if (!m) m = all.find(el => txt(el).toLowerCase() === low);
        if (!m) m = all.find(el => txt(el).toLowerCase().includes(low));
        if (!m) m = all.find(el => aria(el).toLowerCase().includes(low));
        if (!m) return 'NOT FOUND: ' + needle;
        m.scrollIntoView({{block: 'center'}});
        m.click();
        return 'clicked: ' + (txt(m) || aria(m)).slice(0, 80);
    }})()
    """
    return str(eval_js(js))


def fill(selector: str, value: str) -> str:
    js = f"""
    (() => {{
        const el = document.querySelector({json.dumps(selector)});
        if (!el) return 'NOT FOUND: {selector}';
        el.focus();
        el.value = {json.dumps(value)};
        el.dispatchEvent(new Event('input', {{bubbles:true}}));
        el.dispatchEvent(new Event('change', {{bubbles:true}}));
        return 'filled: ' + (el.name || el.id || el.placeholder || el.tagName);
    }})()
    """
    return str(eval_js(js) or "filled")


def fill_by_label(label: str, value: str) -> str:
    """Fill the input whose <label> text matches (or aria-label / placeholder)."""
    js = f"""
    (() => {{
        const needle = {json.dumps(label)}.toLowerCase();
        const inputs = [...document.querySelectorAll('input, textarea, select')];
        let el = inputs.find(i => {{
            const lbl = i.labels && i.labels[0];
            return lbl && lbl.innerText.trim().toLowerCase().includes(needle);
        }});
        if (!el) el = inputs.find(i => (i.getAttribute('aria-label') || '').toLowerCase().includes(needle));
        if (!el) el = inputs.find(i => (i.placeholder || '').toLowerCase().includes(needle));
        if (!el) el = inputs.find(i => (i.name || '').toLowerCase().includes(needle));
        if (!el) return 'NOT FOUND: label ' + {json.dumps(label)};
        el.scrollIntoView({{block: 'center'}});
        el.focus();
        el.value = {json.dumps(value)};
        el.dispatchEvent(new Event('input', {{bubbles:true}}));
        el.dispatchEvent(new Event('change', {{bubbles:true}}));
        return 'filled ' + (el.name || el.id || 'input') + ' = ' + {json.dumps(value)}.slice(0, 40);
    }})()
    """
    return str(eval_js(js))


def elements(limit: int = 60) -> str:
    """Return a numbered list of every visible interactive element on the page.

    Essential before clicking on a page you haven't seen — the LLM can then
    choose the right selector or text instead of guessing.
    """
    js = f"""
    (() => {{
        const sel = 'button, a[href], input:not([type="hidden"]), textarea, select, '
                  + '[role="button"], [role="link"], [role="tab"], [role="menuitem"], '
                  + '[onclick], [tabindex]:not([tabindex="-1"])';
        const all = [...document.querySelectorAll(sel)].filter(el => {{
            const r = el.getBoundingClientRect();
            const s = getComputedStyle(el);
            return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
        }}).slice(0, {limit});
        return all.map((el, i) => {{
            const tag  = el.tagName.toLowerCase();
            const role = el.getAttribute('role');
            const type = el.type || '';
            const text = (el.innerText || el.value || el.placeholder
                        || el.getAttribute('aria-label') || '').trim().replace(/\\s+/g, ' ').slice(0, 80);
            const href = el.href ? ' href="' + el.href.slice(0, 50) + '"' : '';
            const id   = el.id ? ' #' + el.id : '';
            return '[' + i + '] <' + tag + (type ? ' type=' + type : '') + id + (role ? ' role=' + role : '') + href + '>  "' + text + '"';
        }}).join('\\n') || '(no interactive elements found)';
    }})()
    """
    return str(eval_js(js))


# Keys we can press via CDP Input.dispatchKeyEvent.
_KEYS = {
    "Enter":      (13, "Enter",      "Enter",      "\r"),
    "Tab":        (9,  "Tab",        "Tab",        "\t"),
    "Escape":     (27, "Escape",     "Escape",     ""),
    "Backspace":  (8,  "Backspace",  "Backspace",  ""),
    "Delete":     (46, "Delete",     "Delete",     ""),
    "ArrowUp":    (38, "ArrowUp",    "ArrowUp",    ""),
    "ArrowDown":  (40, "ArrowDown",  "ArrowDown",  ""),
    "ArrowLeft":  (37, "ArrowLeft",  "ArrowLeft",  ""),
    "ArrowRight": (39, "ArrowRight", "ArrowRight", ""),
    "Home":       (36, "Home",       "Home",       ""),
    "End":        (35, "End",        "End",        ""),
    "PageUp":     (33, "PageUp",     "PageUp",     ""),
    "PageDown":   (34, "PageDown",   "PageDown",   ""),
    "Space":      (32, " ",          "Space",      " "),
}


def press_key(key: str) -> str:
    """Press a named key (Enter, Tab, Escape, ArrowDown, etc.)."""
    if key not in _KEYS:
        return f"ERROR: unknown key '{key}'. Known: {', '.join(_KEYS)}"
    vk, name, code, text = _KEYS[key]
    params = {
        "windowsVirtualKeyCode": vk,
        "key": name,
        "code": code,
    }
    if text:
        params["text"] = text
    send("Input.dispatchKeyEvent", {"type": "keyDown", **params})
    send("Input.dispatchKeyEvent", {"type": "keyUp", **params})
    return f"pressed {key}"


def wait_for(text: str, timeout: float = 10.0) -> str:
    """Wait until `text` appears anywhere in document.body.innerText."""
    deadline = time.time() + timeout
    js = f"document.body && document.body.innerText.includes({json.dumps(text)})"
    while time.time() < deadline:
        try:
            if eval_js(js):
                return f"found: {text!r} after {time.time() - (deadline - timeout):.1f}s"
        except Exception:
            pass
        time.sleep(0.3)
    return f"TIMEOUT: {text!r} did not appear within {timeout}s"


def scroll(direction: str = "down", amount: int = 600) -> str:
    """Scroll the page. direction: 'up', 'down', 'top', 'bottom'."""
    if direction == "top":
        eval_js("window.scrollTo(0, 0)")
    elif direction == "bottom":
        eval_js("window.scrollTo(0, document.body.scrollHeight)")
    elif direction == "up":
        eval_js(f"window.scrollBy(0, -{amount})")
    else:
        eval_js(f"window.scrollBy(0, {amount})")
    return f"scrolled {direction}"


def go_back() -> str:
    eval_js("history.back()")
    time.sleep(0.5)
    return "went back"


def current_url() -> str:
    return str(eval_js("location.href") or "")


def select_option(selector: str, value: str) -> str:
    """Pick an option in a <select>. Matches by value, then by visible text."""
    js = f"""
    (() => {{
        const el = document.querySelector({json.dumps(selector)});
        if (!el) return 'NOT FOUND: {selector}';
        if (el.tagName !== 'SELECT') return 'NOT A SELECT: ' + el.tagName;
        const v = {json.dumps(value)};
        const low = v.toLowerCase();
        let opt = [...el.options].find(o => o.value === v);
        if (!opt) opt = [...el.options].find(o => o.text.trim() === v);
        if (!opt) opt = [...el.options].find(o => o.text.toLowerCase().includes(low));
        if (!opt) return 'OPTION NOT FOUND: ' + v;
        el.value = opt.value;
        el.dispatchEvent(new Event('change', {{bubbles:true}}));
        return 'selected: ' + opt.text;
    }})()
    """
    return str(eval_js(js))


def screenshot(quality: int = 80) -> tuple[str, str]:
    """Capture a JPEG screenshot of the current tab.

    Returns (base64_string, saved_filepath).
    The file is written to Memory/ so it persists as an artifact.
    The base64 string is passed to the LLM as image content (for vision models).
    """
    import base64
    from pathlib import Path

    result = send("Page.captureScreenshot", {"format": "jpeg", "quality": quality})
    b64: str = result.get("data", "")
    if not b64:
        raise RuntimeError("Page.captureScreenshot returned empty data")

    mem = Path("Memory")
    mem.mkdir(exist_ok=True)
    filepath = mem / f"screenshot_{int(time.time())}.jpg"
    filepath.write_bytes(base64.b64decode(b64))
    return b64, str(filepath)


def status() -> str:
    """Diagnostic dump: all page targets and what we're attached to."""
    lines = [f"CDP port: {CDP_PORT}", f"websocket open: {_ws is not None}"]
    pages = _list_page_targets()
    lines.append(f"page targets: {len(pages)}")
    for p in pages:
        lines.append(f"  · {p.get('type','?'):10} {p.get('id','?')[:8]}…  {p.get('url','?')}")
    if _ws is not None:
        try:
            href = eval_js("location.href")
            ready = eval_js("document.readyState")
            title = eval_js("document.title")
            lines.append(f"current page: href={href!r} readyState={ready} title={title!r}")
        except Exception as e:
            lines.append(f"eval_js failed: {e}")
    return "\n".join(lines)


def stop() -> None:
    global _ws, _chrome_proc
    try:
        if _ws is not None:
            _ws.close()
    except Exception:
        pass
    _ws = None
    if _chrome_proc is not None:
        _chrome_proc.terminate()
        _chrome_proc = None
