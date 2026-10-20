# CCHarness

**A minimal LLM agent harness in ~1,500 lines of plain Python.** No frameworks,
no agent SDK, no hidden abstractions: just a tool-calling loop around any
OpenAI-compatible model, plus a shell, a real browser, a markdown memory vault,
sub-agents and MCP.

It exists to make one point concrete: *an agent is a loop around an LLM call
with a few tools.* Give a model a shell, a browser and a place to write notes,
and most of the interesting behaviour is emergent. Every file is meant to be
read in one sitting.

## Features

- **Provider-agnostic loop** - standard `chat.completions` tool-calling loop.
  Works against OpenAI or any compatible server (LM Studio, Ollama, vLLM,
  OpenRouter) via `OPENAI_API_BASE`. Defaults to a local model.
- **Shell** - run any CLI command (120 s timeout, output tail-truncated).
- **Browser over raw CDP** - talks Chrome DevTools Protocol over a websocket,
  no Playwright/Selenium. 17 tools: `goto`, `read_text`, `click`,
  `click_text`, `fill`, `fill_by_label`, `elements`, `key`, `wait_for`,
  `scroll`, `back`, `select`, `eval`, `snapshot`, and more. Launches an
  isolated Chrome profile or attaches to one already listening on the CDP port.
- **Vision in the loop** - `browser_snapshot` captures a screenshot, saves it
  to the vault and injects it back into the conversation as an image so
  vision-capable models can see the page.
- **Markdown memory vault** - `Memory/` is an Obsidian-compatible vault. The
  agent writes, reads, lists and searches `.md` notes linked with
  `[[wiki-links]]`.
- **Session log** - every run creates `Memory/{epoch}.md`; every user turn,
  tool call and result is appended automatically, and the agent can add its
  own findings with `context_append`. Later sessions can read them back.
- **Sub-agents** - `sub_agent` runs the same loop with a fresh context (and
  without the ability to spawn further sub-agents), saves the full transcript
  as `Memory/research_{ms}.md`, and returns only a short summary plus a
  wiki-link. Deep research without polluting the main context window.
- **MCP client** - a ~150-line stdio JSON-RPC client loads every server in
  `mcp.json` and exposes its tools as `mcp__{server}__{tool}`.

## Architecture

```
                 +--------------------------------------------+
  user prompt -> |  agent.py  _run_loop()                      |
                 |                                            |
                 |   messages --> LLM (chat.completions)      |
                 |        ^             |                     |
                 |        |       tool_calls?                 |
                 |        |       /        \                  |
                 |   tool results  yes      no --> reply      |
                 |        ^         |                         |
                 |        +---- dispatch by name              |
                 +------------------|-------------------------+
                                    |
      +-----------+-----------+-----+------+--------------+-------------+
      |           |           |            |              |             |
    shell     browser.py   memory_*   context_append   sub_agent     mcp_loader.py
  (subprocess) (CDP over   (Memory/    (Memory/        (same loop,   (stdio JSON-RPC
               websocket)   *.md)       {epoch}.md)     fresh ctx)    to MCP servers)
```

| File | Lines | Purpose |
|------|------:|---------|
| `agent.py` | ~840 | System prompt, built-in tools, the loop, session log, sub-agent, REPL |
| `browser.py` | ~510 | CDP client: launch/attach Chrome, navigation, DOM helpers, screenshots |
| `mcp_loader.py` | ~150 | Spawn MCP servers, `initialize` handshake, `tools/list`, `tools/call` |

Tools are plain Python functions decorated with their OpenAI JSON schema; the
registry is a list of `{name, schema, fn}` dicts. MCP tools are appended to the
same list at startup, so the model cannot tell built-in and MCP tools apart.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env      # set endpoint, key and model
python agent.py           # REPL (type /exit to quit)
python agent.py "open example.com and tell me the h1"   # one-shot
```

Chrome (or Chromium/Edge) must be installed for the browser tools.

## Configuration

| Variable | Default | Notes |
|----------|---------|-------|
| `OPENAI_API_BASE` | `http://localhost:1234/v1` | Any OpenAI-compatible endpoint; use `https://api.openai.com/v1` for OpenAI |
| `OPENAI_API_KEY` | empty | Not needed for most local servers |
| `OPENAI_MODEL` | `google/gemma-4-e4b` | Must support tool calling; vision optional |
| `CHROME_PATH` | auto-detected | Path to a Chromium-based browser |
| `CDP_PORT` | `9222` | Remote debugging port to launch on / attach to |

### Adding MCP servers

`mcp.json` uses the common `mcpServers` format:

```json
{
  "mcpServers": {
    "fs": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    }
  }
}
```

On startup each server is spawned, initialised and asked for its tools. A
server that fails to start is reported and skipped.

## Design trade-offs

- **Raw CDP instead of Playwright.** More code to own (target selection,
  load waiting, selector helpers), but zero heavy dependencies and every
  browser action is a visible websocket message you can read and debug.
- **Markdown files instead of a vector store.** Memory is human-readable,
  editable in Obsidian and diffable. Retrieval is substring search, which is
  crude but predictable; the model compensates by linking notes.
- **Screenshots as user messages.** The `tool` role cannot carry images in the
  chat-completions API, so screenshots taken during a turn are batched into a
  single follow-up user message.
- **Sub-agent returns a summary, not a transcript.** Keeps the parent context
  small; the full trace is on disk if the parent needs it.
- **Single process, synchronous.** Easy to follow and to step through in a
  debugger; tool calls within a turn run sequentially.

## Limitations

- No sandboxing: the shell tool runs arbitrary commands as your user. Run it in
  a VM or container if you point it at untrusted tasks.
- No streaming, no token/cost accounting, no context-window management beyond
  truncating tool output (16k chars per result); long sessions will eventually
  overflow the model's context.
- Loop limits are fixed (30 turns for the main agent, 12 for sub-agents).
- Memory search is case-insensitive substring matching only.
- MCP support covers stdio servers and tools; no resources, prompts or HTTP
  transport.
- Small local models are noticeably less reliable at multi-step browser tasks
  than frontier models.

## License

MIT - see [LICENSE](LICENSE).
