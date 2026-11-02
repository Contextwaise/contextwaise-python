<p align="center">
  <a href="https://contextwaise.com">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://contextwaise.com/logo-dark.png">
      <img src="https://contextwaise.com/logo.png" alt="Contextwaise" width="280">
    </picture>
  </a>
</p>

<p align="center"><strong>Shrinks agent tool results before the model reads them: cache-safe, reversible, intent-aware.</strong></p>

<p align="center">
  <a href="https://pypi.org/project/contextwaise/"><img src="https://img.shields.io/pypi/v/contextwaise?labelColor=171717&color=80cd00" alt="PyPI version"></a>
  <a href="https://pypi.org/project/contextwaise/"><img src="https://img.shields.io/pypi/pyversions/contextwaise?labelColor=171717&color=80cd00" alt="Python versions"></a>
  <a href="https://pypi.org/project/contextwaise/"><img src="https://img.shields.io/pypi/types/contextwaise?labelColor=171717&color=80cd00" alt="Typed"></a>
  <a href="https://opensource.org/license/mit"><img src="https://img.shields.io/badge/license-MIT-80cd00?labelColor=171717" alt="MIT license"></a>
</p>

<p align="center">
  <a href="https://contextwaise.com">Website</a> ·
  <a href="https://contextwaise.com/docs/sdk">Docs</a> ·
  <a href="https://www.npmjs.com/package/contextwaise">TypeScript SDK</a>
</p>

---

In an agent loop, every tool result is resent on every later turn. A 20K-token test log read at turn 3 of a 20-turn task is paid ~17 more times. Contextwaise gives the model a short view of each result, once, at the moment the tool returns:

1. **Shapers** (free, local, deterministic): strip color codes and progress bars, drop passing tests, collapse repeated log lines, remove empty JSON fields, turn HTML into text. Errors are never removed.
2. **Size gate**: results that fit the budget pass as they are.
3. **Relevance**: the parts that match what the agent was doing (the text it wrote before the call, plus the call's arguments) are kept within the budget; everything else becomes `… N lines omitted`.
4. **Nothing is lost**: the model can call `expand(ref, question)` to get any part back.

Cache-safe by design: a result's view is decided once and replayed byte-for-byte by tool call id, so the provider's prompt cache keeps hitting and edited-history checks never trip.

```bash
pip install contextwaise
```

Python 3.10+. One dependency: [httpx](https://www.python-httpx.org/).

## Usage

Shrink each tool result before it goes back to the model, and give the model the `expand` tool. With the Anthropic SDK, by hand (the [adapters](#vercel-ai-sdk-for-python) below do this for you):

```python
import anthropic
from contextwaise import EXPAND_TOOL, Contextwaise

client = anthropic.Anthropic()
expand_tool = {
    "name": EXPAND_TOOL["name"],
    "description": EXPAND_TOOL["description"],
    "input_schema": EXPAND_TOOL["parameters"],
}
tools = [*my_tools, expand_tool]

# Reads CONTEXTWAISE_API_KEY and CONTEXTWAISE_URL from the environment.
with Contextwaise(agent="support-triage", budgets={"read_file": 1500}) as cw:
    messages = [{"role": "user", "content": question}]
    while True:
        response = client.messages.create(
            model="claude-opus-5-5", max_tokens=4096, tools=tools, messages=messages
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        # Why the agent calls each tool: the request, then what the model wrote before the call.
        said = question
        results = []
        for block in response.content:
            if block.type == "text":
                said += " " + block.text
            elif block.type == "tool_use":
                cw.note_intent(block.id, said)
                if block.name == "expand":
                    # The call's size: expanding costs one more call about as big,
                    # and the dashboard counts it against the savings.
                    u = response.usage
                    size = (
                        u.input_tokens
                        + (u.cache_read_input_tokens or 0)
                        + (u.cache_creation_input_tokens or 0)
                    )
                    content = cw.expand(**block.input, context_tokens=size)
                else:
                    output = run_tool(block.name, block.input)
                    view = cw.shrink(call_id=block.id, tool=block.name, args=block.input, output=output)
                    content = view.text
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
        messages.append({"role": "user", "content": results})
```

- Your code keeps the raw `output`; only the model's view changes.
- `output` can be text or anything JSON-serializable (Pydantic models and dataclasses are serialized by their fields). `args` can be a dict or JSON text (OpenAI's `arguments`).
- `shrink` returns a `ShrinkResult`: `text` (what the model reads), `mode` (`full`, `shaped` or `trimmed`), `reason` (why: `cut`, `fits_budget`, `small_saving`, `schema`, `marked_full`, `api_unavailable`, …), token counts before and after, and what was kept and why.
- Hosting many agents on one client? Pass `agent=` to `shrink` (and to the Anthropic/OpenAI helpers) to group each call under its own agent.
- Tool schemas and lists of actions (an MCP server describing its tools, OpenAI or Anthropic tool definitions, JSON Schema) are detected and always reach the model whole: the next call must follow them exactly.

### Vercel AI SDK for Python

`ContextwaiseAgent` is an [`ai.Agent`](https://ai-python.dev/docs) whose model reads Contextwaise's view of each tool result. Your app (events, UI) keeps the raw output, the history carries the view (so later turns resend the same bytes), and an `expand` tool is added.

```bash
pip install "contextwaise[ai]"   # the ai package needs Python 3.12+
```

```python
import ai
from contextwaise import AsyncContextwaise
from contextwaise.ai_sdk import ContextwaiseAgent

async with AsyncContextwaise(agent="support-triage") as cw:
    agent = ContextwaiseAgent(cw, tools=[get_orders, search_logs])  # your @ai.tool functions
    async with agent.run(ai.get_model("anthropic:claude-opus-5-5"), [ai.user_message(question)]) as stream:
        async for event in stream:
            ...
```

Pass `full=lambda tool, args: ...` to `ContextwaiseAgent` for calls whose output must reach the model whole (e.g. a file the agent will edit; schemas and lists of actions already are), and `agent=` when one client serves many agents. The adapter also reports the model id with each result, for your dashboard.

### Anthropic and OpenAI SDKs

The adapters do the loop body for you: one call per model response runs your tools and returns what to send back, each output shrunk, `expand` answered, and a tool that raises reported to the model as an error. `run(name, args)` returns your tool's raw output.

```python
from contextwaise.anthropic import EXPAND_TOOL, tool_results

response = client.messages.create(
    model=model, max_tokens=4096, messages=messages, tools=[*my_tools, EXPAND_TOOL]
)
messages.append({"role": "assistant", "content": response.content})
if response.stop_reason == "tool_use":
    messages.append({"role": "user", "content": tool_results(cw, response, run_tool, request=question)})
```

```python
from contextwaise.openai import EXPAND_TOOL, EXPAND_TOOL_RESPONSES, function_outputs, tool_messages

# Chat Completions
message = completion.choices[0].message
usage = completion.usage  # a message doesn't carry it: pass it so expands are counted
messages += [
    message,
    *tool_messages(cw, message, run_tool, request=question, prompt_tokens=usage.prompt_tokens),
]

# Responses API
items += [*response.output, *function_outputs(cw, response, run_tool, request=question)]
```

With `AsyncContextwaise`, use `atool_results`, `atool_messages` and `afunction_outputs`: the calls run concurrently and `run` may be async. The adapters read the SDKs' objects (or plain dicts); neither SDK is a dependency.

### asyncio

`AsyncContextwaise` has the same interface, with `await` on `shrink`, `expand` and `flush`, and `async with`. The relevance API never blocks the event loop.

```python
async with AsyncContextwaise(agent="support-triage") as cw:
    view = await cw.shrink(call_id=call.id, tool=call.name, args=call.input, output=output)
```

## Options

| Option | Default | What it does |
|---|---|---|
| `api_key` | `CONTEXTWAISE_API_KEY` | Your Contextwaise key (`sw_live_…`). Relevance is then judged by the API (Jev), and results are reported to your dashboard: sizes, tool names and the model id only, never tool output. Without a key, only the local cleanup runs: noise is removed, nothing relevant is ever cut. |
| `base_url` | `CONTEXTWAISE_URL` | Where your Contextwaise app runs. Required with a key. |
| `agent` | — | Groups this agent's results on the dashboard. Each call can override it (`agent=` in `shrink`, `ContextwaiseAgent` and the Anthropic/OpenAI helpers). |
| `max_tokens` | `600` | Budget for the model's view of one tool result. |
| `budgets` | — | Per-tool budgets, e.g. `{"read_file": 1500}`: files up to that size reach the model whole. |
| `full` | — | Tools whose output always reaches the model whole. Schemas and lists of actions already are. |
| `ledger` | in memory, last 5,000 calls | Where originals and views live. Implement `get(key)` / `set(key, entry)` over Redis or a database for agents spread over processes. |

The API is never in the agent's way: scoring times out after 4 s and is then skipped for 30 s, or an hour once the free tokens are used up (a single result the API rejects as too big is just passed whole); meanwhile the model reads the cleaned output whole (nothing is cut without the API). Reports are batched and sent from a background thread, and dropped if they can't be sent. Leave the `with` block (or call `close()`) before a short-lived process exits, so pending reports are sent.

Decisions are logged on the `contextwaise` logger (`DEBUG`), API problems at `WARNING`.

## Development

```bash
git clone https://github.com/Contextwaise/contextwaise-python.git && cd contextwaise-python
uv sync                 # virtualenv with the dev tools
uv run pytest           # tests (no network: the API is stubbed)
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv build                # sdist + wheel in dist/
```

The TypeScript SDK ([`contextwaise` on npm](https://www.npmjs.com/package/contextwaise)) lives in [Contextwaise/contextwaise-typescript](https://github.com/Contextwaise/contextwaise-typescript), with the rules it shares with this package in its [`core`](https://github.com/Contextwaise/contextwaise-typescript/tree/main/core) (`shapers.ts`, `relevance.ts`). The two SDKs make the same decisions: a change to the shapers, relevance or the SDK core should land in both.

## License

[MIT](LICENSE)
