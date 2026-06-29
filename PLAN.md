# Implementation Plan: cross-agent-mcp

A Python MCP server that wraps two AI coding agent backends (opencode and Claude Code) behind a unified `prompt` / `continue` / `abort` tool interface, for cross-family adversarial code review. Installed via `uvx` from a public GitHub repo, registered as a stdio MCP server in Claude Code.

## Background and decisions

This plan is the output of a research session. Key decisions reached:

- **Build from scratch** rather than fork `opencode-mcp`. The use case is narrow (send a prompt to a different model family, get its response). Forking means carving around 74 unneeded tools; ~200 lines from scratch gets exactly the right surface.
- **Two backends, both Python.** opencode via `httpx` to `opencode serve`; Claude via `claude-agent-sdk` (Python). Both run the agent loop in-process.
- **Optional Claude dependency.** `claude-agent-sdk` is an extra; if not installed, only `opencode_*` tools register. Lets users install opencode-only or both.
- **Generic prompt tool, not a dedicated reviewer tool.** The caller provides the review-focused prompt, same as how codex MCP works today. Three tools per backend (prompt + continue + abort).
- **Claude auth uses stored enterprise OAuth**, not an API key. Verified: `claude -p` with `ANTHROPIC_API_KEY` unset succeeds using stored CoreWeave enterprise auth. The SDK spawns the same CLI with inherited environment.
- **Public personal repo** at `github.com/npratt/cross-agent-mcp`.
- **Skill migration is out of scope for this plan.** Build the wrapper only; migrating the 30+ `mcp__codex__codex` call sites in `dot_claude/skills/` is a separate follow-up.

## Stack

- Python 3.10+
- `uv` for development and install
- `mcp[cli]` (FastMCP) for the MCP server
- `httpx` for the opencode backend
- `claude-agent-sdk>=0.2.110` for the Claude backend (optional extra)
- `ruff` for lint/format
- `pytest` + `pytest-asyncio` for tests

## Architecture

```
cross-agent-mcp/
├── pyproject.toml
├── LICENSE                          # MIT
├── README.md
├── src/
│   └── cross_agent/
│       ├── __init__.py
│       ├── server.py                # FastMCP entry point, tool registration
│       ├── config.py                # env var config, backend detection
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── base.py              # Backend protocol
│       │   ├── opencode.py          # httpx to opencode serve
│       │   └── claude.py            # claude-agent-sdk
│       └── tools/
│           ├── __init__.py
│           └── prompt.py            # prompt + continue + abort tool factories
├── tests/
│   ├── test_opencode_backend.py
│   ├── test_claude_backend.py
│   ├── test_mcp_tools.py
│   └── conftest.py
└── .github/
    └── workflows/
        └── ci.yml                   # ruff + pytest
```

## Phase 1: The cross-agent-mcp repo

### 1.1 Repo scaffold

- `github.com/npratt/cross-agent-mcp` (public, MIT license)
- Use `uv init --package cross-agent-mcp` to bootstrap, then adjust `pyproject.toml` below.
- Module path: `src/cross_agent/`

### 1.2 `pyproject.toml`

```toml
[project]
name = "cross-agent-mcp"
version = "0.1.0"
description = "MCP server wrapping opencode and Claude Code for cross-family adversarial review"
requires-python = ">=3.10"
license = { text = "MIT" }
dependencies = [
    "mcp[cli]>=1.0",
    "httpx>=0.27",
]

[project.optional-dependencies]
claude = ["claude-agent-sdk>=0.2.110"]
all = ["claude-agent-sdk>=0.2.110"]

[project.scripts]
cross-agent-mcp = "cross_agent.server:main"

[tool.ruff]
line-length = 100
target-version = "py310"

[dependency-groups]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.23",
    "ruff>=0.5",
]
```

Key design: `claude-agent-sdk` is an optional dependency. If not installed, only `opencode_*` tools register. Install variants:

- `uvx --from git+https://github.com/npratt/cross-agent-mcp cross-agent-mcp` (opencode only)
- `uvx --from "git+https://github.com/npratt/cross-agent-mcp[claude]" cross-agent-mcp` (both)

### 1.3 Backend protocol (`src/cross_agent/backends/base.py`)

```python
from typing import Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    @property
    def name(self) -> str:
        """Backend identifier used as the tool name prefix (e.g. 'opencode', 'claude')."""
        ...

    async def prompt(
        self,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        model: str | None = None,
    ) -> tuple[str, str]:
        """One-shot prompt. Returns (response_text, session_id).
        Stateless by default: caller may discard session_id."""
        ...

    async def continue_session(
        self,
        session_id: str,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
    ) -> str:
        """Continue an existing session. Returns response text."""
        ...

    async def abort(self, session_id: str) -> None:
        """Abort a running session."""
        ...
```

### 1.4 OpenCode backend (`src/cross_agent/backends/opencode.py`)

**Transport:** `httpx.AsyncClient` to `opencode serve` (default `http://127.0.0.1:4096`).

**Lifecycle:**
- On first call, health-check `GET /global/health` (3s timeout).
- If unreachable and `CROSS_AGENT_OPENCODE_AUTOSERVE=true`, spawn `opencode serve --port 4096` as a subprocess, wait for health (max 10s, poll every 500ms).
- Store the subprocess handle for cleanup on MCP server shutdown.

**Read-only enforcement:**
- When `read_only=True`, send `agent: "cross-agent-reviewer"` in the message body.
- The `cross-agent-reviewer` agent is defined in `opencode.json` (Phase 2 wiring) with read + glob + grep + list + webfetch + websearch + task permissions, no edit/bash.
- This is belt-and-suspenders with the agent config; the wrapper defaults to read-only AND uses the read-only agent.

**Session flow:**

```
prompt(text, read_only=True, directory=None, model=None):
  1. POST /session {title: text[:80], x-opencode-directory: directory?}  -> session_id
  2. POST /session/{sid}/message {
       parts: [{type: "text", text}],
       agent: read_only ? "cross-agent-reviewer" : undefined,
       model: model ? {providerID, modelID} : undefined,
     }
  3. Parse response parts for assistant text (formatMessageResponse equivalent)
  4. DELETE /session/{sid}  (stateless cleanup)
  5. Return (text, session_id)

continue_session(session_id, text, read_only=True, directory=None):
  1. POST /session/{sid}/message {parts: [{type: "text", text}], agent: read_only ? reviewer}
  2. Parse response
  3. Return text  (no delete - session stays alive)

abort(session_id):
  1. POST /session/{sid}/abort
```

**Config env vars:**

| Env var | Default | Purpose |
|---|---|---|
| `CROSS_AGENT_OPENCODE_BASE_URL` | `http://127.0.0.1:4096` | opencode server URL |
| `CROSS_AGENT_OPENCODE_AUTOSERVE` | `true` | Auto-start `opencode serve` if down |
| `CROSS_AGENT_OPENCODE_AGENT` | `cross-agent-reviewer` | Agent to use for read-only calls |
| `OPENCODE_SERVER_PASSWORD` | unset | Optional basic auth password |
| `OPENCODE_SERVER_USERNAME` | `opencode` | Optional basic auth username |

**Header injection defense:** Strip NUL/CR/LF from `directory` before setting `x-opencode-directory` header (mirrors the pattern from the audited `opencode-mcp` repo).

**Secret redaction:** Scrub API-key-like patterns (prefixes `sk-`, `ghp_`, `hf_`, `xoxb-`, `whsec_`, long hex/base64 tokens) from the response text before returning to the MCP client. Mirrors the `redactSecrets` pattern from `opencode-mcp` - prevents provider keys from leaking into LLM-visible tool output.

### 1.5 Claude backend (`src/cross_agent/backends/claude.py`)

**Transport:** `claude-agent-sdk` Python package (`query()` for one-shots, `ClaudeSDKClient` for continuation).

**Auth:** Inherits `claude` CLI's stored enterprise auth. Verified working: `claude -p` with `ANTHROPIC_API_KEY` explicitly unset returns a successful result using stored CoreWeave enterprise OAuth. The SDK spawns the same CLI subprocess with inherited environment, so no API key is required.

**Read-only enforcement:**

```python
read_only_tools = ClaudeAgentOptions(
    allowed_tools=["Read", "Glob", "Grep", "List", "WebSearch", "WebFetch"],
    disallowed_tools=["Write", "Edit", "Bash", "ApplyPatch"],
)
write_tools = ClaudeAgentOptions(
    # No tool restrictions; inherits defaults
)
```

When `read_only=True`, use the read-only options. `allowed_tools` pre-approves safe operations; `disallowed_tools` removes write/bash from the model's context entirely.

**Session flow:**

```
prompt(text, read_only=True, directory=None, model=None):
  1. options = ClaudeAgentOptions(
       allowed_tools=..., disallowed_tools=...,
       cwd=directory, model=model,
     )
  2. async for msg in query(prompt=text, options=options):
       if SystemMessage and subtype == "init": capture session_id from msg.data
       if ResultMessage: capture msg.result
  3. Return (result, session_id)
  (Sessions persist on disk via CLI; no explicit cleanup needed for stateless one-shots.
   The session ID is returned so the caller can continue or let it age out.)

continue_session(session_id, text, read_only=True, directory=None):
  1. options = ClaudeAgentOptions(
       resume=session_id,
       allowed_tools=..., disallowed_tools=...,
       cwd=directory,
     )
  2. async for msg in query(prompt=text, options=options):
       if ResultMessage: capture msg.result
  3. Return result

abort(session_id):
  1. Track running asyncio tasks per session_id at the Backend level
  2. On abort, cancel the asyncio task for that session
  3. Fallback for already-completed sessions: `claude stop {session_id}` subprocess
```

**Config env vars:**

| Env var | Default | Purpose |
|---|---|---|
| `CROSS_AGENT_CLAUDE_MODEL` | unset (uses CLI default) | Override Claude model |

No auth env vars needed; uses stored CLI auth.

### 1.6 MCP server + tools (`src/cross_agent/server.py` + `tools/prompt.py`)

**Entry point:** `cross_agent.server:main` - constructs `FastMCP("cross-agent")`, registers tools for each available backend, runs via stdio transport.

**Backend detection at startup:**

```python
def get_available_backends() -> list[Backend]:
    backends: list[Backend] = [OpenCodeBackend()]
    try:
        import claude_agent_sdk  # noqa: F401
        backends.append(ClaudeBackend())
    except ImportError:
        pass  # claude-agent-sdk not installed, skip
    return backends
```

**Tool registration:** For each available backend, register 3 tools using the backend's `name` as the prefix:

```python
def register_tools(mcp: FastMCP, backend: Backend) -> None:
    prefix = backend.name  # "opencode" or "claude"

    @mcp.tool()
    async def opencode_prompt(  # name varies by backend
        prompt: str,
        read_only: bool = True,
        directory: str | None = None,
        model: str | None = None,
    ) -> str:
        f"""Send a one-shot prompt to {prefix} for adversarial review.
        Read-only by default (can read files, search web, spawn subagents).
        Returns response text and session_id."""
        text, sid = await backend.prompt(prompt, read_only, directory, model)
        return f"Session: {sid}\n\n{text}"

    @mcp.tool()
    async def opencode_continue(
        session_id: str,
        prompt: str,
        read_only: bool = True,
        directory: str | None = None,
    ) -> str:
        f"""Continue an existing {prefix} session."""
        return await backend.continue_session(session_id, prompt, read_only, directory)

    @mcp.tool()
    async def opencode_abort(session_id: str) -> str:
        f"""Abort a running {prefix} session."""
        await backend.abort(session_id)
        return f"Session {session_id} aborted."
```

Tool names use the backend prefix: `opencode_prompt`, `opencode_continue`, `opencode_abort`, `claude_prompt`, `claude_continue`, `claude_abort`. Claude Code exposes these as `mcp__cross-agent__opencode_prompt`, etc.

**Implementation note:** The tool functions need distinct names at module level. Use a factory that creates closures bound to each backend, or dynamically set `__name__` on the decorated function. FastMCP's `@mcp.tool()` uses the function name by default; ensure each registration uses a unique name like `opencode_prompt` and `claude_prompt`.

### 1.7 Tests

**Unit tests (mocked, no real servers):**

`test_opencode_backend.py`:
- Mock `httpx.AsyncClient`. Test prompt (POST /session -> POST /message -> DELETE /session), continue (POST /message only), abort (POST /abort).
- Test auto-start logic: mock health check failure, verify `opencode serve` subprocess spawn attempt.
- Test read-only agent selection: verify `agent: "cross-agent-reviewer"` in message body when `read_only=True`.
- Test header injection defense: verify NUL/CR/LF stripped from `directory`.
- Test secret redaction: verify `sk-...` tokens scrubbed from response.

`test_claude_backend.py`:
- Mock `claude_agent_sdk.query`. Test prompt (capture session_id + result), continue (verify `resume` param set), abort (verify task cancellation).
- Test read-only tool selection: verify `allowed_tools` and `disallowed_tools` set correctly.
- Skip entire module if `claude-agent-sdk` not installed (`pytest.importorskip`).

`test_mcp_tools.py`:
- Test tool registration: 3 tools per installed backend.
- Test backend detection: with and without `claude-agent-sdk` (mock import).
- Test tool naming: `opencode_prompt` vs `claude_prompt`.

**Integration tests (optional, behind `@pytest.mark.integration`):**
- Start `opencode serve`, send a real prompt, verify non-empty response.
- Use real `claude -p` via SDK, send a real prompt, verify non-empty response.
- Skip in CI (needs real servers/auth). Run manually with `pytest -m integration`.

**CI:** GitHub Actions `.github/workflows/ci.yml`:
- `uv sync --all-extras`
- `ruff check`
- `pytest` (unit only; integration tests auto-skipped without marker)

### 1.8 README

Concise README covering:
- What it does (cross-family adversarial review MCP)
- Install variants (opencode-only, both backends)
- Claude Code config snippet (see Phase 2.1)
- Opencode reviewer agent setup (see Phase 2.2)
- Env var reference (tables from 1.4 and 1.5)
- Tool reference (prompt, continue, abort per backend, with read_only default)

---

## Phase 2: Dotfiles wiring

Out of scope for the build, but documented here so the follow-up session knows what to do. The user explicitly excluded skill migration from this plan.

### 2.1 MCP server entry in Claude config

Add to `~/.claude.json` `mcpServers` (alongside the existing `codex` entry):

```json
"cross-agent": {
  "type": "stdio",
  "command": "uvx",
  "args": ["--from", "git+https://github.com/npratt/cross-agent-mcp[claude]", "cross-agent-mcp"],
  "env": {}
}
```

Uses the `[claude]` extra since both backends are wanted. `~/.claude.json` is not chezmoi-managed (runtime state); add via `claude mcp add` or manual edit.

### 2.2 OpenCode reviewer agent

Add to `dot_config/opencode/opencode.json` in the dotfiles source tree, alongside the existing `provider` block:

```json
"agent": {
  "cross-agent-reviewer": {
    "description": "Read-only reviewer for cross-agent adversarial review. Can read files, search code, search the web, and spawn subagents. Cannot write or execute commands.",
    "mode": "primary",
    "permission": {
      "edit": "deny",
      "bash": "deny",
      "read": "allow",
      "glob": "allow",
      "grep": "allow",
      "list": "allow",
      "webfetch": "allow",
      "websearch": "allow",
      "task": "allow"
    }
  }
}
```

Apply via `chezmoi --source "$PWD" apply` after editing the source. This agent is what the opencode backend uses when `read_only=True` (see 1.4).

### 2.3 CLAUDE.md MCP documentation

Add a section to `dot_claude/CLAUDE.md` parallel to the existing "Codex MCP" section:

```markdown
## Cross-Agent MCP

When using `mcp__cross-agent__opencode_prompt` or `mcp__cross-agent__claude_prompt`:
- Do not manually specify the `model` parameter unless the user explicitly requests a specific model.
- Let each backend use its configured default (opencode: GLM-5.2 via WANDB, claude: your enterprise default).
- `read_only=True` is the default and correct for adversarial review. Only set `read_only=False` if the user explicitly asks for write access.
- Use `opencode_prompt` for cross-family diversity (GLM-5.2 reviewing Claude/GPT work).
- Use `claude_prompt` when you need a Claude opinion from a different model tier (e.g., Opus reviewing Sonnet work) or from within opencode/codex.
- For session continuation, pass the returned `session_id` to the matching `_continue` tool.
```

### 2.4 Codex AGENTS.md MCP documentation

Add a parallel section to `dot_codex/AGENTS.md` so Codex knows how to use cross-agent when it needs an adversarial second opinion.

### 2.5 Skill migration (DEFERRED)

Explicitly out of scope per user instruction. The 30+ `mcp__codex__codex` call sites in `dot_claude/skills/` stay as-is for now. When ready, the migration pattern is:

```
# Before (codex):
Call `mcp__codex__codex` with:
- `sandbox`: `"read-only"`
- `approval-policy`: `"never"`
- `cwd`: "CWD"

# After (opencode via cross-agent):
Call `mcp__cross-agent__opencode_prompt` with:
- `read_only`: `true`
- `directory`: "CWD"
```

Priority files for eventual migration (not part of this plan):
- `team-branch-review/templates/reviewer-prompt.md` (High)
- `parallel-branch-review/templates/reviewer-prompt.md` (High)
- `team-commit-review/templates/reviewer-prompt.md` (High)
- `team-branch-fix/references/codex-validation.md` (High)
- `team-branch-fix/SKILL.md` (High)
- `grug-cut-pass/SKILL.md` (High)
- `grug-cut-pass/references/phase-3-codex-pass.md` (Medium)
- `issue-plan/SKILL.md` (Medium)
- `issue-plan-codex/SKILL.md` (Medium)
- `issue-plan-hybrid/SKILL.md` (Medium)
- `pr-review-import/SKILL.md` (Medium)

---

## Execution order

1. **Scaffold the repo** - `pyproject.toml`, `src/cross_agent/` structure, `tests/`, CI workflow
2. **Implement the backend protocol** (`base.py`)
3. **Implement the opencode backend** (`opencode.py`) - httpx calls, auto-start, read-only agent, secret redaction
4. **Implement the Claude backend** (`claude.py`) - `claude-agent-sdk`, read-only tool selection, session tracking for abort
5. **Implement the MCP server + tools** (`server.py`, `tools/prompt.py`) - backend detection, tool registration, stdio transport
6. **Write tests** - unit tests for both backends, tool registration tests
7. **Write the README** - install, config, env vars, tool reference
8. **Verify locally** - `uvx --from . cross-agent-mcp` starts; `opencode_prompt` returns a real response; `claude_prompt` returns a real response
9. **Commit and push** - first version on `github.com/npratt/cross-agent-mcp`
10. **End of build scope.** Phase 2 (config wiring, docs, skill migration) is a separate follow-up.

## Verification checklist

Before considering the build done:
- [ ] `uv sync --all-extras` succeeds
- [ ] `ruff check` passes
- [ ] `pytest` (unit) passes
- [ ] `uvx --from . cross-agent-mcp` starts and stays alive on stdio
- [ ] `opencode_prompt` with a trivial prompt returns a non-empty response
- [ ] `opencode_prompt` with `read_only=True` sends `agent: "cross-agent-reviewer"` in the message body
- [ ] `opencode_continue` with a returned session_id gets a follow-up response
- [ ] `opencode_abort` cancels a running session
- [ ] `claude_prompt` with a trivial prompt returns a non-empty response (using stored enterprise auth, no API key)
- [ ] `claude_prompt` with `read_only=True` passes `disallowed_tools=["Write", "Edit", "Bash"]`
- [ ] `claude_continue` with a returned session_id gets a follow-up response
- [ ] `claude_abort` cancels a running session
- [ ] Without `claude-agent-sdk` installed, only `opencode_*` tools register (no `claude_*` tools)
- [ ] Secret redaction scrubs `sk-...` patterns from response text
- [ ] NUL/CR/LF stripped from `directory` before setting `x-opencode-directory` header
- [ ] README has install commands, config snippet, env var table, tool reference
- [ ] CI workflow runs ruff + pytest on push

## Open implementation notes

- **FastMCP tool naming with closures:** `@mcp.tool()` uses the function's `__name__` by default. When registering tools in a loop per backend, ensure each decorated function gets a unique name (e.g., via a factory or `functools.wraps` with a computed name). Verify the MCP client sees `opencode_prompt` and `claude_prompt`, not `prompt` or `wrapper`.
- **Opencode server subprocess lifecycle:** If auto-starting `opencode serve`, decide whether to kill it on MCP server shutdown (probably yes, since the wrapper spawned it) or leave it running (if the user started it separately). Track whether the wrapper started it vs. detected an existing one.
- **Claude abort for already-finished sessions:** The asyncio task for a `prompt()` call may have completed by the time `abort()` is called. Handle gracefully: if no running task, attempt `claude stop {session_id}` as a fallback, and return success either way.
- **Error surfacing:** Both backends should return error text in the tool result (not raise) so the MCP client sees a useful message. Match the `toolError` pattern from `opencode-mcp`: catch exceptions, format as `Error: {message}`, return as tool result with `isError: true` if the MCP framework supports it.
- **Response formatting:** The opencode backend needs a `formatMessageResponse` equivalent - parse the JSON response from `POST /session/:id/message` and extract assistant text parts. Look at `opencode-mcp`'s `src/helpers.ts` for the format; opencode messages have `parts` with `type: "text"` for assistant content.
- **Model parameter format:** The opencode API expects `model: {providerID, modelID}` in the message body, not a flat string. The MCP tool takes a flat string like `wandb/zai-org/GLM-5.2`; the backend splits on `/` to build the object. Decide whether to expose `providerID` + `modelID` as separate params or keep the flat `provider/model` string (flat matches how opencode config and `opencode run -m` work).
