# cross-agent-mcp

> **Warning: this project was vibe coded.** It was built interactively with GLM-5.2 on the [opencode](https://opencode.ai) harness, with inference provided by [WandB](https://wandb.ai). Expect rough edges, untested code paths, and bugs. The author is actively using it day to day, so issues are being found and fixed as they surface, but treat the current state accordingly.

An MCP server that wraps two AI coding-agent backends - [opencode](https://opencode.ai) and [Claude Code](https://docs.claude.com/en/docs/claude-code) - behind a unified `prompt` / `continue` / `abort` tool interface, for cross-family adversarial code review. One backend lets a Claude session ask GLM-5.2 (via opencode) for a second opinion; the other lets opencode or Codex ask Claude for one. The server runs as a stdio MCP server and is intended to be registered in whichever coding agent you drive day to day.

Install just the opencode backend, or install both with the optional `[claude]` extra. No API keys are required: opencode uses its own configured provider auth, and the Claude backend reuses the `claude` CLI's stored login.

## Install

Both variants install the `cross-agent-mcp` command via [`uvx`](https://docs.astral.sh/uv/):

```bash
# opencode backend only (GLM-5.2 via opencode)
uvx --from git+https://github.com/NavarrePratt/cross-agent-mcp cross-agent-mcp

# opencode + Claude Code backends (adds the optional claude-agent-sdk extra)
uvx --from "git+https://github.com/NavarrePratt/cross-agent-mcp[claude]" cross-agent-mcp
```

The `[claude]` extra pulls in `claude-agent-sdk`, which spawns the `claude` CLI in-process. If it is not installed, only the `opencode_*` tools register and the server still works.

### Prerequisites

- `uv` / `uvx` on your PATH.
- `opencode` on your PATH (for the opencode backend). The server will auto-start `opencode serve` if one is not already running; see [environment variables](#environment-variables).
- `claude` on your PATH with a stored login (for the Claude backend). The SDK inherits the CLI's stored auth - no `ANTHROPIC_API_KEY` is needed.

## Configuration

Register the server as a stdio MCP server in whichever client you use. The examples below install both backends; drop `[claude]` for an opencode-only setup.

### Claude Code

Add to the `mcpServers` block in `~/.claude.json` (or run `claude mcp add`):

```json
{
  "mcpServers": {
    "cross-agent": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--from", "git+https://github.com/NavarrePratt/cross-agent-mcp[claude]", "cross-agent-mcp"],
      "env": {}
    }
  }
}
```

Claude Code exposes the tools as `mcp__cross-agent__opencode_prompt`, `mcp__cross-agent__claude_prompt`, etc.

### Codex

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.cross-agent]
command = "uvx"
args = ["--from", "git+https://github.com/NavarrePratt/cross-agent-mcp[claude]", "cross-agent-mcp"]
```

### OpenCode

Add to the `"mcp"` block in your `opencode.json`:

```json
{
  "mcp": {
    "cross-agent": {
      "type": "stdio",
      "command": ["uvx", "--from", "git+https://github.com/NavarrePratt/cross-agent-mcp[claude]", "cross-agent-mcp"],
      "enabled": true
    }
  }
}
```

### Optional: a read-only opencode agent

By default the opencode backend sends no `agent` field, so `read_only=True` calls use opencode's default agent and rely on the caller's prompt to stay read-only. For real sandboxed enforcement, define a read-only agent in `opencode.json` and point the backend at it:

```json
{
  "agent": {
    "cross-agent-reviewer": {
      "description": "Read-only reviewer for cross-agent adversarial review.",
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
}
```

Then set `CROSS_AGENT_OPENCODE_AGENT=cross-agent-reviewer` (see below). With that set, every `read_only=True` call asks opencode to use that agent.

## Environment variables

### OpenCode backend

| Env var | Default | Purpose |
|---|---|---|
| `CROSS_AGENT_OPENCODE_BASE_URL` | `http://127.0.0.1:4096` | opencode server URL. Include an explicit port when autoserve is enabled. |
| `CROSS_AGENT_OPENCODE_AUTOSERVE` | `true` | Auto-start `opencode serve` if the server is unreachable |
| `CROSS_AGENT_OPENCODE_AGENT` | _(empty)_ | opencode agent name to request for read-only calls; empty = use opencode's default agent |
| `CROSS_AGENT_QUERY_TIMEOUT` | `1200` (20 min) | Query timeout in seconds for both backends |
| `OPENCODE_SERVER_USERNAME` | unset | Optional basic-auth username for the opencode server |
| `OPENCODE_SERVER_PASSWORD` | unset | Optional basic-auth password for the opencode server |

### Claude backend

| Env var | Default | Purpose |
|---|---|---|
| `CROSS_AGENT_CLAUDE_MODEL` | unset (CLI default) | Override the Claude model used for every call |
| `CROSS_AGENT_QUERY_TIMEOUT` | `1200` (20 min) | Query timeout in seconds for both backends |

No auth env vars are needed for the Claude backend; it reuses the `claude` CLI's stored login.

## Tool reference

Each backend registers three tools, prefixed with the backend name: `opencode_prompt` / `opencode_continue` / `opencode_abort` and `claude_prompt` / `claude_continue` / `claude_abort`. The Claude tools only register when the `[claude]` extra is installed.

Errors are returned as `"Error: <Type>: <message>"` text rather than raised, so the calling agent always gets a readable result.

Successful prompt calls return `Session: <native-session-id>` followed by the
response text. Continue calls return only the response text. Abort calls return
`Abort requested for session <native-session-id>.`

### `<prefix>_prompt`

Start a backend-native session. `read_only=True` is the default request mode. Claude enforces this with tool restrictions; opencode hard enforcement requires `CROSS_AGENT_OPENCODE_AGENT` to point at a read-only opencode agent. Returns the response text prefixed with the native session id, which can be passed to `<prefix>_continue` to resume from this MCP server or from the backend harness directly.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | `str` | _(required)_ | The prompt text |
| `read_only` | `bool` | `true` | Request read-only mode. Claude enforces tool restrictions; opencode requires a configured read-only agent for hard enforcement. |
| `directory` | `str \| null` | `null` | Working directory for the call |
| `model` | `str \| null` | `null` | Model override. For opencode, a flat `provider/model` string like `wandb/zai-org/GLM-5.2` (split into `{providerID, modelID}`). For Claude, a model slug like `claude-sonnet-4`. |
| `timeout` | `float \| null` | `null` | Query timeout in seconds; `null`, non-positive, and non-finite values use `CROSS_AGENT_QUERY_TIMEOUT` (package default: 1200s / 20 min). Set high for long reviews. |

### `<prefix>_continue`

Continue an existing backend-native session by id. The id may come from `<prefix>_prompt` or from the underlying backend harness. `read_only=True` is requested by default, matching the prompt tool. Claude enforces that with tool restrictions; opencode hard enforcement still depends on a configured read-only opencode agent.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `session_id` | `str` | _(required)_ | Native backend session id returned from `<prefix>_prompt` or copied from the backend harness |
| `prompt` | `str` | _(required)_ | The follow-up prompt |
| `read_only` | `bool` | `true` | Request read-only mode. Claude enforces tool restrictions; opencode requires a configured read-only agent for hard enforcement. |
| `directory` | `str \| null` | `null` | Working directory for the call |
| `timeout` | `float \| null` | `null` | Query timeout in seconds; `null`, non-positive, and non-finite values use `CROSS_AGENT_QUERY_TIMEOUT` (package default: 1200s / 20 min). |

### `<prefix>_abort`

Request best-effort cancellation of a running session by id. Unknown or already-finished sessions may still call the backend's native stop/abort mechanism, depending on that backend.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `session_id` | `str` | _(required)_ | Session id to abort |

## Usage examples

### Ask GLM-5.2 to review this code (from Claude Code)

```
Call mcp__cross-agent__opencode_prompt with:
- prompt: "Review src/auth.py for security issues. Focus on input validation and the token-refresh flow."
- read_only: true
- directory: "/Users/you/myproject"
```

The response comes back prefixed with the opencode session id. With `CROSS_AGENT_OPENCODE_AGENT` set to a read-only opencode agent, GLM-5.2 reviews with sandboxed read/search access. Without that configuration, the MCP call requests read-only behavior but opencode's default agent permissions still apply.

### Ask Claude for a second opinion (from Codex)

```
Call claude_prompt with:
- prompt: "I'm refactoring the session manager into an async pool. Here's the plan: ... Are there race conditions in this design?"
- read_only: true
- directory: "/Users/you/myproject"
```

Codex invokes the cross-agent Claude backend, which drives `claude-agent-sdk` with your stored Claude login to produce the review.

### Multi-turn conversation with continue

Start a native backend session and capture the returned session id:

```
Call opencode_prompt with:
- prompt: "Audit this branch for missing error handling."
- directory: "/Users/you/myproject"

-> "Session: ses_abc123\n\n<findings>"
```

Then continue the same native session with a follow-up, reusing the session id:

```
Call opencode_continue with:
- session_id: "ses_abc123"
- prompt: "Of those findings, which are critical versus nice-to-have? Justify each."
```

Use `opencode_abort` (or `claude_abort`) to request cancellation of a long-running session early; unknown or already-finished sessions are handled best-effort by the backend's native stop/abort path.

You can also pass a session id copied from the underlying backend harness. For
example, if an opencode session is already running elsewhere, call
`opencode_continue` with that native opencode session id. The MCP server does
not store transcripts or wrap ids; the native backend remains the source of
truth for history and deletion.

There is intentionally no MCP delete or TTL mechanism. Clean up stored
transcripts and session history with the native backend's own tools, because
that is where durable history lives. The MCP server keeps only process-local
maps for in-flight calls so it can cancel active work and serialize concurrent
MCP-originated writes to the same native session. If a prompt creates a native
session and the first message fails, the error includes that native session id
so you can inspect, continue, stop, or delete it using the backend harness.

### Prompting tips

The reviewing agent can explore your codebase on its own. Claude uses restricted read/search tools in `read_only=True`; opencode gets equivalent hard enforcement when you configure a read-only opencode agent. Keep prompts short and let the reviewer form its own understanding.

- **Give paths, not diffs.** Say "Review `src/auth.py` for security issues" with the `directory` parameter set, not "Review this diff: ```diff ... 200 lines ...```". The reviewer reads the files itself.
- **Point at the change set.** For branch reviews, name the files or the commit range ("Review the changes in `src/session/` from the last 3 commits") rather than pasting the diff.
- **Ask focused questions.** "Are there race conditions in the session pool?" gets better results than dumping the code and asking "thoughts?".
- **Use `directory` for every call.** The reviewer needs the working directory to read files; without it, it can't see your tree.

Large prompts waste tokens and context window on both sides. The reviewer is a coding agent with file access, not a chatbot that needs everything pasted in.

## How it works

The server is a thin [FastMCP](https://gofastmcp.com) stdio server. At startup it detects which backends are available and registers their tools:

- **opencode backend** drives a local `opencode serve` HTTP API over `httpx` (default `http://127.0.0.1:4096`). On the first call it health-checks the server and, if `CROSS_AGENT_OPENCODE_AUTOSERVE` is enabled, spawns `opencode serve` as a subprocess it owns and tears it down on shutdown. Read-only calls optionally request a configured read-only opencode agent. The backend leaves successful prompt sessions durable in opencode, sanitizes the `directory` value before it lands in an HTTP header (stripping NUL/CR/LF to prevent header injection), URL-encodes session ids in request paths, and redacts secret-like patterns (`sk-...`, `sk_live_...`, `ghp_...`, `github_pat_...`, `hf_...`, `xox...`, `whsec_...`, long hex/base64 blobs) from response text before it reaches the MCP client.
- **Claude backend** drives `claude-agent-sdk`, which spawns the `claude` CLI in-process and inherits the calling environment, so it reuses the CLI's stored enterprise auth with no API key. Read-only calls restrict the model to `Read`, `Glob`, `Grep`, `LS`, `WebSearch`, `WebFetch` and disallow `Write`, `Edit`, `MultiEdit`, `NotebookEdit`, and `Bash`. Running sessions are tracked by asyncio task so `abort` can cancel an in-flight call, falling back to `claude stop <session_id>` for already-finished sessions.

Prompt calls create backend-native sessions that remain durable in the native
harness. The MCP server only tracks in-flight calls for local cancellation and
serializes MCP-originated requests to the same native session.

## Development

```bash
# install all extras (claude-agent-sdk) + dev deps
uv sync --all-extras

# lint
uv run ruff check

# unit tests (integration tests are auto-skipped without the marker)
uv run pytest

# run the integration tests against real backends
uv run pytest -m integration

# run the live verification script (spawns the installed server, calls both backends)
uv run python scripts/verify_live.py
```

CI (`.github/workflows/ci.yml`) runs `ruff check` and `pytest -m "not integration"` on push and pull requests.
