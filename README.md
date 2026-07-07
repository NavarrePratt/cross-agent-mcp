# cross-agent-mcp

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

By default the opencode backend sends no `agent` field, so read-only calls use opencode's default agent and rely on the caller's prompt to stay read-only. For real sandboxed enforcement, define a read-only agent in `opencode.json` and point the backend at it:

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
| `CROSS_AGENT_OPENCODE_BASE_URL` | `http://127.0.0.1:4096` | opencode server URL |
| `CROSS_AGENT_OPENCODE_AUTOSERVE` | `true` | Auto-start `opencode serve` if the server is unreachable |
| `CROSS_AGENT_OPENCODE_AGENT` | _(empty)_ | opencode agent name to request for read-only calls; empty = use opencode's default agent |
| `OPENCODE_SERVER_USERNAME` | unset | Optional basic-auth username for the opencode server |
| `OPENCODE_SERVER_PASSWORD` | unset | Optional basic-auth password for the opencode server |

### Claude backend

| Env var | Default | Purpose |
|---|---|---|
| `CROSS_AGENT_CLAUDE_MODEL` | unset (CLI default) | Override the Claude model used for every call |

No auth env vars are needed for the Claude backend; it reuses the `claude` CLI's stored login.

## Tool reference

Each backend registers three tools, prefixed with the backend name: `opencode_prompt` / `opencode_continue` / `opencode_abort` and `claude_prompt` / `claude_continue` / `claude_abort`. The Claude tools only register when the `[claude]` extra is installed.

Errors are returned as `"Error: <Type>: <message>"` text rather than raised, so the calling agent always gets a readable result.

### `<prefix>_prompt`

Send a one-shot prompt. Read-only by default: the backend can read files, search, and fetch, but not write or execute commands. Returns the response text prefixed with the session id, which can be passed to `<prefix>_continue` to resume.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | `str` | _(required)_ | The prompt text |
| `read_only` | `bool` | `true` | Read-only mode (no write/exec) |
| `directory` | `str \| null` | `null` | Working directory for the call |
| `model` | `str \| null` | `null` | Model override. For opencode, a flat `provider/model` string like `wandb/zai-org/GLM-5.2` (split into `{providerID, modelID}`). For Claude, a model slug like `claude-sonnet-4`. |

### `<prefix>_continue`

Continue an existing session by id. Read-only by default, matching the one-shot tool.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `session_id` | `str` | _(required)_ | Session id returned from `<prefix>_prompt` |
| `prompt` | `str` | _(required)_ | The follow-up prompt |
| `read_only` | `bool` | `true` | Read-only mode |
| `directory` | `str \| null` | `null` | Working directory for the call |

### `<prefix>_abort`

Abort a running session by id. No-op if the session already finished.

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

The response comes back prefixed with the opencode session id. GLM-5.2 runs the review with read-only access to your tree; nothing is written or executed.

### Ask Claude for a second opinion (from Codex)

```
Call claude_prompt with:
- prompt: "I'm refactoring the session manager into an async pool. Here's the plan: ... Are there race conditions in this design?"
- read_only: true
- directory: "/Users/you/myproject"
```

Codex invokes the cross-agent Claude backend, which drives `claude-agent-sdk` with your stored Claude login to produce the review.

### Multi-turn conversation with continue

Start with a one-shot prompt and capture the returned session id:

```
Call opencode_prompt with:
- prompt: "Audit this branch for missing error handling."
- directory: "/Users/you/myproject"

-> "Session: ses_abc123\n\n<findings>"
```

Then continue the same session with a follow-up, reusing the session id:

```
Call opencode_continue with:
- session_id: "ses_abc123"
- prompt: "Of those findings, which are critical versus nice-to-have? Justify each."
```

Use `opencode_abort` (or `claude_abort`) to cancel a long-running session early; it is a no-op if the session already finished.

## How it works

The server is a thin [FastMCP](https://gofastmcp.com) stdio server. At startup it detects which backends are available and registers their tools:

- **opencode backend** drives a local `opencode serve` HTTP API over `httpx` (default `http://127.0.0.1:4096`). On the first call it health-checks the server and, if `CROSS_AGENT_OPENCODE_AUTOSERVE` is enabled, spawns `opencode serve` as a subprocess it owns and tears it down on shutdown. Read-only calls optionally request a configured read-only opencode agent. The backend sanitizes the `directory` value before it lands in an HTTP header (stripping NUL/CR/LF to prevent header injection), URL-encodes session ids in request paths, and redacts secret-like patterns (`sk-...`, `ghp_...`, `hf_...`, `xox...`, `whsec_...`, long hex/base64 blobs) from response text before it reaches the MCP client.
- **Claude backend** drives `claude-agent-sdk`, which spawns the `claude` CLI in-process and inherits the calling environment, so it reuses the CLI's stored enterprise auth with no API key. Read-only calls restrict the model to `Read`, `Glob`, `Grep`, `LS`, `WebSearch`, `WebFetch` and disallow `Write`, `Edit`, `MultiEdit`, `NotebookEdit`, and `Bash`. Running sessions are tracked by asyncio task so `abort` can cancel an in-flight call, falling back to `claude stop <session_id>` for already-finished sessions.

One-shot prompts create an ephemeral session that is cleaned up after the response; `continue` keeps the session alive across turns.

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

## License

MIT. See [LICENSE](LICENSE).
