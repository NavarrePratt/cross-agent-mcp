"""Live verification client for cross-agent-mcp.

Spawns the real installed server via `uvx --from . cross-agent-mcp`, lists tools,
and calls opencode_prompt + claude_prompt expecting real responses.
"""

from __future__ import annotations

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def main() -> None:
    params = StdioServerParameters(
        command="uvx",
        args=["--from", ".[claude]", "cross-agent-mcp"],
        cwd=REPO,
        # Pass the full user env so the server (and any opencode serve it spawns)
        # inherits provider credentials like WANDB_API_KEY that opencode reads.
        env=os.environ,
    )
    print(">> launching: uvx --from . cross-agent-mcp", flush=True)
    async with stdio_client(params, errlog=sys.stderr) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            names = [t.name for t in tools]
            print(f">> registered tools ({len(names)}): {names}", flush=True)

            assert "opencode_prompt" in names, "opencode_prompt missing"
            assert "opencode_continue" in names
            assert "opencode_abort" in names

            print(">> calling opencode_prompt ...", flush=True)
            oc = await session.call_tool("opencode_prompt", {"prompt": "Reply with exactly: PONG"})
            oc_text = "".join(b.text for b in oc.content if getattr(b, "type", "") == "text")
            print(f">> opencode_prompt result:\n{oc_text}\n", flush=True)
            assert "Error:" not in oc_text, "opencode_prompt returned an error"
            assert "PONG" in oc_text, "opencode_prompt did not return a real response"

            assert "claude_prompt" in names, (
                "claude_prompt missing (claude-agent-sdk extra not installed?)"
            )
            print(">> calling claude_prompt ...", flush=True)
            cl = await session.call_tool("claude_prompt", {"prompt": "Reply with exactly: PONG"})
            cl_text = "".join(b.text for b in cl.content if getattr(b, "type", "") == "text")
            print(f">> claude_prompt result:\n{cl_text}\n", flush=True)
            assert "Error:" not in cl_text, "claude_prompt returned an error"
            assert "PONG" in cl_text, "claude_prompt did not return a real response"

    print(">> VERIFICATION PASSED", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f">> VERIFICATION FAILED: {exc}", file=sys.stderr, flush=True)
        raise
