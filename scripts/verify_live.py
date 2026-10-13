"""Live verification client for cross-agent-mcp.

Spawns the real installed server via `uvx --from . cross-agent-mcp`, lists tools,
and optionally calls opencode_prompt + claude_prompt expecting real responses.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def main(*, startup_only: bool = False) -> None:
    uvx_args = ["--from", ".[claude]", "cross-agent-mcp"]
    if startup_only:
        uvx_args.insert(0, "--refresh")
    params = StdioServerParameters(
        command="uvx",
        args=uvx_args,
        cwd=REPO,
        # Pass the full user env so the server (and any opencode serve it spawns)
        # inherits provider credentials like WANDB_API_KEY that opencode reads.
        env=os.environ,
    )
    print(f">> launching: uvx {' '.join(uvx_args)}", flush=True)
    async with stdio_client(params, errlog=sys.stderr) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            names = [t.name for t in tools]
            print(f">> registered tools ({len(names)}): {names}", flush=True)

            assert "opencode_prompt" in names, "opencode_prompt missing"
            assert "opencode_continue" in names
            assert "opencode_abort" in names

            if startup_only:
                print(">> STARTUP VERIFICATION PASSED", flush=True)
                return

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

            assert "codex_prompt" in names, "codex_prompt missing (codex CLI not on PATH?)"
            print(">> calling codex_prompt ...", flush=True)
            cx = await session.call_tool(
                "codex_prompt",
                {"prompt": "Remember the word PONG. Reply with exactly: OK", "directory": REPO},
            )
            cx_text = "".join(b.text for b in cx.content if getattr(b, "type", "") == "text")
            print(f">> codex_prompt result:\n{cx_text}\n", flush=True)
            assert "Error:" not in cx_text, "codex_prompt returned an error"
            codex_session = cx_text.split("\n", 1)[0].removeprefix("Session: ")

            print(">> calling codex_continue ...", flush=True)
            cc = await session.call_tool(
                "codex_continue",
                {
                    "session_id": codex_session,
                    "prompt": "Which word did I ask you to remember? Reply with just the word.",
                    "directory": REPO,
                },
            )
            cc_text = "".join(b.text for b in cc.content if getattr(b, "type", "") == "text")
            print(f">> codex_continue result:\n{cc_text}\n", flush=True)
            assert "Error:" not in cc_text, "codex_continue returned an error"
            assert "PONG" in cc_text, "codex_continue did not resume the session"

    print(">> VERIFICATION PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--startup-only",
        action="store_true",
        help="Freshly resolve the package, initialize the server, and list its tools.",
    )
    args = parser.parse_args()
    try:
        asyncio.run(main(startup_only=args.startup_only))
    except Exception as exc:
        print(f">> VERIFICATION FAILED: {exc}", file=sys.stderr, flush=True)
        raise
