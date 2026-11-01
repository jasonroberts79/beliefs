"""MCP server: `observe` and `recall` tools, plus a startup digest of firm beliefs."""

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from mcp import MCPError
from mcp.server.mcpserver import Context, MCPServer
from pydantic import create_model

from beliefs.judge import judge
from beliefs.store import HOME, Belief, Store, Unavailable, render

# Each Claude Code session starts its own stdio server, so this names the session.
SESSION = f"session:{datetime.now(timezone.utc).isoformat(timespec='seconds')}"

GUIDE = """Belief memory: what you learned about the user and their projects in earlier sessions.
Each belief is hedged by how likely it is. A belief marked "verify before acting on this" is loosely held, so check it before relying on it.
A belief marked "conflicts with #N" has a rival the user has not settled yet; do not act on either one without checking.

Call `observe` with one plain statement whenever:
- the user states a fact or a preference;
- a tool result confirms or contradicts something you believed;
- you learn something durable about this repository's tooling or conventions.
Call `recall` before relying on memory for anything not listed below.

Current context: {context}
Firm beliefs here:
{digest}"""


def _git(path: Path, *args: str) -> str | None:
    r = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def _root(path: Path) -> str | None:
    """The repo's first commit as a context element. It survives renames and moves, unlike the folder name."""
    roots = _git(path, "rev-list", "--max-parents=0", "HEAD")  # merged histories have several, newest first
    return f"repo:{roots.splitlines()[-1][:12]}" if roots else None


def environment(path=None, config: Path = HOME / "config.json") -> tuple[str, ...]:
    """Context tuple for a folder: optional heads from the folder map, then owner and repo."""
    path = Path(path or os.getcwd()).resolve()
    heads: list[str] = []
    if config.exists():
        folders = json.loads(config.read_text()).get("folders", {})
        matches = [(Path(k).expanduser().resolve(), v) for k, v in folders.items()]
        matches = [(k, v) for k, v in matches if k == path or k in path.parents]
        if matches:
            heads = max(matches, key=lambda kv: len(kv[0].parts))[1]
    if m := re.search(r"[:/]([^/:]+)/([^/]+?)(?:\.git)?/?$", _git(path, "remote", "get-url", "origin") or ""):
        return (*heads, f"owner:{m[1]}", f"repo:{m[2]}")
    return (*heads, _root(path) or f"repo:{Path(_git(path, 'rev-parse', '--show-toplevel') or path).name}")


def _lines(beliefs: list[Belief], conflicts: list[tuple[Belief, Belief]]) -> list[str]:
    rivals: dict[int, list[int]] = {}
    for a, b in conflicts:
        rivals.setdefault(a.id, []).append(b.id)
        rivals.setdefault(b.id, []).append(a.id)
    return [render(b) + "".join(f" - conflicts with #{r}, unsettled" for r in rivals.get(b.id, [])) for b in beliefs]


async def ask_user(ctx: Context, a: Belief, b: Belief) -> int | None | bool:
    """Ask the user which rival holds. Returns a belief id, None for both, or False to settle later."""
    caps = ctx.request_context.session.client_capabilities
    if caps is None or caps.elicitation is None:
        return False  # this client cannot ask anyone
    first, second = f"#{a.id}: {a.statement}", f"#{b.id}: {b.statement}"
    Answer = create_model("Answer", holds=(Literal[first, second, "Both hold", "Not sure"], ...))
    try:
        result = await ctx.elicit("Two remembered beliefs conflict. Which one holds?", schema=Answer)
    except (MCPError, ValueError):
        return False
    if result.action != "accept" or result.data.holds == "Not sure":
        return False  # a cancel is what headless sessions send, so this queues the conflict
    return {first: a.id, second: b.id, "Both hold": None}[result.data.holds]


def build() -> MCPServer:
    store = Store()
    env = environment()
    if (root := _root(Path.cwd())) and root not in env:  # the repo has gained a remote since
        store.move_context(root, list(env[-2:]))
    asked: set[frozenset[int]] = set()  # never ask about the same pair twice in one session
    digest = "\n".join(f"- {line}" for line in _lines(store.digest(env), store.conflicts())) or "(none yet)"
    server = MCPServer("beliefs", instructions=GUIDE.format(context="/".join(env), digest=digest))

    @server.tool()
    def observe(
        statement: str,
        source_kind: Literal["user_said", "tool_observed", "document", "agent_inferred"] = "user_said",
        source_ref: str | None = None,
        premises: list[int] | None = None,
    ) -> str:
        """Record one piece of evidence about the user, their projects or their tools.

        statement: one plain fact, such as "This repo runs tests with pytest".
        source_kind: user_said for what the user told you, tool_observed for what a tool showed,
        document for what a file or page claims, agent_inferred for your own conclusion.
        source_ref: a file path or URL for documents and tools; defaults to this session.
        premises: belief ids an agent_inferred statement was derived from.
        """
        return store.observe(statement, source_kind, source_ref or SESSION, env, judge, premises or [])

    @server.tool()
    async def recall(query: str, ctx: Context, labels: list[str] | None = None) -> str:
        """Find beliefs related to a topic in the current context, most relevant first.

        labels: optionally keep only beliefs carrying one of these labels.
        """
        try:
            notes = store.drain(judge)  # statements queued during an outage count before this recall
        except Unavailable:
            waiting = store.db.execute("SELECT COUNT(*) FROM pending").fetchone()[0]
            notes = [f"{waiting} recent statement(s) wait for the judge, so newer facts may be missing"]
        found = store.recall(env, query, labels)
        ids = {b.id for b in found}
        for a, b in store.conflicts():
            pair = frozenset((a.id, b.id))
            if pair <= ids and pair not in asked:
                asked.add(pair)
                answer = await ask_user(ctx, a, b)
                if answer is not False:
                    store.settle(a.id, b.id, answer, SESSION)
                    found = store.recall(env, query, labels)
        return "\n".join([*_lines(found, store.conflicts()), *notes]) or "No beliefs match."

    return server
