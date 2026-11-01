"""Command-line tool. Retraction, purging and replay live here so the agent cannot run them."""

import argparse
import json

from beliefs.store import Store, render


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="beliefs")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="run the MCP server on stdio")
    sub.add_parser("show", help="list beliefs, or show one with its evidence").add_argument("id", nargs="?", type=int)
    sub.add_parser("retract", help="stop all evidence from one source counting").add_argument("source_ref")
    sub.add_parser("purge", help="delete one event, then replay").add_argument("event_id", type=int)
    merge = sub.add_parser("merge-labels", help="fold label OLD into label NEW")
    merge.add_argument("old")
    merge.add_argument("new")
    sub.add_parser("replay", help="rebuild every belief from the event log")
    sub.add_parser("conflicts", help="list rival beliefs waiting to be settled")
    settle = sub.add_parser("settle", help="settle a conflict: WINNER holds and LOSER closes")
    settle.add_argument("winner", type=int)
    settle.add_argument("loser", type=int)
    settle.add_argument("--both", action="store_true", help="both hold; neither closes")
    sub.add_parser("sleep", help="consolidate offline: merge duplicates and labels you confirm, generalize across repos")
    a = p.parse_args(argv)

    if a.cmd == "serve":
        from beliefs.server import build  # imports the MCP and Anthropic SDKs only when serving

        build().run()
        return

    store = Store()
    if a.cmd == "show" and a.id is None:
        for b in store.beliefs():
            print(render(b) + ("" if b.status == "open" else f" [{b.status}, replaced by #{b.superseded_by}]"))
    elif a.cmd == "show":
        b = store.get(a.id)
        if b is None:
            raise SystemExit(f"no belief {a.id}")
        print(render(b))
        print(f"  likelihood {b.likelihood:.2f}, firmness {b.firmness:.1f}, status {b.status}")
        print(f"  labels {b.labels}, premises {b.premises}, narrows {b.narrows}")
        for e in store.history(a.id):
            d = json.loads(e["data"])
            detail = f"{d['verdict']} {d['source_kind']} {d['source_ref']}: {d['text']}" if e["kind"] == "evidence" else d["statement"]
            print(f"  event {e['id']} {e['ts']} {e['kind']}: {detail}")
    elif a.cmd == "retract":
        store.retract(a.source_ref)
    elif a.cmd == "purge":
        store.purge(a.event_id)
    elif a.cmd == "merge-labels":
        store.merge_labels(a.old, a.new)
    elif a.cmd == "replay":
        store.replay()
    elif a.cmd == "conflicts":
        for x, y in store.conflicts():
            print(f"{render(x)}\n  vs {render(y)}\n  settle with: beliefs settle {x.id} {y.id}  (or {y.id} {x.id}, or --both)")
    elif a.cmd == "settle":
        store.settle(a.winner, a.loser, None if a.both else a.winner, "cli")
    elif a.cmd == "sleep":
        from beliefs.judge import consolidate, judge

        lines = store.sleep(consolidate, judge, lambda q: input(f"{q}\n[y/N] ").strip().lower() == "y")
        print("\n".join(lines) or "nothing to consolidate")
