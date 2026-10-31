"""Event-sourced belief store.

Every observation is appended to `events`. The `beliefs` table is a snapshot built by
applying those events in order, so `replay()` can always rebuild it from scratch.
"""

import json
import os
import sqlite3
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import numpy as np

PRIOR = 0.5  # support and against counts of a belief before any evidence
WEIGHTS = {"user_said": 1.0, "tool_observed": 1.0, "document": 0.5, "agent_inferred": 0.2}
STRONG = {"user_said", "tool_observed"}  # the only sources that may close or override a belief
CANDIDATES = 8  # beliefs the judge sees per observation
TIE = 0.05  # ponytail: fixed recall tie margin; tune from real recall logs
DIGEST = 20  # firm beliefs in the startup digest; Claude Code truncates long instructions
HOME = Path(os.environ.get("BELIEFS_HOME", Path.home() / ".beliefs"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,           -- created | evidence | label_merged | retracted | settled
    belief_id INTEGER,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_belief ON events(belief_id);
CREATE TABLE IF NOT EXISTS beliefs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    statement TEXT NOT NULL,
    context TEXT NOT NULL,
    labels TEXT NOT NULL,
    premises TEXT NOT NULL,
    narrows INTEGER,              -- the general belief this one carves an exception from
    support REAL NOT NULL,
    against REAL NOT NULL,
    n INTEGER NOT NULL,           -- evidence events applied
    status TEXT NOT NULL,         -- open | closed
    superseded_by INTEGER,
    embedding BLOB
);
"""


@dataclass
class Belief:
    id: int
    statement: str
    context: tuple[str, ...]
    labels: list[str]
    premises: list[int]
    narrows: int | None
    support: float
    against: float
    n: int
    status: str
    superseded_by: int | None
    embedding: np.ndarray | None

    @property
    def firmness(self) -> float:
        return self.support + self.against

    @property
    def likelihood(self) -> float:
        return self.support / self.firmness

    @property
    def loose(self) -> bool:
        return self.firmness < 3 or 0.3 <= self.likelihood <= 0.7

    def applies(self, env: tuple[str, ...]) -> bool:
        return env[: len(self.context)] == self.context


def render(b: Belief) -> str:
    p = b.likelihood
    hedge = (
        "almost certainly" if p >= 0.9 else
        "probably" if p >= 0.7 else
        "unsure whether" if p > 0.3 else
        "probably not" if p > 0.1 else
        "almost certainly not"
    )
    where = "/".join(b.context) or "everywhere"
    text = f"#{b.id} {hedge}: {b.statement} ({b.n} observation{'' if b.n == 1 else 's'}; {where})"
    return text + (" - verify before acting on this" if b.loose else "")


@cache
def _model():
    from fastembed import TextEmbedding  # slow import; only paid when embedding is needed

    return TextEmbedding("BAAI/bge-small-en-v1.5", cache_dir=str(HOME / "models"))


def default_embed(texts: list[str]) -> np.ndarray:
    vecs = np.array(list(_model().embed(texts)), dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _prefix(ctx: list[str], env: tuple[str, ...]) -> tuple[str, ...]:
    """The longest prefix of env that ctx agrees with."""
    n = 0
    while n < min(len(ctx), len(env)) and ctx[n] == env[n]:
        n += 1
    return env[:n]


class Store:
    def __init__(self, path=None, embed=default_embed):
        path = Path(path or os.environ.get("BELIEFS_DB", HOME / "beliefs.db"))
        path.parent.mkdir(parents=True, exist_ok=True)
        # Tools may run on worker threads; every write still goes through _tx.
        self.db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript(SCHEMA)
        self.embed = embed

    @contextmanager
    def _tx(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.db.execute("COMMIT")

    # --- reading the snapshot ---

    def beliefs(self, where: str = "1", params: tuple = ()) -> list[Belief]:
        rows = self.db.execute(f"SELECT * FROM beliefs WHERE {where} ORDER BY id", params).fetchall()
        return [
            Belief(
                id=r["id"], statement=r["statement"], context=tuple(json.loads(r["context"])),
                labels=json.loads(r["labels"]), premises=json.loads(r["premises"]), narrows=r["narrows"],
                support=r["support"], against=r["against"], n=r["n"], status=r["status"],
                superseded_by=r["superseded_by"],
                embedding=np.frombuffer(r["embedding"], dtype=np.float32) if r["embedding"] else None,
            )
            for r in rows
        ]

    def get(self, belief_id: int) -> Belief | None:
        found = self.beliefs("id = ?", (belief_id,))
        return found[0] if found else None

    def labels(self) -> list[str]:
        return sorted({label for b in self.beliefs("status = 'open'") for label in b.labels})

    def history(self, belief_id: int) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM events WHERE belief_id = ? ORDER BY id", (belief_id,)).fetchall()

    def applicable(self, env: tuple[str, ...]) -> list[Belief]:
        """Open beliefs with evidence whose context is a prefix of env, minus overridden ones."""
        found = [b for b in self.beliefs("status = 'open' AND n > 0") if b.applies(env)]
        overridden = {b.narrows for b in found}  # a carved exception beats the general belief
        return [b for b in found if b.id not in overridden]

    def similar(self, vec: np.ndarray, beliefs: list[Belief], k: int) -> list[Belief]:
        # ponytail: brute-force cosine; switch to sqlite-vec past ~50k beliefs.
        beliefs = [b for b in beliefs if b.embedding is not None]
        if not beliefs:
            return []
        sims = np.stack([b.embedding for b in beliefs]) @ vec
        return [beliefs[i] for i in np.argsort(-sims)[:k]]

    def recall(self, env, query: str | None = None, labels: list[str] | None = None, limit: int = 10) -> list[Belief]:
        found = self.applicable(env)
        if labels:
            found = [b for b in found if set(b.labels) & set(labels)]
        found = [b for b in found if b.embedding is not None]
        if not query or not found:
            return sorted(found, key=lambda b: -b.firmness)[:limit]
        sims = np.stack([b.embedding for b in found]) @ self.embed([query])[0]
        near = sims.max() - TIE
        # Near-ties with the best match go firmest first; everything else by similarity.
        order = sorted(range(len(found)), key=lambda i: (0, -found[i].firmness) if sims[i] >= near else (1, -sims[i]))
        return [found[i] for i in order[:limit]]

    def digest(self, env) -> list[Belief]:
        return [b for b in self.recall(env, limit=10**6) if not b.loose][:DIGEST]

    # --- writing ---

    def observe(self, statement, source_kind, source_ref, env, judge, premises=()) -> str:
        if source_kind not in WEIGHTS:
            raise ValueError(f"source_kind must be one of {sorted(WEIGHTS)}")
        env = tuple(env)
        pool = [b for b in self.beliefs("status = 'open' AND n > 0") if b.applies(env)]
        candidates = self.similar(self.embed([statement])[0], pool, CANDIDATES)
        v = judge(statement, source_kind, env, candidates, self.labels())
        # `obs` ties together the events of one observation, which is how rival beliefs are found.
        evidence = {"source_kind": source_kind, "source_ref": source_ref, "text": statement,
                    "reason": v.reason, "obs": uuid.uuid4().hex}

        # Turn each effect into (action, target, created), where created describes a new belief.
        by_id, touched, plans = {b.id: b for b in candidates}, set(), []
        for e in v.effects:
            target, action = by_id.get(e.belief_id), e.action
            if action in ("supports", "contradicts", "supersedes", "carve") and target is None:
                action = "new"  # the judge named a belief it was not shown
            if action == "carve" and len(env) <= len(target.context):
                action = "contradicts"  # no narrower context exists here
            # A document or a guess must not close or override a belief on its own. It contradicts
            # the belief at its own weight and starts a rival, so the user settles the pair.
            rival = action in ("supersedes", "carve") and source_kind not in STRONG
            if rival:
                if target.id not in touched:
                    touched.add(target.id)
                    plans.append(("contradicts", target, None))
            elif target is not None and action in ("supports", "contradicts", "supersedes"):
                if target.id in touched:
                    continue  # one observation counts once per belief
                touched.add(target.id)
            if action in ("supports", "contradicts"):
                plans.append((action, target, None))
                continue
            # new, carve and supersedes all start a new belief.
            if action == "supersedes":
                ctx, labels = target.context, e.labels or target.labels
            else:
                ctx, labels = _prefix(e.context or [], env), e.labels or []
                if action == "carve" and len(ctx) <= len(target.context):
                    ctx = env
            plans.append(("rival" if rival else action, target, {
                "statement": e.statement or statement, "context": list(ctx), "labels": labels,
                "premises": list(premises), "narrows": target.id if action == "carve" and not rival else None,
            }))

        new_texts = [c["statement"] for _, _, c in plans if c]
        vecs = iter(self.embed(new_texts) if new_texts else [])
        results = []  # (how, belief_id)
        with self._tx():
            if not plans:
                self._record("evidence", None, {**evidence, "verdict": "irrelevant"})
            for action, target, created in plans:
                if created is None:
                    self._record("evidence", target.id, {**evidence, "verdict": action})
                    results.append((action, target.id))
                    continue
                seq = self.db.execute("SELECT seq FROM sqlite_sequence WHERE name = 'beliefs'").fetchone()
                new_id = (seq[0] if seq else 0) + 1  # AUTOINCREMENT never reuses a purged id
                self._record("created", new_id, created, next(vecs).tobytes())
                if action == "supersedes":
                    self._record("evidence", new_id, {**evidence, "verdict": "supersedes", "old": target.id})
                else:
                    self._record("evidence", new_id, {**evidence, "verdict": "supports"})
                verb = {"carve": "carved from", "supersedes": "supersedes", "rival": "rival to"}.get(action)
                how = f"{verb} #{target.id}" if verb else "new"
                results.append((how, new_id))
        return "\n".join(f"{how}: {render(self.get(i))}" for how, i in results) or "irrelevant: no belief changed"

    def _record(self, kind: str, belief_id: int | None, data: dict, embedding: bytes | None = None) -> int:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cur = self.db.execute(
            "INSERT INTO events (ts, kind, belief_id, data) VALUES (?, ?, ?, ?)",
            (ts, kind, belief_id, json.dumps(data)),
        )
        self._apply(kind, belief_id, data, embedding)
        return cur.lastrowid

    def _retracted(self, source_ref: str) -> bool:
        return self.db.execute(
            "SELECT 1 FROM events WHERE kind = 'retracted' AND json_extract(data, '$.source_ref') = ? LIMIT 1",
            (source_ref,),
        ).fetchone() is not None

    def _apply(self, kind: str, belief_id: int | None, data: dict, embedding: bytes | None = None) -> None:
        """Project one event onto the snapshot. Live writes and replay share this path."""
        if kind == "created":
            self.db.execute(
                """INSERT INTO beliefs (id, statement, context, labels, premises, narrows,
                                        support, against, n, status, superseded_by, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 'open', NULL, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     statement = excluded.statement, context = excluded.context, labels = excluded.labels,
                     premises = excluded.premises, narrows = excluded.narrows, support = excluded.support,
                     against = excluded.against, n = 0, status = 'open', superseded_by = NULL,
                     embedding = COALESCE(excluded.embedding, beliefs.embedding)""",
                (belief_id, data["statement"], json.dumps(data["context"]), json.dumps(data["labels"]),
                 json.dumps(data["premises"]), data["narrows"], PRIOR, PRIOR, embedding),
            )
        elif kind == "evidence":
            verdict = data["verdict"]
            if verdict == "irrelevant" or self._retracted(data["source_ref"]):
                return
            column = "against" if verdict == "contradicts" else "support"
            self.db.execute(
                f"UPDATE beliefs SET {column} = {column} + ?, n = n + 1 WHERE id = ?",
                (WEIGHTS[data["source_kind"]], belief_id),
            )
            if verdict == "supersedes":
                self.db.execute(
                    "UPDATE beliefs SET status = 'closed', superseded_by = ? WHERE id = ?", (belief_id, data["old"])
                )
        elif kind == "label_merged":
            for row in self.db.execute("SELECT id, labels FROM beliefs").fetchall():
                labels = json.loads(row["labels"])
                if data["old"] in labels:
                    merged = list(dict.fromkeys(data["new"] if x == data["old"] else x for x in labels))
                    self.db.execute("UPDATE beliefs SET labels = ? WHERE id = ?", (json.dumps(merged), row["id"]))
        elif kind == "settled":
            if data["winner"] is not None and not self._retracted(data["source_ref"]):
                loser = data["b"] if data["winner"] == data["a"] else data["a"]
                self.db.execute(
                    "UPDATE beliefs SET status = 'closed', superseded_by = ? WHERE id = ?", (data["winner"], loser)
                )
        # "retracted" changes nothing here: _retracted() makes every apply of that source a no-op.

    # --- conflicts between rival beliefs ---

    def conflicts(self) -> list[tuple[Belief, Belief]]:
        """Open belief pairs that one observation set against each other and nobody has settled."""
        # ponytail: scans every event per call; index rivals in their own table if this gets slow.
        retracted = {r[0] for r in self.db.execute(
            "SELECT json_extract(data, '$.source_ref') FROM events WHERE kind = 'retracted'")}
        sides = defaultdict(lambda: (set(), set()))  # obs -> (contradicted ids, supported ids)
        settled = set()
        for e in self.db.execute("SELECT kind, belief_id, data FROM events WHERE kind IN ('evidence', 'settled')"):
            d = json.loads(e["data"])
            if d["source_ref"] in retracted:
                continue
            if e["kind"] == "settled":
                settled.add(frozenset((d["a"], d["b"])))
            elif d.get("obs") and d["verdict"] in ("contradicts", "supports"):
                sides[d["obs"]][d["verdict"] == "supports"].add(e["belief_id"])
        pairs = {frozenset((x, y)) for against, favour in sides.values() for x in against for y in favour if x != y}
        live = {b.id: b for b in self.beliefs("status = 'open' AND n > 0")}
        return [tuple(live[i] for i in sorted(p)) for p in pairs - settled if all(i in live for i in p)]

    def settle(self, a: int, b: int, winner: int | None, source_ref: str) -> None:
        """Record the user's answer: `winner` holds and the other closes, or None when both hold."""
        if winner not in (a, b, None):
            raise ValueError("winner must be one of the two beliefs, or None for both")
        evidence = {"source_kind": "user_said", "source_ref": source_ref, "text": "settled a conflict",
                    "reason": "the user settled a conflict", "verdict": "supports", "obs": uuid.uuid4().hex}
        with self._tx():
            for belief_id in [winner] if winner else [a, b]:
                self._record("evidence", belief_id, evidence)
            self._record("settled", None, {"a": a, "b": b, "winner": winner, "source_ref": source_ref})

    # --- exceptional operations (command line only) ---

    def _replay(self) -> None:
        for e in self.db.execute("SELECT kind, belief_id, data FROM events ORDER BY id").fetchall():
            self._apply(e["kind"], e["belief_id"], json.loads(e["data"]))

    def replay(self) -> None:
        with self._tx():
            self._replay()

    def retract(self, source_ref: str) -> None:
        with self._tx():
            self._record("retracted", None, {"source_ref": source_ref})
            self._replay()

    def purge(self, event_id: int) -> None:
        """Delete one event. Purging a 'created' event deletes that belief and all its events."""
        with self._tx():
            event = self.db.execute("SELECT kind, belief_id FROM events WHERE id = ?", (event_id,)).fetchone()
            if event is None:
                raise ValueError(f"no event {event_id}")
            if event["kind"] == "created":
                self.db.execute("DELETE FROM events WHERE belief_id = ?", (event["belief_id"],))
                self.db.execute("DELETE FROM beliefs WHERE id = ?", (event["belief_id"],))
            else:
                self.db.execute("DELETE FROM events WHERE id = ?", (event_id,))
            self._replay()

    def merge_labels(self, old: str, new: str) -> None:
        with self._tx():
            self._record("label_merged", None, {"old": old, "new": new})
