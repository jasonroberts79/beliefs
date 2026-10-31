"""The judge: a small model decides how one observation bears on existing beliefs."""

import json
import os
from functools import cache
from typing import Literal

import anthropic
from pydantic import BaseModel

MODEL = os.environ.get("BELIEFS_JUDGE_MODEL", "claude-haiku-5-5")

Action = Literal["supports", "contradicts", "supersedes", "carve", "new"]


class Effect(BaseModel):
    action: Action
    belief_id: int | None = None
    statement: str | None = None
    context: list[str] | None = None
    labels: list[str] | None = None


class Verdict(BaseModel):
    reason: str = ""  # first, so the model reasons before it commits to any effect
    effects: list[Effect] = []  # empty means the observation is irrelevant


SYSTEM = """You maintain an agent's long-term belief store. Each belief is one short, general \
proposition about the user, their projects, or their tools. Given one new observation and the \
most similar existing beliefs, decide how the observation bears on them.

Return a list of effects. Most observations have exactly one. Use two when the observation both \
rules out a belief and asserts an alternative: "this repo uses pnpm, not npm" contradicts the npm \
belief and supports or creates a pnpm belief. Never add an effect for a belief the observation \
only loosely touches. Return an empty list when the observation says nothing durable about the \
user, their projects or their tools.

Actions for each effect:
- supports: the observation is evidence that candidate `belief_id` is true.
- contradicts: the observation is evidence that candidate `belief_id` is false in the same context.
- supersedes: the world changed. Candidate `belief_id` was true but no longer is, for example \
"I switched from X to Y". Give the new belief as `statement`.
- carve: candidate `belief_id` still holds in general, but the observation shows it does not hold \
in a narrower context. Give the narrower belief as `statement` and a `context` longer than the candidate's.
- new: no candidate is about the same proposition. Give `statement`, `context` and `labels`.

Prefer an existing candidate whenever it states the same proposition in other words, even if its \
context is broader or narrower than you would pick. A single event, such as "the tests passed on \
this run", is evidence for the durable proposition it bears on, such as "the tests pass reliably".

Statements: one sentence, present tense, general enough to be confirmed again later. No dates, \
no session details, no hedging words.

Context: a prefix of the environment tuple. Use [] only for the user's personal preferences and \
habits, which hold everywhere. Use the full environment for facts about the current repository or \
team, such as tooling, services, layout and conventions. "We" and "our" usually mean the team or \
project, not the user personally. Use an intermediate prefix only when the observation \
says the fact holds for that whole group.

Labels: one to three short lowercase topic labels. Reuse an existing label whenever one fits. \
Create a new label only when none fits.

Reason: write one short sentence first, then list the effects that match it."""


@cache
def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def judge(statement: str, source_kind: str, env: tuple[str, ...], candidates: list, labels: list[str]) -> Verdict:
    payload = {
        "observation": statement,
        "source_kind": source_kind,
        "environment": list(env),
        "candidates": [
            {"belief_id": b.id, "statement": b.statement, "context": list(b.context), "labels": b.labels}
            for b in candidates
        ],
        "existing_labels": labels,
    }
    response = _client().messages.parse(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": json.dumps(payload, indent=1)}],
        output_format=Verdict,
    )
    if response.stop_reason != "end_turn" or response.parsed_output is None:
        raise RuntimeError(f"judge stopped with {response.stop_reason} (request {response._request_id})")
    return response.parsed_output
