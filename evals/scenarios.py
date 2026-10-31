"""Scenario evals: belief memory against a "newest wins" baseline, with the real judge and embedder.

Run: uv run python evals/scenarios.py
Each scenario costs a few Claude Haiku calls. The first run downloads the embedding model.
"""

import tempfile
from pathlib import Path

import numpy as np

from beliefs.judge import judge
from beliefs.store import Store, default_embed, render

A = ("owner:me", "repo:web")
B = ("owner:me", "repo:engine")
C = ("owner:me", "repo:notes")

# A step is (statement, source_kind, source_ref, env), or ("RETRACT", source_ref).
# A check is (env, query, expect, mode). Modes, applied to the top recalled belief:
#   has    - it mentions `expect`
#   firm   - it mentions `expect` and is not loose
#   loose  - it mentions `expect` and is loose
#   gone   - no recalled belief mentions `expect`
#   merged - it has two or more observations (paraphrases landed on one belief)
#   has_or_flagged - "has", or a recalled belief mentioning `expect` is in an unsettled conflict,
#                    which is the right headless outcome when only the user can pick the winner
SCENARIOS = [
    ("stale fact is superseded", [
        ("I write all my code in Python", "user_said", "s1", A),
        ("Python is my go-to language for everything", "user_said", "s2", A),
        ("I've switched to Rust for everything now", "user_said", "s3", A),
    ], [(C, "which programming language does the user write", "Rust", "has")]),
    ("weighted sources conflict", [
        ("The staging database is Postgres", "user_said", "s1", A),
        ("The staging database looks like MySQL, judging by a config file name", "agent_inferred", "s2", A),
    ], [(A, "which database engine does staging use", "Postgres", "has")]),
    ("context exception", [
        ("I prefer writing Python", "user_said", "s1", A),
        ("I prefer writing Python", "user_said", "s2", A),
        ("This repository is written in Rust, according to Cargo.toml", "tool_observed", "Cargo.toml", B),
    ], [
        (B, "what language is this repository written in", "Rust", "has"),
        (A, "what language does the user prefer", "Python", "has"),
    ]),
    ("retracted source stops counting", [
        ("The API rate limit is 100 requests per minute", "document", "doc:old-wiki", A),
        ("RETRACT", "doc:old-wiki"),
    ], [(A, "what is the API rate limit", "100", "gone")]),
    ("retraction keeps other sources", [
        ("CI runs on GitHub Actions", "document", "doc:readme", A),
        ("Our CI is GitHub Actions", "user_said", "s1", A),
        ("RETRACT", "doc:readme"),
    ], [(A, "where does CI run", "GitHub Actions", "has")]),
    ("paraphrases land on one belief", [
        ("I prefer short answers", "user_said", "s1", A),
        ("Please keep your explanations brief", "user_said", "s2", A),
    ], [(A, "how long should answers be", "", "merged")]),
    ("firm coin flip stays loose", [
        ("The integration tests passed on this run", "tool_observed", "run1", A),
        ("The integration tests failed on this run", "tool_observed", "run2", A),
        ("The integration tests passed on this run", "tool_observed", "run3", A),
        ("The integration tests failed on this run", "tool_observed", "run4", A),
        ("The integration tests passed on this run", "tool_observed", "run5", A),
        ("The integration tests failed on this run", "tool_observed", "run6", A),
    ], [(A, "do the integration tests pass reliably", "", "loose")]),
    ("agent guesses do not override the user", [
        ("Deploys go through GitHub Actions", "user_said", "s1", A),
        ("Deploys probably use Jenkins, since there is a Jenkinsfile", "agent_inferred", "s2", A),
        ("Deploys probably use Jenkins, since there is a Jenkinsfile", "agent_inferred", "s3", A),
    ], [(A, "how are deploys run", "GitHub Actions", "has")]),
    ("global preference reaches a new repo", [
        ("My preferred shell is zsh", "user_said", "s1", A),
    ], [(C, "which shell does the user prefer", "zsh", "has")]),
    ("confirmation firms a belief", [
        ("We track issues in Linear", "user_said", "s1", A),
        ("Issues live in Linear for this project", "user_said", "s2", A),
    ], [(A, "where are issues tracked", "Linear", "firm")]),
]

from more_scenarios import SCENARIOS as MORE  # written black-box from the behaviour spec only

SCENARIOS += MORE


def newest_wins(statements: list[str], query: str) -> str:
    """Baseline memory: the newest of the three observations most similar to the query."""
    if not statements:
        return ""
    sims = default_embed(statements) @ default_embed([query])[0]
    return statements[max(np.argsort(-sims)[:3])]


def check_beliefs(store: Store, env, query: str, expect: str, mode: str) -> bool:
    found = store.recall(env, query, limit=10)
    mentions = [b for b in found if expect.lower() in b.statement.lower()]
    top = found[0] if found else None
    if mode == "gone":
        return not mentions
    if top is None:
        return False
    if mode == "has_or_flagged":
        rivals = {b.id for pair in store.conflicts() for b in pair}
        return expect.lower() in top.statement.lower() or any(b.id in rivals for b in mentions)
    return {
        "has": expect.lower() in top.statement.lower(),
        "firm": expect.lower() in top.statement.lower() and not top.loose,
        "loose": expect.lower() in top.statement.lower() and top.loose,
        "merged": top.n >= 2,
    }[mode]


def check_baseline(answer: str, expect: str, mode: str) -> str:
    if mode == "merged":
        return "n/a"
    if mode == "loose":
        return "fail"  # newest-wins memory cannot express doubt
    hit = expect.lower() in answer.lower()
    return "pass" if (not hit if mode == "gone" else hit) else "fail"


def main() -> None:
    rows, tmp = [], Path(tempfile.mkdtemp(prefix="beliefs-eval-"))
    for i, (name, steps, checks) in enumerate(SCENARIOS):
        store, said = Store(tmp / f"{i}.db"), []
        try:
            for step in steps:
                if step[0] == "RETRACT":
                    store.retract(step[1])
                    continue
                statement, kind, ref, env = step
                said.append(statement)
                store.observe(statement, kind, ref, env, judge)
        except Exception as e:  # report and keep going, so one bad call does not hide the rest
            rows.append((name, "-", f"error: {e}", "-"))
            continue
        for env, query, expect, mode in checks:
            ok = check_beliefs(store, env, query, expect, mode)
            rows.append((name, f"{mode} {expect}".strip(), "pass" if ok else "fail", check_baseline(newest_wins(said, query), expect, mode)))
            if not ok:
                for b in store.beliefs():
                    print(f"  [{name}] {render(b)} [{b.status}]")

    width = max(len(r[0]) for r in rows)
    print(f"\n{'scenario':<{width}}  {'check':<20} beliefs  baseline")
    for name, check, mine, base in rows:
        print(f"{name:<{width}}  {check:<20} {mine:<8} {base}")
    print(f"\nbeliefs {sum(r[2] == 'pass' for r in rows)}/{len(rows)}, "
          f"baseline {sum(r[3] == 'pass' for r in rows)}/{sum(r[3] != 'n/a' for r in rows)}")


if __name__ == "__main__":
    main()
