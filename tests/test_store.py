"""Store logic with a scripted judge and a bag-of-words embedder, so no model or network is needed."""

import hashlib
import subprocess

import numpy as np
import pytest

from beliefs.judge import Effect, Verdict
from beliefs.server import environment
from beliefs.store import PRIOR, Store

ENV = ("owner:me", "repo:a")
OTHER = ("owner:me", "repo:b")


def fake_embed(texts):
    out = np.zeros((len(texts), 64), np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def says(*effects, **one):
    """A judge that always returns these effects: says(action=...) for one, says(dict, dict) for several."""
    return lambda *args: Verdict(effects=[Effect(**e) for e in (effects or ([one] if one else []))])


@pytest.fixture
def s(tmp_path):
    return Store(tmp_path / "b.db", embed=fake_embed)


def new_python(s, ref="s1", env=ENV):
    return s.observe("I write Python", "user_said", ref, env, says(action="new", statement="User writes Python", context=[], labels=["language"]))


def snapshot(s):
    cols = "id, statement, context, labels, support, against, n, status, superseded_by"
    return [tuple(r) for r in s.db.execute(f"SELECT {cols} FROM beliefs ORDER BY id")]


def test_beta_math_and_source_weights(s):
    new_python(s)
    b = s.get(1)
    assert (b.likelihood, b.firmness, b.loose) == (0.75, 2.0, True)  # one statement stays loose
    s.observe("I write Python", "user_said", "s2", ENV, says(action="supports", belief_id=1))
    assert s.get(1).loose is False and s.get(1).firmness == 3.0
    s.observe("looks like Rust", "agent_inferred", "s3", ENV, says(action="contradicts", belief_id=1))
    assert s.get(1).against == pytest.approx(0.7)  # agent inference weighs 0.2


def test_firm_coin_flip_is_loose(s):
    new_python(s)
    for i in range(4):
        s.observe("x", "user_said", f"c{i}", ENV, says(action="contradicts", belief_id=1))
        s.observe("x", "user_said", f"p{i}", ENV, says(action="supports", belief_id=1))
    b = s.get(1)
    assert b.firmness > 9 and 0.3 <= b.likelihood <= 0.7 and b.loose


def test_supersede_closes_old_and_recall_shows_new(s):
    new_python(s)
    out = s.observe("I switched to Rust", "user_said", "s2", ENV, says(action="supersedes", belief_id=1, statement="User writes Rust"))
    assert out.startswith("supersedes #1")
    assert (s.get(1).status, s.get(1).superseded_by) == ("closed", 2)
    assert [b.statement for b in s.recall(ENV)] == ["User writes Rust"]
    assert s.get(2).labels == ["language"]  # inherited from the superseded belief


def test_carve_overrides_only_in_the_narrower_context(s):
    new_python(s)
    s.observe("this repo is Rust", "tool_observed", "t1", ENV, says(action="carve", belief_id=1, statement="This repo is written in Rust", context=list(ENV)))
    assert s.get(2).narrows == 1 and s.get(2).context == ENV
    assert [b.id for b in s.recall(ENV)] == [2]  # longest matching context wins
    assert [b.id for b in s.recall(OTHER)] == [1]  # elsewhere the general belief still holds
    assert s.get(1).firmness == 2.0  # carving does not weaken the general belief


def test_context_is_clipped_to_a_prefix_of_the_environment(s):
    s.observe("uses uv", "tool_observed", "t1", ENV, says(action="new", statement="Repo uses uv", context=["owner:me", "repo:zzz"]))
    assert s.get(1).context == ("owner:me",)


def test_unknown_belief_id_falls_back_to_new(s):
    s.observe("tabs", "user_said", "s1", ENV, says(action="supports", belief_id=99, statement="User prefers tabs"))
    assert s.get(1).statement == "User prefers tabs" and s.get(1).n == 1


def test_one_observation_contradicts_and_asserts_an_alternative(s):
    s.observe("install with npm", "document", "README.md", ENV, says(action="new", statement="Repo uses npm", context=list(ENV)))
    out = s.observe("pnpm-lock.yaml, no package-lock", "tool_observed", "t1", ENV, says(
        dict(action="contradicts", belief_id=1),
        dict(action="new", statement="Repo uses pnpm", context=list(ENV)),
    ))
    assert out.count("\n") == 1  # one line per effect
    assert s.get(1).against == 1.5 and s.get(2).statement == "Repo uses pnpm"
    assert [e["belief_id"] for e in s.db.execute("SELECT belief_id FROM events WHERE kind = 'evidence'")] == [1, 1, 2]


def rivals(s):
    s.observe("install with npm", "document", "README.md", ENV, says(action="new", statement="Repo uses npm", context=list(ENV)))
    s.observe("pnpm-lock.yaml", "tool_observed", "t1", ENV, says(
        dict(action="contradicts", belief_id=1), dict(action="new", statement="Repo uses pnpm", context=list(ENV))))


def test_contradict_plus_support_makes_rivals_until_settled(s):
    rivals(s)
    assert [(a.id, b.id) for a, b in s.conflicts()] == [(1, 2)]
    s.settle(1, 2, 2, "s9")
    assert s.conflicts() == [] and s.get(1).status == "closed" and s.get(2).n == 2
    s.replay()
    assert s.get(1).status == "closed"  # the settlement is an event, so replay keeps it


def test_both_hold_keeps_both_open(s):
    rivals(s)
    s.settle(1, 2, None, "s9")
    assert s.conflicts() == [] and s.get(1).status == s.get(2).status == "open"


def test_retracting_a_settlement_brings_the_conflict_back(s):
    rivals(s)
    s.settle(1, 2, 2, "s9")
    s.retract("s9")
    assert s.get(1).status == "open" and len(s.conflicts()) == 1


@pytest.mark.parametrize("source_kind", ["document", "agent_inferred"])
def test_weak_sources_cannot_supersede_or_carve(s, source_kind):
    new_python(s)
    s.observe("switched to Rust", source_kind, "doc:planted", ENV, says(action="supersedes", belief_id=1, statement="User writes Rust"))
    assert s.get(1).status == "open" and s.get(1).against > PRIOR  # contradicted at the source's weight, not closed
    s.observe("Go here", source_kind, "doc:planted", ENV, says(action="carve", belief_id=1, statement="This repo uses Go", context=list(ENV)))
    assert s.get(3).narrows is None  # no override that would hide the general belief
    assert {frozenset((a.id, b.id)) for a, b in s.conflicts()} == {frozenset((1, 2)), frozenset((1, 3))}
    assert 1 in {b.id for b in s.recall(ENV)}  # the user's belief is still recalled, flagged instead of hidden


def test_effects_on_the_same_belief_count_once(s):
    new_python(s)
    s.observe("Python", "user_said", "s2", ENV, says(dict(action="supports", belief_id=1), dict(action="supports", belief_id=1)))
    assert s.get(1).n == 2


def test_no_effects_means_irrelevant(s):
    assert s.observe("brb coffee", "user_said", "s1", ENV, says()) == "irrelevant: no belief changed"
    assert s.beliefs() == [] and s.db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


def test_recall_near_ties_go_firmest_first(tmp_path):
    q = np.array([1.0, 0.0, 0.0], np.float32)

    def unit(sim, z=0.0):
        return np.array([sim, np.sqrt(1 - sim**2 - z**2), z], np.float32)

    vecs = {"query": q, "Neovim": unit(0.67), "VS Code": unit(0.69), "zsh": unit(0.40, 0.5)}
    s = Store(tmp_path / "b.db", embed=lambda texts: np.stack([vecs.get(t, unit(0.0)) for t in texts]))
    for name in ("Neovim", "VS Code", "zsh"):
        s.observe(name, "user_said", "s1", ENV, says(action="new", statement=name, context=[]))
    for _ in range(3):
        s.observe("x", "user_said", "s2", ENV, says(action="supports", belief_id=1))  # Neovim gets firm
        s.observe("x", "user_said", "s3", ENV, says(action="supports", belief_id=3))  # zsh is firm but far away
    assert [b.statement for b in s.recall(ENV, "query")] == ["Neovim", "VS Code", "zsh"]


def test_retract_removes_a_source_and_replay_matches_snapshot(s):
    new_python(s, ref="doc:old-wiki")
    s.observe("Python again", "user_said", "s2", ENV, says(action="supports", belief_id=1))
    s.observe("Python again", "document", "doc:old-wiki", ENV, says(action="supports", belief_id=1))
    before = snapshot(s)
    s.replay()
    assert snapshot(s) == before

    s.retract("doc:old-wiki")
    assert (s.get(1).support, s.get(1).n) == (1.5, 1)  # only the user's statement still counts
    s.observe("Python", "document", "doc:old-wiki", ENV, says(action="supports", belief_id=1))
    assert s.get(1).n == 1  # later evidence from a retracted source is ignored too


def test_belief_with_no_remaining_evidence_disappears(s):
    new_python(s, ref="doc:bad")
    s.retract("doc:bad")
    assert s.recall(ENV) == []


def test_purge_created_event_deletes_belief_and_reopens_what_it_superseded(s):
    new_python(s)
    s.observe("Rust now", "user_said", "s2", ENV, says(action="supersedes", belief_id=1, statement="User writes Rust"))
    created = s.history(2)[0]["id"]
    s.purge(created)
    assert s.get(2) is None and s.get(1).status == "open"
    new_python(s, ref="s3")
    assert s.get(3) is not None  # purged ids are never reused


def test_merge_labels_survives_replay(s):
    new_python(s)
    s.merge_labels("language", "tooling")
    s.replay()
    assert s.get(1).labels == ["tooling"]


def test_digest_holds_only_firm_beliefs(s):
    new_python(s)
    assert s.digest(ENV) == []
    s.observe("Python", "user_said", "s2", ENV, says(action="supports", belief_id=1))
    assert [b.id for b in s.digest(ENV)] == [1]


def test_environment_reads_remote_and_folder_map(tmp_path):
    repo = tmp_path / "work" / "api"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", "git@github.com:acme/api.git"], check=True)
    config = tmp_path / "config.json"
    config.write_text(f'{{"folders": {{"{tmp_path / "work"}": ["work"]}}}}')
    assert environment(repo, config) == ("work", "owner:acme", "repo:api")
    plain = tmp_path / "scratch"
    plain.mkdir()
    assert environment(plain, tmp_path / "missing.json") == ("repo:scratch",)
