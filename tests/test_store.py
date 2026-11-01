"""Store logic with a scripted judge and a bag-of-words embedder, so no model or network is needed."""

import hashlib
import subprocess
import threading

import numpy as np
import pytest

from beliefs.judge import Consolidation, Effect, Generalization, LabelMerge, Merge, Verdict
from beliefs.server import build, environment
from beliefs.store import PRIOR, Store, Unavailable

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
    s.observe("this repo is Rust", "user_said", "s2", ENV, says(action="carve", belief_id=1, statement="This repo is written in Rust", context=list(ENV)))
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


@pytest.mark.parametrize("source_kind", ["tool_observed", "document", "agent_inferred"])
def test_only_the_user_can_supersede_or_carve(s, source_kind):
    new_python(s)
    s.observe("switched to Rust", source_kind, "doc:planted", ENV, says(action="supersedes", belief_id=1, statement="User writes Rust"))
    assert s.get(1).status == "open" and s.get(1).against > PRIOR  # contradicted at the source's weight, not closed
    s.observe("Go here", source_kind, "doc:planted", ENV, says(action="carve", belief_id=1, statement="This repo uses Go", context=list(ENV)))
    assert s.get(3).narrows is None  # no override that would hide the general belief
    assert {frozenset((a.id, b.id)) for a, b in s.conflicts()} == {frozenset((1, 2)), frozenset((1, 3))}
    assert 1 in {b.id for b in s.recall(ENV)}  # the user's belief is still recalled, flagged instead of hidden


def test_each_source_holds_one_latest_vote_per_belief(s):
    s.observe("install with npm", "document", "README.md", ENV, says(action="new", statement="Repo uses npm", context=list(ENV)))
    for _ in range(4):  # re-reading the same README adds nothing
        s.observe("install with npm", "document", "README.md", ENV, says(action="supports", belief_id=1))
    assert (s.get(1).support, s.get(1).n) == (1.0, 1)
    s.observe("README now says pnpm", "document", "README.md", ENV, says(action="contradicts", belief_id=1))
    assert (s.get(1).support, s.get(1).against, s.get(1).n) == (0.5, 1.0, 1)  # its vote flipped
    s.observe("npm", "user_said", "s1", ENV, says(action="supports", belief_id=1))
    assert s.get(1).n == 2
    before = snapshot(s)
    s.replay()
    assert snapshot(s) == before


def test_a_weak_source_never_replaces_the_users_vote_from_the_same_session(s):
    new_python(s, ref="session:1")
    s.observe("looks like Rust", "agent_inferred", "session:1", ENV, says(action="contradicts", belief_id=1))
    assert (s.get(1).support, s.get(1).against, s.get(1).n) == (1.5, pytest.approx(0.7), 2)


def down(*args):
    raise Unavailable("connection refused")


def test_an_outage_queues_statements_and_judges_them_in_order_later(s):
    assert s.observe("I write Python", "user_said", "s1", ENV, down).startswith("queued")
    s.observe("I switched to Rust", "user_said", "s2", ENV, down)
    s.replay()  # the queue is input, so replay keeps it
    assert s.beliefs() == [] and s.db.execute("SELECT COUNT(*) FROM pending").fetchone()[0] == 2

    seen = []
    verdicts = {"I write Python": says(action="new", statement="User writes Python", context=[]),
                "I switched to Rust": says(action="supersedes", belief_id=1, statement="User writes Rust"),
                "I use zsh": says(action="new", statement="User uses zsh", context=[])}
    def judge(statement, *rest):
        seen.append(statement)
        return verdicts[statement](statement, *rest)
    s.observe("I use zsh", "user_said", "s3", ENV, judge)
    assert seen == ["I write Python", "I switched to Rust", "I use zsh"]
    assert [b.statement for b in s.recall(ENV)] == ["User writes Rust", "User uses zsh"]
    assert s.db.execute("SELECT COUNT(*) FROM pending").fetchone()[0] == 0


def test_a_queued_statement_the_judge_rejects_is_dropped_instead_of_blocking(s):
    s.observe("bad", "user_said", "s1", ENV, down)
    def judge(statement, *rest):
        if statement == "bad":
            raise RuntimeError("judge stopped with refusal")
        return says(action="new", statement="User uses zsh", context=[])()
    out = s.observe("I use zsh", "user_said", "s2", ENV, judge)
    assert "dropped queued statement 'bad'" in out and s.get(1).statement == "User uses zsh"


def test_parallel_tool_calls_keep_separate_transactions(s):
    gate, errors, first = threading.Barrier(2, timeout=5), [], set()
    def judge(statement, *rest):
        if statement not in first:  # both calls leave their first judging together, then race to write
            first.add(statement)
            gate.wait()
        return says(action="new", statement=statement, context=[])()
    def call(text):
        try:
            s.observe(text, "user_said", text, ENV, judge)
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=call, args=(t,)) for t in ("I use zsh", "I use vim")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == [] and len(s.beliefs()) == 2


def test_a_write_during_judging_makes_observe_judge_again(s):
    seen = []
    def judge(statement, source_kind, env, candidates, labels):
        seen.append([b.id for b in candidates])
        if len(seen) == 1:  # another session records the same fact while this judge call runs
            s.observe("we use pnpm", "user_said", "other", ENV, says(action="new", statement="Repo uses pnpm", context=list(ENV)))
            return says(action="new", statement="Repo uses pnpm", context=list(ENV))()
        return says(action="supports", belief_id=candidates[0].id)()
    s.observe("pnpm-lock.yaml is present", "tool_observed", "t1", ENV, judge)
    assert seen == [[], [1]] and len(s.beliefs()) == 1 and s.get(1).n == 2


def test_re_judging_gives_up_after_two_retries(s):
    calls = []
    def judge(statement, *rest):
        calls.append(statement)
        if statement == "pnpm":  # every judge call races with another write
            s.observe(f"noise {len(calls)}", "user_said", "other", ENV, says())
        return says(action="new", statement="Repo uses pnpm", context=list(ENV))()
    s.observe("pnpm", "tool_observed", "t1", ENV, judge)
    assert calls.count("pnpm") == 3 and len(s.beliefs()) == 1


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


def git_repo(path):
    path.mkdir(parents=True)
    git = lambda *a: subprocess.run(["git", "-C", str(path), *a], check=True, capture_output=True, text=True).stdout
    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "commit", "-q", "--allow-empty", "-m", "first")
    return git, f"repo:{git('rev-parse', 'HEAD')[:12]}"


def test_a_repo_without_a_remote_is_named_by_its_first_commit(tmp_path):
    git, root = git_repo(tmp_path / "api")
    assert environment(tmp_path / "api", tmp_path / "missing.json") == (root,)
    (tmp_path / "api").rename(tmp_path / "api-v2")
    assert environment(tmp_path / "api-v2", tmp_path / "missing.json") == (root,)  # survives a rename


def test_beliefs_follow_a_repo_when_it_gains_a_remote(tmp_path, monkeypatch):
    git, root = git_repo(tmp_path / "api")
    monkeypatch.setenv("BELIEFS_DB", str(tmp_path / "b.db"))
    monkeypatch.chdir(tmp_path / "api")
    s = Store(embed=fake_embed)
    s.observe("tests use pytest", "user_said", "s1", (root,), says(action="new", statement="Tests use pytest", context=[root]))
    git("remote", "add", "origin", "git@github.com:acme/api.git")
    build()
    remote = ("owner:acme", "repo:api")
    assert [b.statement for b in s.recall(remote)] == ["Tests use pytest"]
    s.replay()
    assert s.get(1).context == remote


def pnpm(s, ref, env=ENV):
    return s.observe("pnpm", "tool_observed", ref, env, says(action="new", statement="Repo uses pnpm", context=list(env), labels=["build"]))


def dreams(**plan):
    return lambda beliefs, labels: Consolidation(**plan)


def test_sleep_merges_only_the_duplicates_you_confirm_and_keeps_their_evidence(s):
    pnpm(s, "t1")
    s.observe("pnpm again", "user_said", "s1", ENV, says(action="supports", belief_id=1))
    pnpm(s, "t2")  # a duplicate, written by a session that raced the first
    plan = dreams(merges=[Merge(reason="same fact", ids=[1, 2])])
    assert s.sleep(plan, None, lambda q: False) == [] and s.get(2).status == "open"
    assert s.sleep(plan, None, lambda q: True) == ["merged #2 into #1"]
    assert (s.get(2).status, s.get(2).superseded_by) == ("merged", 1)
    assert s.get(1).n == 3 and [b.id for b in s.recall(ENV)] == [1]  # t2's vote moved over
    before = snapshot(s)
    s.replay()
    assert snapshot(s) == before


def test_sleep_never_merges_across_contexts(s):
    pnpm(s, "t1")
    pnpm(s, "t2", OTHER)
    assert s.sleep(dreams(merges=[Merge(reason="same", ids=[1, 2])]), None, lambda q: True) == []


def test_a_conflict_follows_a_merged_duplicate(s):
    rivals(s)  # npm #1 against pnpm #2
    s.observe("npm", "user_said", "s1", ENV, says(action="new", statement="Repo uses npm", context=list(ENV)))
    s.observe("npm", "user_said", "s2", ENV, says(action="supports", belief_id=3))
    s.sleep(dreams(merges=[Merge(reason="same", ids=[1, 3])]), None, lambda q: True)
    assert [(a.id, b.id) for a, b in s.conflicts()] == [(2, 3)]


def test_sleep_folds_labels_you_confirm(s):
    pnpm(s, "t1")
    s.observe("uv", "tool_observed", "t2", ENV, says(action="new", statement="Repo uses uv", context=list(ENV), labels=["tooling"]))
    s.observe("ruff", "tool_observed", "t3", ENV, says(action="new", statement="Repo uses ruff", context=list(ENV), labels=["tooling"]))
    s.sleep(dreams(labels=[LabelMerge(reason="same", old="build", new="tooling")]), None, lambda q: True)
    assert s.labels() == ["tooling"]


def test_sleep_generalizes_a_pattern_from_three_repos_as_a_weak_inference(s):
    for i, repo in enumerate(("a", "b", "c")):
        pnpm(s, f"t{i}", ("owner:me", f"repo:{repo}"))
    guess = says(action="new", statement="User prefers pnpm", context=[])
    two = dreams(generalizations=[Generalization(reason="r", ids=[1, 2], statement="User prefers pnpm")])
    assert s.sleep(two, guess, lambda q: True) == []  # two repos are not a pattern yet
    three = dreams(generalizations=[Generalization(reason="r", ids=[1, 2, 3], statement="User prefers pnpm")])
    s.sleep(three, guess, lambda q: False)  # generalizations need no confirmation
    b = s.get(4)
    assert (b.statement, b.premises, b.loose) == ("User prefers pnpm", [1, 2, 3], True)
    s.sleep(three, says(action="supports", belief_id=4), lambda q: True)
    assert s.get(4).n == 1  # rerunning sleep never adds a second vote
