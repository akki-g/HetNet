"""Independent regressions for committed-prefix and raw-index integrity."""
import json

import pytest

from softrole.config import Config
from softrole.ledger import ChunkedEpisodes, audit_run


def fixture_run(path, *, chunked=False, unlogged_update=False):
    path.mkdir()
    config = Config(model="shared", max_steps=3, nprocesses=1, batch_steps=1)
    (path / "config.json").write_text(json.dumps(config.to_dict()))
    episodes = [dict(update=u, collector=0, scenario_id=f"scenario-{u}", steps=1,
                     num_p=2, num_a=1, num_agents=3, composition=[2, 1], success=False,
                     team_return=-1., mean_agent_return=-1 / 3, gate_load=[1.])
                for u in (1, 2)]
    updates = [dict(update=u, steps=1, episodes=1, total_steps=u,
                    gradient_norm_before_clip=1.) for u in (1, 2)]
    if unlogged_update:
        episodes.append(dict(episodes[-1], update=3, scenario_id="scenario-3"))
    (path / "updates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in updates))
    if chunked:
        writer = ChunkedEpisodes(path, config.max_steps)
        for row in episodes:
            writer.append([row], row["update"])
        writer.seal()
        writer.close()
    else:
        (path / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in episodes))
    return episodes


def test_torn_update_append_after_a_committed_boundary_is_not_parsed(tmp_path):
    run = tmp_path / "run"
    fixture_run(run)
    baseline = audit_run(run, through_update=2)
    with (run / "updates.jsonl").open("a") as stream:
        stream.write('{"update": 3, "unfinished":')
    assert audit_run(run, through_update=2) == baseline


def test_nonfinite_nested_metrics_are_not_admitted(tmp_path):
    run = tmp_path / "run"
    episodes = fixture_run(run)
    episodes[0]["gate_load"] = [float("nan")]
    (run / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in episodes))
    with pytest.raises(ValueError, match="[Nn]onfinite"):
        audit_run(run)


def test_a_chunk_cannot_hide_raw_updates_outside_its_claimed_range(tmp_path):
    run = tmp_path / "run"
    fixture_run(run, chunked=True)
    path = run / "episodes.index.json"
    index = json.loads(path.read_text())
    chunk = index["chunks"][0]
    chunk["first_update"], chunk["last_update"] = 2, 99
    chunk["updates"] = chunk["updates"][1:]
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError):
        audit_run(run, through_update=2)


def test_unbounded_audit_rejects_a_duplicate_after_the_last_counted_episode(tmp_path):
    run = tmp_path / "run"
    episodes = fixture_run(run)
    with (run / "episodes.jsonl").open("a") as stream:
        stream.write(json.dumps(episodes[-1]) + "\n")
    with pytest.raises(ValueError):
        audit_run(run)


@pytest.mark.parametrize("chunked", [False, True])
def test_unbounded_audit_rejects_episodes_without_a_corresponding_update(tmp_path, chunked):
    run = tmp_path / "run"
    fixture_run(run, chunked=chunked, unlogged_update=True)
    with pytest.raises(ValueError):
        audit_run(run)
    # An explicit checkpoint boundary deliberately excludes the later tail.
    assert audit_run(run, through_update=2)["total_episodes"] == 2
