"""Protocol version 2: diagnostics on composition/failure-trained and new-learner checkpoints.

Historical next-update replay must reproduce the production gradient for every
learner variant, which proves the shared loss-term builder matches training.
"""
import json

import pytest
import torch

from softrole.config import Config, recipe
from softrole.diagnostics import (execute, learner_identity, load_checkpoint, prepare, sha,
                                   training_study, version_one_eligible)
from softrole.train import train


COMMON = dict(optimizer="adam", lr=.001, comm_rounds=3, independent_heads=True, msg_dim=3, heads=2,
              head_dim=3, pre_dim=5, hidden_dim=5, max_steps=4, epochs=2, updates_per_epoch=1,
              nprocesses=1, batch_steps=12, save_every=1)
FIXTURES = {
    "composition_decomposed": lambda: recipe("pcp", "composition", model="shared", communication="real",
                                             critic_mode="decomposed", credit_beta=.5, vision=1, **COMMON),
    "failure_banked_binary": lambda: recipe("pcp", "failure", model="banked", communication="binary",
                                            null_degree_correction=True, failure_window=(1, 1), vision=1, **COMMON),
    "fixed_physical_mean_agent": lambda: Config(model="shared", communication="binary_deterministic",
                                                actor_capabilities="physical_only", objective="mean_agent",
                                                vision=1, **COMMON),
}


def test_fixed_nonnominal_roster_is_fixed_but_not_version_one_eligible():
    cfg = Config(num_p=3, num_a=1)
    assert training_study(cfg) == "fixed"
    assert not version_one_eligible(cfg)
    assert training_study(Config(compositions=((1, 2),))) == "fixed"
    assert not version_one_eligible(Config(compositions=((1, 2),)))
    assert version_one_eligible(Config())
    assert training_study(Config(compositions=((2, 1), (3, 1)))) == "composition"
    assert training_study(Config(failure_prob=.5)) == "failure"
    assert not version_one_eligible(Config(failure_prob=.5))


def test_version_one_rejects_a_fixed_nonnominal_checkpoint(tmp_path):
    cfg = Config(num_p=3, num_a=1, model="shared", communication="real", vision=1, **COMMON)
    checkpoint = train(cfg, tmp_path / "run")
    with pytest.raises(ValueError, match="nominal fixed 2P1A"):
        load_checkpoint(checkpoint, protocol_version=1)
    _, loaded, _, _ = load_checkpoint(checkpoint, protocol_version=2)
    assert training_study(loaded) == "fixed"


def test_version_two_replays_a_chunked_episode_ledger(tmp_path):
    cfg = FIXTURES["fixed_physical_mean_agent"]()
    train(cfg, tmp_path / "run", episode_log="chunked-gzip")
    checkpoint = tmp_path / "run" / "checkpoints" / "epoch0001.pt"
    manifest = prepare([checkpoint], tmp_path / "manifest.json", "bounded first saved epoch",
                       probe_episodes=1, d5_instances=1, d5_bits=2, d5_samples=32,
                       protocol_version=2)
    assert not (tmp_path / "run" / "episodes.jsonl").exists()
    assert any(name.endswith(".gz") for name in manifest["entries"][0]["inputs"])
    execute(tmp_path / "manifest.json", 0, tmp_path / "out")
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    actual = torch.load(checkpoint.with_name("epoch0002.pt"), weights_only=False)
    assert report["D1"]["historical_update_verified"]
    assert report["D1"]["disposable_post_update_signature"] == actual["signature"]["sha256"]


def fixture_run(tmp_path, name):
    cfg = FIXTURES[name]()
    train(cfg, tmp_path / "run")
    return cfg, tmp_path / "run" / "checkpoints" / "epoch0001.pt"


def file_hashes(root):
    return {p: sha(p) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("name", list(FIXTURES))
def test_version_two_replays_the_actual_next_update_for_every_learner_variant(tmp_path, name):
    cfg, checkpoint = fixture_run(tmp_path, name)
    with pytest.raises(ValueError, match="protocol-version 2"):
        prepare([checkpoint], tmp_path / "v1.json", "fixture")
    assert not (tmp_path / "v1.json").exists()
    manifest = prepare([checkpoint], tmp_path / "manifest.json", "first saved epoch; engineering fixture",
                       probe_episodes=1, d5_instances=1, d5_bits=2, d5_samples=32, protocol_version=2)
    entry = manifest["entries"][0]
    assert manifest["version"] == 2 and entry["learner"] == learner_identity(cfg)
    assert entry["training_study"] == training_study(cfg) == name.split("_")[0]
    before = file_hashes(tmp_path / "run")
    execute(tmp_path / "manifest.json", 0, tmp_path / "out")
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    actual = torch.load(checkpoint.with_name("epoch0002.pt"), weights_only=False)
    assert report["protocol_version"] == 2
    assert report["D1"]["historical_update_verified"]
    assert report["D1"]["disposable_post_update_signature"] == actual["signature"]["sha256"]
    assert report["learner"] == learner_identity(cfg) and report["training_study"] == training_study(cfg)
    compositions = {tuple(e["composition"]) for e in report["episodes"]}
    assert {tuple(map(int, k.split(","))) for k in report["D3"]["by_composition"]} == compositions
    assert sum(report["episode_compositions"].values()) == len(report["episodes"])
    assert all(e["steps"] == r["steps"] for e, r in zip(report["D3_per_episode"], report["episodes"]))
    assert all(tuple(e["composition"]) == tuple(r["composition"]) for e, r in zip(report["D3_per_episode"], report["episodes"]))
    variants = report["D4"]["variants"]
    protocol = report["D4"]["protocol"]
    if cfg.critic_mode == "decomposed":
        assert report["D3_learned"]["overall"]["episodes"] == len(report["episodes"])
        assert report["D3_learned"]["overall"]["agent_steps"] == sum(sum(e["composition"]) * e["steps"] for e in report["episodes"])
        assert {"learned_gae_beta_1", "learned_gae_beta_0.5", "learned_gae_beta_0"} <= set(variants)
        assert protocol["production_variant"] == f"learned_gae_beta_{cfg.credit_beta:g}"
        for check in report["D4"]["identity_checks_per_episode"]:
            assert check["learned_values_sum_max_abs_error"] < 1e-12
            assert check["learned_beta_1_advantage_max_abs_error"] < 1e-10
            assert check["learned_beta_1_gradient_max_abs_error"] < 1e-9
    else:
        assert report["D3_learned"] is None and not any(v.startswith("learned") for v in variants)
        assert protocol["production_variant"] == "team_gae"
    expected_scale = 1.0 if cfg.objective == "team" else None
    scales = {c["episode_scale"] for c in report["D4"]["identity_checks_per_episode"]}
    assert scales == {1.0} if expected_scale == 1.0 else all(0 < s < 1 for s in scales)
    assert variants["allocated_gae_beta_1"]["mean_difference_from_team_gae_l2"] < 1e-9
    if cfg.communication == "binary":
        assert report["D5"]["status"] == "conditional_binary_audit" and report["D5"]["instances"]
    else:
        assert report["D5"]["status"].startswith("not_applicable") and not report["D5"]["instances"]
        if cfg.communication == "binary_deterministic":
            assert "score_function" in report["D5"]["status"]
    assert len(report["D6"]["strata"]) == 9
    support = report["D6"]["training_support"]
    assert support["training_study"] == training_study(cfg)
    assert support["step_one_sensor_loss_in_training"] == (cfg.failure_prob > 0)
    assert support["compositions_in_training"] == [list(c) for c in ((1, 2), (2, 2), (3, 1)) if c in cfg.training_compositions]
    assert before == file_hashes(tmp_path / "run")


def test_version_two_metadata_tamper_is_rejected(tmp_path):
    cfg, checkpoint = fixture_run(tmp_path, "fixed_physical_mean_agent")
    path = tmp_path / "manifest.json"
    manifest = prepare([checkpoint], path, "fixture", probe_episodes=1, protocol_version=2)
    manifest["entries"][0]["learner"]["objective"] = "team"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="learner/training metadata"):
        execute(path, 0, tmp_path / "out", dry_run=True)
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="protocol version"):
        load_checkpoint(checkpoint, protocol_version=3)
    saved, loaded_cfg, _, _ = load_checkpoint(checkpoint, protocol_version=2)
    assert loaded_cfg.to_dict() == cfg.to_dict()


def test_version_one_manifest_shape_is_unchanged_for_nominal_checkpoints(tmp_path):
    cfg = Config(model="shared", communication="real", vision=1, **COMMON)
    train(cfg, tmp_path / "run")
    checkpoint = tmp_path / "run" / "checkpoints" / "epoch0001.pt"
    manifest = prepare([checkpoint], tmp_path / "manifest.json", "fixture", probe_episodes=1)
    assert manifest["version"] == 1
    assert set(manifest["entries"][0]) == {"checkpoint", "inputs", "training_seed", "communication",
                                           "progress", "model", "source_sha256"}
    execute(tmp_path / "manifest.json", 0, tmp_path / "out")
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["protocol_version"] == 1 and report["D1"]["historical_update_verified"]
    assert report["D6"]["training_support"] == {"training_study": "fixed", "compositions_in_training": [],
                                                "step_one_sensor_loss_in_training": False}
