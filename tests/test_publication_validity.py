"""Reject metadata that could silently mislabel a reconstruction comparison.

These are synthetic corruption/misconfiguration cases, not findings of damage
in the archived research runs. No policy or environment is executed here.
"""
from copy import deepcopy

import pytest

from publication_reconstruction import artifacts, study
from publication_reconstruction.__main__ import parser, resolve


def locked_metadata(tmp_path, task="pcp"):
    index = {"pp": 0, "pcp": 3, "fc": 6}[task]
    protocol = resolve(parser().parse_args(study.train_arguments(index, tmp_path)))
    # Recorded scientific defaults matter even when the launcher does not spell
    # them out as command-line options. This fixture contains no trained weights.
    arguments = dict(
        seed=0, nprocesses=4, epoch_size=10, batch_size=500,
        max_steps=300 if task == "fc" else 80,
        num_epochs=1400 if task == "fc" else 2000,
        publication_env_version="corrected-v1", use_binary=False,
        model_spec="supplement-v1", max_env_steps=protocol["max_env_steps"],
        lrate=.001, env_name="fire_commander" if task == "fc" else "predator_capture",
        nfriendly_P=3 if task == "pp" else 2, nfriendly_A=0 if task == "pp" else 1,
        nagents=3, dim=5, vision=1 if task == "fc" else 2, detach_gap=5,
        hid_size=128, hetgat=True, hetgat_a2c=True,
        milestones=protocol["milestones"], rng_scheme="seedsequence-v1",
        save_every=10, episode_log="stdout", wall_seconds=165600,
        gamma=1.0, tau=1.0, normalize_rewards=False, entr=0., value_coeff=.01,
        comm_range_P=-1, comm_range_A=-1, lossy_comm=False,
        min_comm_loss=0., max_comm_loss=.3, recurrent=False,
        commnet=False, hetcomm=False, ic3net=False, use_cuda=False,
        random=False, eval=False, msg_dim=16, action_scale=1., nactions="1",
        rnn_type="MLP", total_state_action_in_batch=500,
        no_stay=False, mode="mixed", tensor_obs=False, A_vision=-1,
    )
    if task == "fc":
        arguments.update(nfires=1, reward_type=3,
                         fire_spread_off=False, max_wind_speed=None)
    else:
        arguments.update(nenemies=1, moving_prey=False,
                         enemy_comm=False, second_reward_scheme=False)
    return protocol, arguments


@pytest.mark.parametrize("task", ["pp", "pcp", "fc"])
def test_locked_recorded_defaults_remain_accepted(tmp_path, task):
    protocol, arguments = locked_metadata(tmp_path, task)
    artifacts._validate_scientific_protocol(protocol, arguments)


@pytest.mark.parametrize("task,field,value", [
    ("pcp", "gamma", .5),
    ("pcp", "second_reward_scheme", True),
    ("pcp", "nenemies", 2),
    ("pcp", "comm_range_P", 0),
    ("fc", "fire_spread_off", True),
    ("fc", "max_wind_speed", 7.),
])
def test_scientific_defaults_cannot_drift_behind_unchanged_command(
        tmp_path, task, field, value):
    protocol, arguments = locked_metadata(tmp_path, task)
    artifacts._validate_scientific_protocol(protocol, arguments)
    changed = {**arguments, field: value}
    with pytest.raises(ValueError, match=field):
        artifacts._validate_scientific_protocol(protocol, changed)


@pytest.mark.parametrize("task,field", [
    ("pcp", "gamma"),
    ("pcp", "nenemies"),
    ("fc", "fire_spread_off"),
    ("fc", "max_wind_speed"),
])
def test_schema_two_cannot_certify_missing_scientific_defaults(tmp_path, task, field):
    protocol, arguments = locked_metadata(tmp_path, task)
    del arguments[field]
    with pytest.raises(ValueError, match=field):
        artifacts._validate_scientific_protocol(protocol, arguments)


def candidate_dependencies(tmp_path, monkeypatch):
    expected, arguments = locked_metadata(tmp_path)
    final_run, parent_run = tmp_path / "final", tmp_path / "parent"
    parent_protocol = deepcopy(expected)
    counts = dict(updates=15000, env_steps=30_000_001,
                  episodes=1_000_000, epoch=1500)
    row = dict(run_dir=str(parent_run), path=str(parent_run / "checkpoint.pt"),
               checkpoint_sha256="recorded-hash", counts=counts)
    saved = dict(seed=0, reconstruction=dict(counts=counts, resolved_args=arguments))

    def verify(run):
        return (parent_protocol if str(run) == str(parent_run) else expected), {}

    monkeypatch.setattr(artifacts, "verify_run", verify)
    monkeypatch.setattr(study, "checkpoint_candidates", lambda _run: [row])
    monkeypatch.setattr(artifacts, "digest", lambda _path: "recorded-hash")
    monkeypatch.setattr(artifacts, "load_checkpoint", lambda *_args: saved)
    return expected, final_run, parent_protocol, saved


def test_selected_parent_protocol_must_match_expected_training_seed(tmp_path, monkeypatch):
    expected, run, parent_protocol, _ = candidate_dependencies(tmp_path, monkeypatch)
    parent_protocol["seed"] = 2
    with pytest.raises(ValueError, match="locked seed"):
        study.select_checkpoint(run, expected, 30_000_000)


def test_selected_checkpoint_seed_is_not_relabelled_from_expected_metadata(tmp_path, monkeypatch):
    expected, run, _, saved = candidate_dependencies(tmp_path, monkeypatch)
    saved["seed"] = 2
    with pytest.raises(ValueError, match="training seed"):
        study.select_checkpoint(run, expected, 30_000_000)


def test_valid_parent_candidate_retains_its_real_seed(tmp_path, monkeypatch):
    expected, run, _, saved = candidate_dependencies(tmp_path, monkeypatch)
    selected = study.select_checkpoint(run, expected, 30_000_000)
    assert selected["training_seed"] == saved["seed"] == expected["seed"]
