"""Graded inputs are opt-in; historical configs remain hash compatible."""
from dataclasses import replace
import hashlib
import json

import pytest

from softrole.config import Config, recipe


@pytest.mark.parametrize("task,study,digest", [
    ("pp", "fixed", "9abb14fda798049798063fe3c915724e7713f4fb49a0fcfa1d13114853390cfb"),
    ("pcp", "fixed", "df5d64ebfead39cdbe78336f0b82ff2712fd05dce3497d30a1812ed0a0321458"),
    ("fc", "fixed", "428e65af427a1d729f152da0b0737d1e8bfd33f844c239664c13bac859ff2096"),
    ("pcp", "composition", "f2532201ce52d72843d23bbc87cb73a6706243b95b70b7cff041af89050e42db"),
    ("pcp", "failure", "a93b45d17c4b1baaa187b2dac70621a6c6ae6de4250e1b1fc52d83121312d3ff"),
])
def test_binary_config_and_model_layout_match_459bce7(task, study, digest):
    config = recipe(task, study)
    payload = json.dumps({"config": config.to_dict(), "model": config.model_kwargs()}, sort_keys=True)
    assert hashlib.sha256(payload.encode()).hexdigest() == digest


def test_default_fields_are_omitted_and_model_layout_is_unchanged():
    binary = Config(model="shared")
    graded = replace(binary, capabilities="graded", sensing_p=(1, 3), sensing_a=(0,),
                     held_out_sensing_p=(2,), held_out_sensing_a=(1, 2, 3))
    for name in ("capabilities", "descriptor", "sensing_p", "sensing_a",
                 "held_out_sensing_p", "held_out_sensing_a"):
        assert name not in binary.to_dict()
    assert binary.model_kwargs() == graded.model_kwargs()
    assert replace(graded, descriptor="type").model_kwargs() == binary.model_kwargs()
    assert Config(**json.loads(json.dumps(graded.to_dict()))) == graded


@pytest.mark.parametrize("options", [
    {"capabilities": "continuous"}, {"descriptor": "identity"},
    {"descriptor": "type", "actor_capabilities": "physical_only"},
    {"task": "fc"}, {"task": "pp", "num_a": 0}, {"env_version": "paper-v1"},
    {"sensing_p": (-1,)}, {"sensing_p": (4,)}, {"sensing_p": (1.0,)},
    {"sensing_p": (True,)}, {"sensing_p": (1, 1)}, {"sensing_p": None},
    {"sensing_p": ""}, {"sensing_p": {1: "invalid"}}, {"sensing_p": {1, 3}},
    {"sensing_p": frozenset((1, 3))},
    {"sensing_p": (1, 3), "held_out_sensing_p": (3,)},
    {"held_out_sensing_p": (3,)}, {"held_out_sensing_a": (0,)},
    {"sensing_a": (float("nan"),)}, {"held_out_sensing_p": (float("inf"),)},
])
def test_invalid_graded_configuration_fails_before_collection(options):
    with pytest.raises(ValueError):
        Config(**dict({"capabilities": "graded"}, **options))


@pytest.mark.parametrize("field", ["sensing_p", "sensing_a", "held_out_sensing_p", "held_out_sensing_a"])
def test_binary_mode_rejects_sensing_lists(field):
    with pytest.raises(ValueError, match="binary"):
        Config(**{field: (1,)})


def test_failure_support_counts_physical_sensors_in_either_class():
    config = Config(capabilities="graded", num_p=1, num_a=2, sensing_p=(0,), sensing_a=(1, 3),
                    failure_prob=1)
    assert config.training_compositions == ((1, 2),)
    with pytest.raises(ValueError, match="two functioning sensors"):
        replace(config, sensing_a=(0, 1, 3))
    with pytest.raises(ValueError, match="two functioning sensors"):
        replace(config, compositions=((1, 1),))


def test_binary_only_diagnostics_reject_before_creating_run(tmp_path):
    from softrole.train import train
    config = Config(capabilities="graded", sensing_p=(1, 3))
    with pytest.raises(ValueError, match="binary capabilities only"):
        train(config, tmp_path / "invalid", diagnostics_every=1)
    assert not (tmp_path / "invalid").exists()


def test_cli_rejects_graded_diagnostics_even_in_dry_run(tmp_path, capsys):
    from softrole.__main__ import main
    with pytest.raises(SystemExit) as error:
        main(["train", "--output", str(tmp_path / "invalid"), "--capabilities", "graded",
              "--diagnostics-every", "1", "--dry-run"])
    assert error.value.code == 2
    assert "binary capabilities only" in capsys.readouterr().err
    assert not (tmp_path / "invalid").exists()


def test_released_hetnet_rejects_graded_adapter_and_non_native_scenarios(tmp_path):
    from softrole.hetnet import evaluate_hetnet
    from softrole.scenarios import GradedScenario
    config = Config(capabilities="graded")
    with pytest.raises(ValueError, match="binary-mode adapter"):
        evaluate_hetnet(tmp_path / "missing.pt", config, [], tmp_path / "report.json")
    scenario = GradedScenario("sensing-A", 1, 2, 3, 2, 1, sensing=(3, 3, 1))
    with pytest.raises(ValueError, match="native sensing"):
        evaluate_hetnet(tmp_path / "missing.pt", Config(), [scenario], tmp_path / "report.json")
