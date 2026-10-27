"""Exact compatibility with the design's pre-study source revision.

The reference is executed from Git objects, never from retained run evidence or
the current implementation. Both probes use the same Python/dependency build,
so this tests source compatibility without platform-specific floating fixtures.
Only the imported source files are extracted into pytest temporary storage.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "3e946b79631b78e7aa1e96979ad72f69bde57b50"
SOURCE_FILES = tuple(f"src/softrole/{name}.py" for name in (
    "__init__", "config", "env", "scenarios", "model", "rollout", "learning",
    "optimizers", "paths", "train", "programme", "graded_pilot",
)) + (
    "src/hetnet_ext/__init__.py", "src/hetnet_ext/signatures.py",
    "src/envs/ic3net_envs/__init__.py",
    "src/envs/ic3net_envs/fire_commander_env.py",
    "src/envs/ic3net_envs/predator_capture_env.py",
    "src/WildFire_Simulate_Original.py",
)
CASES = ("binary_default", "binary_capcon", "g_cap", "g_type")


def _probe(source_root, output):
    """Run only in a fresh process, with imports explicitly bound to one tree."""
    source_root = Path(source_root).resolve()
    sys.path[:0] = [str(source_root / "src"), str(source_root / "src/envs")]
    import torch
    from softrole.config import Config
    from softrole.env import make_env
    from softrole.graded_pilot import expected_entries, pilot_config
    from softrole.model import SoftRoleNet
    from softrole.programme import CAPCON_PROFILE, programme_config
    from softrole.rollout import run_episode
    from softrole.scenarios import make_scenarios, scenario_from_record

    torch.set_num_threads(1)

    def freeze(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().copy()
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return {key: freeze(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(freeze(item) for item in value)
        return value

    configs = {
        "binary_default": Config(),
        "binary_capcon": programme_config("l1b_decomposed_beta1", 0, 1e-3,
                                          profile=CAPCON_PROFILE),
        "g_cap": pilot_config(0), "g_type": pilot_config(1),
    }
    result = {"pilot_entries": expected_entries(), "cases": {}}
    for case_index, (name, config) in enumerate(configs.items()):
        # These are the real existing configurations, including their full
        # horizon and architecture. No compact surrogate config is substituted.
        np.random.seed(61)
        random.seed(62)
        torch.manual_seed(63)
        model = SoftRoleNet(**config.model_kwargs()).double()
        settings = (dict(sensing_p=config.sensing_p, sensing_a=config.sensing_a,
                         vision=config.vision) if config.capabilities == "graded" else {})
        nominal = make_scenarios(941 + case_index, 12, config.training_compositions,
                                 config.failure_prob, config.failure_window, **settings)
        # Event probes exercise the supported historical failure path without
        # altering the locked pilot config or its no-failure training schedule.
        failure = make_scenarios(951 + case_index, 12, config.training_compositions,
                                 failure_prob=1.0, failure_window=(1, 1), **settings)
        records = [asdict(scenario) for scenario in nominal + failure]
        canonical = json.dumps(records, sort_keys=True, allow_nan=False)
        entry = {"config": config.to_dict(), "model_config": config.model_kwargs(),
                 "scenarios": records,
                 "scenario_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                 "round_trip": [asdict(scenario_from_record(json.loads(json.dumps(row))))
                                for row in records],
                 "model_state": freeze(model.state_dict()), "episodes": []}
        for scenario in (nominal[0], failure[0]):
            adapter = make_env(config)
            observed, forwards = [], []
            original_observe = adapter.observe

            def observe(kappa):
                value = original_observe(kappa)
                observed.append(freeze({"observation": value, "physical": kappa,
                                        "raw": adapter.raw_observation,
                                        "positions": adapter.positions}))
                return value

            def forward_hook(module, arguments, keywords, values):
                forwards.append(freeze({"arguments": arguments, "keywords": keywords,
                                        "outputs": values}))

            adapter.observe = observe
            handle = model.register_forward_hook(forward_hook, with_kwargs=True)
            try:
                episode = run_episode(model, adapter, config, scenario,
                                      training=False, trace=True)
            finally:
                handle.remove()
            entry["episodes"].append({
                "scenario": asdict(scenario), "observations": observed,
                "forwards": forwards, "episode": freeze(vars(episode)),
                "simulator_numpy_rng": freeze(adapter._rng.numpy_state),
                "simulator_python_rng": adapter._rng.python_state,
            })
        entry["outer_rng"] = freeze({"numpy": np.random.get_state(),
                                      "python": random.getstate(),
                                      "torch": torch.get_rng_state()})
        result["cases"][name] = entry

    # Editable installs must never let a missing historical dependency silently
    # resolve to the current checkout, which would weaken the comparison.
    origins = {}
    for name, module in tuple(sys.modules.items()):
        if (name.split(".")[0] in ("softrole", "hetnet_ext", "ic3net_envs")
                or name == "WildFire_Simulate_Original"):
            origin = Path(module.__file__).resolve()
            assert origin.is_relative_to(source_root), (name, origin, source_root)
            origins[name] = str(origin.relative_to(source_root))
    result["import_origins"] = origins
    with Path(output).open("xb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)


@pytest.fixture(scope="module")
def comparison(tmp_path_factory):
    work = tmp_path_factory.mktemp("stale-defaults")
    baseline_root = work / "baseline"
    for relative in SOURCE_FILES:
        data = subprocess.check_output(["git", "show", f"{BASELINE}:{relative}"], cwd=ROOT)
        destination = baseline_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    results = []
    for label, source in (("baseline", baseline_root), ("current", ROOT)):
        output = work / f"{label}.pickle"
        environment = dict(os.environ, PYTHONPATH=os.pathsep.join(
            (str(source / "src"), str(source / "src/envs"))))
        environment.update({name: "1" for name in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--probe", str(source), str(output)],
            cwd=work, env=environment, text=True, capture_output=True, timeout=120,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        with output.open("rb") as stream:
            results.append(pickle.load(stream))
    return tuple(results)


def assert_bit_identical(actual, expected, location):
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray), location
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, location
        # Raw bytes also distinguish signed zero and preserve masked -inf logits.
        assert actual.tobytes() == expected.tobytes(), location
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), location
        for key in expected:
            assert_bit_identical(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected), location
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_bit_identical(left, right, f"{location}[{index}]")
    else:
        assert type(actual) is type(expected) and actual == expected, location


def test_pilot_manifest_entries_and_probe_imports_match_baseline(comparison):
    baseline, current = comparison
    assert_bit_identical(current["pilot_entries"], baseline["pilot_entries"], "pilot_entries")
    assert current["import_origins"] == baseline["import_origins"]


@pytest.mark.parametrize("name", CASES)
def test_existing_configs_scenarios_observations_and_full_episodes_are_bit_identical(comparison, name):
    baseline, current = comparison
    reference = baseline["cases"][name]
    # Guard against a vacuous probe: complete trajectories and the scheduled
    # event must actually be present, including observations after the failure.
    assert all(row["episode"]["metrics"]["success"]
               or row["episode"]["metrics"]["steps"] == reference["config"]["max_steps"]
               for row in reference["episodes"])
    assert reference["episodes"][1]["episode"]["metrics"]["event_exposed"]
    assert all(row["forwards"] and row["observations"] for row in reference["episodes"])
    assert_bit_identical(current["cases"][name], reference, name)


if __name__ == "__main__":
    assert len(sys.argv) == 4 and sys.argv[1] == "--probe"
    _probe(sys.argv[2], sys.argv[3])
