"""Replay, schema and random-stream contracts for graded sensing scenarios."""
from dataclasses import FrozenInstanceError, asdict
import hashlib
import json
import random

import numpy as np
import pytest

from softrole.scenarios import GradedScenario, Scenario, make_scenarios, scenario_from_record


@pytest.mark.parametrize("probability, expected", [
    (0.0, "cb130fbd979a15a3c9ba7cedb81d222208748f1ed46b5dfd80d5703fcd434283"),
    (1.0, "7135ea8007bb0de3b8b92c9e19c5c176c2a5196add8b57b0ac2939cc83e1bd54"),
])
def test_binary_serialized_panels_match_pre_amendment_goldens(probability, expected):
    # Captured from the unchanged scenario generator at main 459bce7.
    scenarios = make_scenarios(44, 37, [(2, 1), (3, 2)], probability, (10, 12))
    assert all(type(scenario) is Scenario for scenario in scenarios)
    records = [asdict(scenario) for scenario in scenarios]
    assert all("sensing" not in record for record in records)
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(payload).hexdigest() == expected


@pytest.mark.parametrize("vision", [0, 1, 2, 3])
def test_native_graded_scenarios_retain_all_five_streams_and_victim_draws(vision):
    binary = make_scenarios(71, 90, [(2, 1), (3, 2)], 0.7, (1, 5))
    graded = make_scenarios(71, 90, [(2, 1), (3, 2)], 0.7, (1, 5),
                            sensing_p=(vision + 1,), sensing_a=(0,), vision=vision)
    for original, scenario in zip(binary, graded):
        record = asdict(scenario)
        assert record.pop("sensing") == (vision + 1,) * scenario.num_p + (0,) * scenario.num_a
        assert record == asdict(original)


def test_levels_have_stable_prefixes_and_independent_class_and_event_streams():
    options = dict(sensing_p=(1, 3), sensing_a=(0, 2))
    short = make_scenarios(17, 7, [(2, 1), (3, 2)], **options)
    long = make_scenarios(17, 100, [(2, 1), (3, 2)], **options)
    failed = make_scenarios(17, 100, [(2, 1), (3, 2)], 1.0, (1, 5), **options)
    alternate = make_scenarios(17, 100, [(2, 1), (3, 2)], sensing_p=(2,), sensing_a=(1,))
    assert short == long[:len(short)]
    for plain, event, other in zip(long, failed, alternate):
        assert isinstance(plain, GradedScenario)
        assert plain.sensing == event.sensing
        assert len(plain.sensing) == plain.num_p + plain.num_a
        assert set(plain.sensing[:plain.num_p]) <= {1, 3}
        assert set(plain.sensing[plain.num_p:]) <= {0, 2}
        for name in ("scenario_id", "num_p", "num_a", "env_seed", "action_seed", "message_seed"):
            assert getattr(plain, name) == getattr(event, name) == getattr(other, name)


def test_each_agent_uniformly_samples_its_class_levels():
    scenarios = make_scenarios(810, 2000, [(2, 2)], sensing_p=(1, 3), sensing_a=(0, 2))
    levels = np.asarray([scenario.sensing for scenario in scenarios])
    for agent in range(4):
        high = 3 if agent < 2 else 2
        assert 0.45 < np.mean(levels[:, agent] == high) < 0.55
    # Independent per-agent draws must not collapse into one class-wide level.
    assert np.any(levels[:, 0] != levels[:, 1])
    assert np.any(levels[:, 2] != levels[:, 3])


def test_omitted_class_support_uses_native_levels_for_the_window():
    sensing_a = make_scenarios(5, 3, [(2, 2)], sensing_a=(1,), vision=3)
    assert all(scenario.sensing == (4, 4, 1, 1) for scenario in sensing_a)
    sensing_p = make_scenarios(5, 3, [(2, 2)], sensing_p=(2,), vision=1)
    assert all(scenario.sensing == (2, 2, 0, 0) for scenario in sensing_p)


def test_failures_can_target_sensing_capture_agents_without_targeting_blind_agents():
    scenarios = make_scenarios(19, 400, [(1, 2)], 1.0, (2, 2),
                               sensing_p=(0,), sensing_a=(1, 3))
    assert {scenario.victim for scenario in scenarios} == {1, 2}
    assert 0.4 < np.mean([scenario.victim == 1 for scenario in scenarios]) < 0.6
    assert all(scenario.event_step == 2 and scenario.sensing[scenario.victim] >= 1
               for scenario in scenarios)


@pytest.mark.parametrize("composition, p_levels, a_levels", [
    ((1, 1), (3,), (0,)),
    ((2, 1), (0, 3), (1,)),
    ((3, 2), (0,), (0, 1)),
])
def test_failure_support_must_always_retain_two_initial_sensors(composition, p_levels, a_levels):
    with pytest.raises(ValueError, match="at least two sensing agents"):
        make_scenarios(1, 0, [composition], 0.1, sensing_p=p_levels, sensing_a=a_levels)
    # Blind draws remain legitimate when no failure is requested.
    assert len(make_scenarios(1, 2, [composition], sensing_p=p_levels, sensing_a=a_levels)) == 2


@pytest.mark.parametrize("graded", [False, True])
def test_json_round_trip_preserves_the_scenario_class_and_exact_fields(graded):
    options = dict(sensing_p=(1, 3), sensing_a=(0,)) if graded else {}
    original = make_scenarios(4, 1, [(2, 1)], **options)[0]
    record = json.loads(json.dumps(asdict(original)))
    restored = scenario_from_record(record)
    assert restored == original
    assert type(restored) is type(original)
    if graded:
        assert isinstance(restored.sensing, tuple)
        assert isinstance(record["sensing"], list)  # Loader does not mutate input.
        with pytest.raises(FrozenInstanceError):
            restored.sensing = ()
    assert asdict(restored) == asdict(original)


def test_empty_sensing_record_remains_explicitly_graded():
    record = asdict(Scenario("native", 1, 2, 3, 2, 1))
    record["sensing"] = []
    restored = scenario_from_record(record)
    assert type(restored) is GradedScenario and restored.sensing == ()


@pytest.mark.parametrize("levels", [None, "13", {1: 3}, {1, 3}, frozenset((1, 3)),
                                    (True,), (np.bool_(False),),
                                    (1.0,), (1.5,), (-1,), (float("nan"),)])
def test_malformed_level_lists_are_rejected_without_coercion(levels):
    with pytest.raises(ValueError, match="integer sensing levels"):
        make_scenarios(1, 1, [(2, 1)], sensing_p=levels)
    record = asdict(Scenario("invalid", 1, 2, 3, 2, 1))
    record["sensing"] = levels
    with pytest.raises(ValueError, match="integer sensing levels"):
        scenario_from_record(record)


@pytest.mark.parametrize("vision, levels", [(2, (4,)), (0, (2,)), (-1, (1,)),
                                            (1.0, (1,)), (True, (1,))])
def test_graded_window_and_level_bounds_are_checked(vision, levels):
    with pytest.raises(ValueError, match="vision"):
        make_scenarios(1, 1, [(2, 1)], sensing_p=levels, vision=vision)


def test_graded_generation_does_not_touch_global_random_streams():
    numpy_before, python_before = np.random.get_state(), random.getstate()
    make_scenarios(10, 20, [(2, 1)], sensing_p=(1, 3), sensing_a=(0, 2))
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert random.getstate() == python_before
