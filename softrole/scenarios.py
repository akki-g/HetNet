"""Architecture-independent scenarios and isolated legacy simulator randomness."""
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral
import random

import numpy as np


class IsolatedRNG:
    """Own NumPy/Python RNG states without advancing the caller's global streams.

    Legacy environments use module-level random functions. Collectors run in
    separate processes; this context is deliberately not a thread-safety device.
    """

    def __init__(self, seed):
        self.numpy_state = np.random.RandomState(int(seed)).get_state()
        self.python_state = random.Random(int(seed)).getstate()

    @contextmanager
    def use(self):
        numpy_outer, python_outer = np.random.get_state(), random.getstate()
        np.random.set_state(self.numpy_state)
        random.setstate(self.python_state)
        try:
            yield
        finally:
            self.numpy_state, self.python_state = np.random.get_state(), random.getstate()
            np.random.set_state(numpy_outer)
            random.setstate(python_outer)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    env_seed: int
    action_seed: int
    message_seed: int
    num_p: int
    num_a: int
    event_step: int = -1
    victim: int = -1


@dataclass(frozen=True)
class GradedScenario(Scenario):
    """Per-agent sensing levels in simulator order: P agents, then A agents."""

    sensing: tuple = ()


def _sensing_levels(values, name):
    """Normalize new graded inputs without silently truncating levels."""
    if isinstance(values, (str, bytes, dict, set, frozenset)):
        raise ValueError(f"{name} must contain nonnegative integer sensing levels")
    try:
        levels = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain nonnegative integer sensing levels") from exc
    if any(isinstance(level, (bool, np.bool_)) or not isinstance(level, Integral)
           or level < 0 for level in levels):
        raise ValueError(f"{name} must contain nonnegative integer sensing levels")
    return tuple(int(level) for level in levels)


def scenario_from_record(record):
    """Load either schema, preserving the historical binary Scenario fields."""
    if "sensing" not in record:
        return Scenario(**record)
    fields = dict(record)
    fields["sensing"] = _sensing_levels(fields["sensing"], "sensing")
    return GradedScenario(**fields)


def make_scenarios(seed, count, compositions, failure_prob=0.0,
                   failure_window=(10, 30), sensing_p=(), sensing_a=(), vision=2):
    """Return repeatable episode scenarios, with stable prefixes as count grows.

    Event steps are zero-based and the window includes both endpoints. Failure
    scenarios retain at least one sensing teammate. Binary failures require at
    least two P agents; graded failures require two initially sensing agents.
    Nonempty class level lists opt into graded records and an independent sixth
    random stream. An empty class list uses its native level (vision + 1 for P,
    zero for A). Both empty lists retain the exact historical binary records.
    Task-specific event support is checked by the environment adapter.
    """
    sensing_p = _sensing_levels(sensing_p, "sensing_p")
    sensing_a = _sensing_levels(sensing_a, "sensing_a")
    graded = bool(sensing_p or sensing_a)
    if graded:
        if isinstance(vision, (bool, np.bool_)) or not isinstance(vision, Integral) or vision < 0:
            raise ValueError("vision must be a nonnegative integer")
        p_levels, a_levels = sensing_p or (int(vision) + 1,), sensing_a or (0,)
        if any(level > vision + 1 for level in p_levels + a_levels):
            raise ValueError("sensing levels must lie in [0, vision + 1]")
    compositions = [tuple(map(int, composition)) for composition in compositions]
    if count < 0 or not compositions:
        raise ValueError("count must be nonnegative and compositions must be nonempty")
    if any(len(c) != 2 or c[0] < 1 or c[1] < 0 for c in compositions):
        raise ValueError("compositions must contain (num_p >= 1, num_a >= 0)")
    if not 0 <= failure_prob <= 1:
        raise ValueError("failure_prob must be in [0, 1]")
    low, high = map(int, failure_window)
    if low < 0 or high < low:
        raise ValueError("failure_window must be an inclusive nonnegative interval")
    if failure_prob:
        if graded:
            # Do not condition or resample level draws on event eligibility.
            insufficient = any(p * (min(p_levels) >= 1) + a * (min(a_levels) >= 1) < 2
                               for p, a in compositions)
        else:
            insufficient = any(n_p < 2 for n_p, _ in compositions)
        if insufficient:
            raise ValueError("failure scenarios require at least two sensing agents"
                             + (" in every level draw" if graded else ""))

    scenarios = []
    for index in range(count):
        children = np.random.SeedSequence([int(seed), index]).spawn(6 if graded else 5)
        team_ss, event_ss, env_ss, action_ss, message_ss = children[:5]
        team_rng, event_rng = np.random.default_rng(team_ss), np.random.default_rng(event_ss)
        n_p, n_a = compositions[int(team_rng.integers(len(compositions)))]
        if graded:
            sensing_rng = np.random.default_rng(children[5])
            sensing = tuple(int(sensing_rng.choice(p_levels)) for _ in range(n_p)) + tuple(
                int(sensing_rng.choice(a_levels)) for _ in range(n_a))
            candidates = [agent for agent, level in enumerate(sensing) if level >= 1]
        event_step, victim = -1, -1
        if event_rng.random() < failure_prob:
            event_step = int(event_rng.integers(low, high + 1))
            victim = (candidates[int(event_rng.integers(len(candidates)))] if graded
                      else int(event_rng.integers(n_p)))
        scenario_type = GradedScenario if graded else Scenario
        scenarios.append(scenario_type(
            scenario_id=f"{int(seed)}:{index}",
            env_seed=int(env_ss.generate_state(1)[0]),
            action_seed=int(action_ss.generate_state(1)[0]),
            message_seed=int(message_ss.generate_state(1)[0]),
            num_p=n_p, num_a=n_a, event_step=event_step, victim=victim,
            **({"sensing": sensing} if graded else {})))
    return scenarios
