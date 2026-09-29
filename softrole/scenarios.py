"""Architecture-independent scenarios and isolated legacy simulator randomness."""
from contextlib import contextmanager
from dataclasses import dataclass
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


def make_scenarios(seed, count, compositions, failure_prob=0.0,
                   failure_window=(10, 30)):
    """Return repeatable episode scenarios, with stable prefixes as count grows.

    Event steps are zero-based and the window includes both endpoints. Failure
    scenarios retain at least one sensing teammate, so their compositions must
    have at least two P agents. Fully blind tests may explicitly build Scenario.
    Task-specific event support is checked by the environment adapter.
    """
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
    if failure_prob and any(n_p < 2 for n_p, _ in compositions):
        raise ValueError("failure scenarios require at least two sensing agents")

    scenarios = []
    for index in range(count):
        team_ss, event_ss, env_ss, action_ss, message_ss = np.random.SeedSequence(
            [int(seed), index]).spawn(5)
        team_rng, event_rng = np.random.default_rng(team_ss), np.random.default_rng(event_ss)
        n_p, n_a = compositions[int(team_rng.integers(len(compositions)))]
        event_step, victim = -1, -1
        if event_rng.random() < failure_prob:
            event_step = int(event_rng.integers(low, high + 1))
            victim = int(event_rng.integers(n_p))
        scenarios.append(Scenario(
            scenario_id=f"{int(seed)}:{index}",
            env_seed=int(env_ss.generate_state(1)[0]),
            action_seed=int(action_ss.generate_state(1)[0]),
            message_seed=int(message_ss.generate_state(1)[0]),
            num_p=n_p, num_a=n_a, event_step=event_step, victim=victim))
    return scenarios
