"""Opt-in, label-independent observations for the released HetNet domains."""
import argparse

import numpy as np

from ic3net_envs.fire_commander_env import FireCommanderEnv
from ic3net_envs.predator_capture_env import PredatorCaptureEnv

from .scenarios import IsolatedRNG


ENVIRONMENT_VERSION = "corrected-observation-v1"


def _option(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def adapt_observation(raw_obs, kappa, base):
    """x = [position one-hots, sense*(outside,target,P+A), own kappa].

    The occupancy sum preserves counts, including co-located agents, but removes
    information about neighbor physical types. No internal completion flags enter
    the actor. The transform is pure and never writes to simulator observations.
    """
    raw_obs, kappa = np.asarray(raw_obs), np.asarray(kappa, dtype=np.float64)
    if raw_obs.ndim < 2 or kappa.shape != (raw_obs.shape[0], 2):
        raise ValueError("observation and kappa must share the agent dimension")
    if np.any((kappa != 0) & (kappa != 1)):
        raise ValueError("kappa entries must be binary current capabilities")
    if raw_obs.size % (raw_obs.shape[0] * (base + 4)):
        raise ValueError("native observations require base+4 features per cell")
    cells = raw_obs.reshape(raw_obs.shape[0], -1, base + 4)
    sensory = np.stack((cells[..., base + 1], cells[..., base + 2],
                        cells[..., base] + cells[..., base + 3]), axis=-1)
    sensory = sensory * kappa[:, None, :1]
    return np.concatenate((cells[..., :base].reshape(len(cells), -1),
                           sensory.reshape(len(cells), -1), kappa), axis=1).astype(np.float64)


class EnvironmentAdapter:
    environment_version = ENVIRONMENT_VERSION

    def __init__(self, config):
        self.task = _option(config, "task", "pcp")
        if self.task not in ("pp", "pcp", "fc"):
            raise ValueError("task must be pp, pcp, or fc")
        self.dim = int(_option(config, "dim", 5))
        self.vision = int(_option(config, "vision", 1 if self.task == "fc" else 2))
        self.max_steps = int(_option(config, "max_steps", 300 if self.task == "fc" else 80))
        self.num_p = int(_option(config, "num_p", 3 if self.task == "pp" else 2))
        self.num_a = int(_option(config, "num_a", 0 if self.task == "pp" else 1))
        if self.dim < 1 or self.vision < 0 or self.max_steps < 1:
            raise ValueError("dim/max_steps must be positive and vision nonnegative")
        self.base = self.dim ** 2
        self.observation_dim = (2 * self.vision + 1) ** 2 * (self.base + 3) + 2
        self.raw = FireCommanderEnv() if self.task == "fc" else PredatorCaptureEnv()
        parser = argparse.ArgumentParser(add_help=False)
        self.raw.init_args(parser)
        self.args = parser.parse_args([])
        self.args.dim, self.args.vision = self.dim, self.vision
        self.args.max_steps = self.max_steps
        self.args.nfriendly_P, self.args.nfriendly_A = self.num_p, self.num_a
        self.args.eval = False
        if self.task == "fc":
            self.args.nfires = int(_option(config, "nfires", 1))
            self.args.reward_type = int(_option(config, "reward_type", 3))
            self.args.shared_reward = bool(_option(config, "shared_reward", False))
            if self.args.nfires < 1 or self.args.reward_type not in range(4):
                raise ValueError("FC requires nfires >= 1 and reward_type in [0,3]")
        self.raw.multi_agent_init(self.args)
        self.raw.independent_observations = True
        self._raw_obs = None
        self._rng = None
        self._done = False
        self.kappa = None
        self._set_composition(self.num_p, self.num_a)

    def _set_composition(self, num_p, num_a):
        num_p, num_a = int(num_p), int(num_a)
        if num_p < 1 or num_a < 0:
            raise ValueError("require num_p >= 1 and num_a >= 0")
        if self.task == "pp" and num_a != 0:
            raise ValueError("the PP recipe requires zero actuation agents")
        if self.task != "pp" and num_a < 1:
            raise ValueError("PCP/FC require at least one actuation agent")
        targets = self.args.nfires if self.task == "fc" else self.args.nenemies
        if num_p + num_a + targets > self.base:
            raise ValueError("initial agents and targets must fit distinct map cells")
        self.num_p, self.num_a = num_p, num_a
        self.args.nfriendly_P, self.args.nfriendly_A = num_p, num_a
        self.args.nagents = num_p + num_a
        self.raw.nfriendly_P, self.raw.nfriendly_A = num_p, num_a
        self.raw.npredator, self.raw.npredator_capture = num_p, num_a
        self.raw.predator_capture_index = num_p
        if self.task == "fc":
            self.raw.captured_fire_index = num_p + num_a
        else:
            self.raw.captured_prey_index = num_p + num_a
            self.raw.state_len = num_p + num_a + 1
        self.native_kappa = np.array([[1., 0.]] * num_p + [[0., 1.]] * num_a)

    def _validate_kappa(self, kappa):
        kappa = np.asarray(kappa, dtype=np.float64)
        if kappa.shape != self.native_kappa.shape or np.any((kappa != 0) & (kappa != 1)):
            raise ValueError("kappa must be an Nx2 array of binary capabilities")
        if not np.array_equal(kappa[:, 1], self.native_kappa[:, 1]):
            raise ValueError("changes to physical actuation capability are not implemented")
        if self.task != "pcp" and not np.array_equal(kappa, self.native_kappa):
            raise ValueError("sensor-change scenarios are implemented only for PCP")
        if np.any(kappa[:, 0] > self.native_kappa[:, 0]):
            raise ValueError("the adapter cannot add a sensor to a physically blind agent")
        return kappa

    def reset(self, seed, num_p=None, num_a=None):
        self._set_composition(self.num_p if num_p is None else num_p,
                              self.num_a if num_a is None else num_a)
        self._rng = IsolatedRNG(seed)
        with self._rng.use():
            self._raw_obs = np.array(self.raw.reset(), copy=True)
        self._steps, self._done = 0, False
        self.kappa = self.native_kappa.copy()
        return self.observe(self.kappa), self.kappa.copy()

    @property
    def positions(self):
        if self._raw_obs is None:
            raise RuntimeError("reset the environment before reading positions")
        return np.vstack((self.raw.predator_loc, self.raw.predator_capture_loc)).copy()

    @property
    def raw_observation(self):
        """Independent typed observation copy for the contextual legacy baseline."""
        if self._raw_obs is None:
            raise RuntimeError("reset the environment before observing")
        return self._raw_obs.copy()

    def observe(self, kappa):
        if self._raw_obs is None:
            raise RuntimeError("reset the environment before observing")
        return adapt_observation(self._raw_obs, self._validate_kappa(kappa), self.base)

    def step(self, actions, kappa):
        if self._raw_obs is None or self._done:
            raise RuntimeError("reset the environment before stepping a new episode")
        kappa = self._validate_kappa(kappa)
        actions = np.asarray(actions)
        if actions.shape != (self.num_p + self.num_a,) or np.any(actions != actions.astype(int)):
            raise ValueError("actions must be one integer per agent")
        if np.any((actions < 0) | (actions > 5)):
            raise ValueError("actions must be in [0,5]")
        if np.any((actions == 5) & (kappa[:, 1] == 0)):
            raise ValueError("actuation action is unavailable to this agent")
        with self._rng.use():
            obs, rewards, terminated, info = self.raw.step(actions.astype(int))
        self._raw_obs = np.array(obs, copy=True)
        self._steps += 1
        self._done = bool(terminated or self._steps >= self.max_steps)
        self.kappa = kappa.copy()
        info = dict(info)
        info.update(success=bool(self.raw.stat.get("success", False)),
                    terminated=bool(terminated),
                    truncated=bool(not terminated and self._steps >= self.max_steps),
                    environment_version=self.environment_version)
        return self.observe(kappa), np.asarray(rewards, dtype=np.float64), self._done, info


def make_env(config):
    return EnvironmentAdapter(config)
