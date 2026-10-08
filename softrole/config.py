"""Explicit experimental defaults; these values are choices, not proved optima."""
from dataclasses import asdict, dataclass
import math


COMPOSITION_TRAIN = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1))
FAILURE_TRAIN = ((2, 1), (2, 2), (3, 1))
COMPOSITION_TEST = ((3, 2),)
EXTRAPOLATION_TEST = ((4, 1), (4, 2))


@dataclass
class Config:
    task: str = "pcp"
    env_version: str = "corrected-observation-v1"
    num_p: int = 2
    num_a: int = 1
    dim: int = 5
    vision: int = 2
    max_steps: int = 80
    nfires: int = 1
    reward_type: int = 3
    model: str = "banked"
    experts: int = 4
    pre_dim: int = 128
    hidden_dim: int = 64
    heads: int = 4
    head_dim: int = 16
    msg_dim: int = 16
    feedback: bool = True
    comm_range: float = -1.0
    seed: int = 0
    epochs: int = 2000
    updates_per_epoch: int = 10
    batch_steps: int = 500
    nprocesses: int = 4
    total_steps: int | None = None
    optimizer: str = "rmsprop"
    lr: float = 0.0001
    gamma: float = 1.0
    gae_lambda: float = 0.95
    actor_coeff: float = 50.0
    value_coeff: float = 1.0
    detach_gap: int = 5
    max_grad_norm: float = 0.75
    save_every: int = 50
    compositions: tuple = ()
    held_out: tuple = ()
    failure_prob: float = 0.0
    failure_window: tuple = (10, 30)

    def __post_init__(self):
        self.compositions = tuple(tuple(pair) for pair in self.compositions)
        self.held_out = tuple(tuple(pair) for pair in self.held_out)
        self.failure_window = tuple(self.failure_window)
        self.validate()

    @property
    def training_compositions(self):
        return self.compositions or ((self.num_p, self.num_a),)

    def model_kwargs(self):
        settings = dict(base=self.dim ** 2, n_squares=(2 * self.vision + 1) ** 2,
                    mode=self.model, experts=self.experts, pre_dim=self.pre_dim,
                    hidden_dim=self.hidden_dim, heads=self.heads,
                    head_dim=self.head_dim, msg_dim=self.msg_dim, feedback=self.feedback)
        # Missing fields in old checkpoint model_config retain the old layout.
        if self.task == "fc" and self.env_version == "paper-v1":
            settings["allow_stay"] = False
        return settings

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if self.task not in ("pp", "pcp", "fc"):
            raise ValueError("task must be pp, pcp, or fc")
        if self.env_version not in ("corrected-observation-v1", "paper-v1"):
            raise ValueError("unknown SoftRole environment version")
        if self.task == "fc" and self.env_version == "paper-v1" and self.reward_type != 3:
            raise ValueError("paper-v1 FC uses the fixed paper reward (reward_type=3)")
        if self.model not in ("banked", "shared", "capability", "constant"):
            raise ValueError("unknown deterministic model variant")
        if self.optimizer not in ("rmsprop", "adam"):
            raise ValueError("optimizer must be rmsprop or adam")
        for name in ("dim", "max_steps", "nfires", "experts", "pre_dim", "hidden_dim",
                     "heads", "head_dim", "msg_dim", "epochs", "updates_per_epoch",
                     "batch_steps", "nprocesses", "detach_gap", "save_every"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if any(type(getattr(self, name)) is not int or getattr(self, name) < 0
               for name in ("seed", "vision")):
            raise ValueError("seed and vision must be nonnegative integers")
        for name in ("lr", "max_grad_norm", "actor_coeff"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.value_coeff) or self.value_coeff < 0:
            raise ValueError("value_coeff must be finite and nonnegative")
        # This study defines an undiscounted, finite-horizon team objective.
        if self.gamma != 1.0:
            raise ValueError("gamma must be 1 for the specified finite-horizon objective")
        if not 0 <= self.gae_lambda <= 1 or not 0 <= self.failure_prob <= 1:
            raise ValueError("GAE lambda and failure probability must lie in [0,1]")
        if not math.isfinite(self.comm_range) or (self.comm_range < 0 and self.comm_range != -1):
            raise ValueError("comm_range must be -1 (all pairs) or finite and nonnegative")
        if self.total_steps is not None and (type(self.total_steps) is not int or self.total_steps <= 0):
            raise ValueError("total_steps must be a positive integer")
        if self.reward_type not in (0, 1, 2, 3):
            raise ValueError("unknown FireCommander reward type")
        if len(set(self.training_compositions)) != len(self.training_compositions):
            raise ValueError("training compositions must be unique")
        if set(self.training_compositions) & set(self.held_out):
            raise ValueError("held-out compositions must never appear in training")
        for pair in self.training_compositions + self.held_out:
            self.validate_composition(pair)
        if len(self.failure_window) != 2 or any(type(x) is not int for x in self.failure_window):
            raise ValueError("failure_window needs two integer inclusive endpoints")
        lo, hi = self.failure_window
        if not 1 <= lo <= hi:
            raise ValueError("failure times must be ordered and at least 1")
        if self.failure_prob:
            if self.task != "pcp":
                raise ValueError("sensor-failure experiments are defined only for PCP")
            if any(p < 2 for p, _ in self.training_compositions):
                raise ValueError("failure training requires at least two functioning sensors initially")
            if not 1 <= lo <= hi < self.max_steps:
                raise ValueError("failure times must be between 1 and max_steps-1")

    def validate_composition(self, pair):
        if len(pair) != 2 or any(not isinstance(x, int) or x < 0 for x in pair):
            raise ValueError("composition must contain two nonnegative integer counts")
        p, a = pair
        targets = self.nfires if self.task == "fc" else 1
        if p + a < 1 or p + a + targets > self.dim ** 2:
            raise ValueError("composition and targets must fit on the grid")
        if self.task == "pp" and (p < 1 or a != 0):
            raise ValueError("PP uses perception agents only")
        if self.task in ("pcp", "fc") and (p < 1 or a < 1):
            raise ValueError("PCP/FC studies require both sensing and actuating agents")


def recipe(task="pcp", study="fixed", **overrides):
    """Author domain sizes/budgets plus the explicitly defined new study splits."""
    settings = {"task": task}
    if task == "pp":
        settings.update(num_p=3, num_a=0)
    elif task == "fc":
        settings.update(vision=1, max_steps=300, epochs=1400)
    if study != "fixed" and task != "pcp":
        raise ValueError("composition/failure study presets are defined for PCP")
    if study == "composition":
        settings.update(compositions=COMPOSITION_TRAIN, held_out=COMPOSITION_TEST)
    elif study == "failure":
        settings.update(compositions=FAILURE_TRAIN, held_out=COMPOSITION_TEST, failure_prob=0.5)
    elif study != "fixed":
        raise ValueError("study must be fixed, composition, or failure")
    settings.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**settings)
