"""Explicit optimizer recipes and fail-closed checkpoint compatibility.

Optimizer selection changes neither the episode objective nor its aggregation.
Old checkpoints without a selector retain the historical RMSprop recipe.
"""


def optimizer_spec(config):
    common = dict(lr=config.lr, weight_decay=0, maximize=False, differentiable=False)
    if config.optimizer == "rmsprop":
        return {"name": "RMSprop", "parameters": dict(
            common, alpha=0.97, eps=1e-6, momentum=0, centered=False, foreach=None)}
    if config.optimizer == "adam":
        return {"name": "Adam", "parameters": dict(
            common, betas=[0.9, 0.999], eps=1e-8, amsgrad=False,
            foreach=False, fused=False, capturable=False)}
    raise ValueError("optimizer must be rmsprop or adam")


def make_optimizer(parameters, config):
    import torch

    spec = optimizer_spec(config)
    options = spec["parameters"].copy()
    if config.optimizer == "adam":
        options["betas"] = tuple(options["betas"])
        return torch.optim.Adam(parameters, **options)
    return torch.optim.RMSprop(parameters, **options)


def validate_optimizer_resume(config, checkpoint):
    """Validate configuration AND saved groups before loading optimizer state.

    A missing selector/recipe is supported only for legacy RMSprop checkpoints.
    Switching optimizers is a fresh experiment, never a continuation.
    """
    previous = checkpoint["config"].get("optimizer", "rmsprop")
    if previous != config.optimizer:
        raise ValueError(f"cannot resume {previous} with {config.optimizer}; start a fresh run")
    expected = optimizer_spec(config)
    saved = checkpoint.get("optimizer")
    if saved is not None and saved != expected:
        raise ValueError("resume optimizer metadata differs from the requested recipe")
    if saved is None and config.optimizer != "rmsprop":
        raise ValueError("Adam resume requires explicit optimizer metadata")
    groups = checkpoint["optimizer_state"].get("param_groups", [])
    if not groups:
        raise ValueError("resume checkpoint has no optimizer parameter groups")
    for group in groups:
        for name, value in expected["parameters"].items():
            actual = group.get(name)
            if name == "betas" and isinstance(actual, (tuple, list)):
                actual = list(actual)
            if name not in group or actual != value:
                raise ValueError(f"resume optimizer parameter {name} differs from the requested recipe")
        if (config.optimizer == "rmsprop" and "betas" in group or
                config.optimizer == "adam" and "alpha" in group):
            raise ValueError("resume optimizer state belongs to a different optimizer")
    required = ({"step", "square_avg"} if config.optimizer == "rmsprop"
                else {"step", "exp_avg", "exp_avg_sq"})
    for state in checkpoint["optimizer_state"].get("state", {}).values():
        if set(state) != required:
            raise ValueError("resume optimizer moments differ from the requested recipe")
