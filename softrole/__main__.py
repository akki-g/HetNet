"""Standalone training, frozen evaluation, native PCP pilot and summaries."""
import argparse
import json
from pathlib import Path

from softrole.config import Config, recipe


def composition(text):
    try:
        p, a = map(int, text.split(","))
        if p < 1 or a < 0:
            raise ValueError
        return p, a
    except ValueError as error:
        raise argparse.ArgumentTypeError("use a composition such as 2,1 (P,A counts)") from error


def parser():
    cli = argparse.ArgumentParser(description="Class-independent capability-conditioned HetNet")
    commands = cli.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train in a fresh run directory")
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--resume", type=Path, help="resume into a new output directory")
    train.add_argument("--episode-log", choices=("file", "stdout"), default="file",
                       help="episode records: buffered JSONL file or tagged stdout batch per update")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--task", choices=("pp", "pcp", "fc"))
    train.add_argument("--study", choices=("fixed", "composition", "failure"))
    train.add_argument("--model", choices=("banked", "shared", "capability", "constant"))
    # Familiar release spellings are aliases, not a second set of defaults.
    integers = {
        "num_p": ("--num-p", "--nfriendly_P"), "num_a": ("--num-a", "--nfriendly_A"),
        "dim": ("--dim",), "vision": ("--vision",), "max_steps": ("--max-steps", "--max_steps"),
        "nfires": ("--nfires",), "reward_type": ("--reward-type", "--reward_type"),
        "experts": ("--experts",), "pre_dim": ("--pre-dim",), "hidden_dim": ("--hidden-dim",),
        "heads": ("--heads",), "head_dim": ("--head-dim",), "msg_dim": ("--msg-dim", "--msg_dim"),
        "seed": ("--seed",), "epochs": ("--epochs", "--num_epochs"),
        "updates_per_epoch": ("--updates-per-epoch", "--epoch_size"),
        "batch_steps": ("--batch-steps", "--batch_size"), "nprocesses": ("--nprocesses",),
        "total_steps": ("--total-steps",), "detach_gap": ("--detach-gap", "--detach_gap"),
        "save_every": ("--save-every", "--save_every"),
    }
    for name, flags in integers.items():
        train.add_argument(*flags, dest=name, type=int)
    for name, flags in {
        "comm_range": ("--comm-range",), "lr": ("--lr", "--lrate"),
        "gamma": ("--gamma",), "gae_lambda": ("--gae-lambda",),
        "actor_coeff": ("--actor-coeff",), "value_coeff": ("--value-coeff",),
        "max_grad_norm": ("--max-grad-norm",), "failure_prob": ("--failure-prob",),
    }.items():
        train.add_argument(*flags, dest=name, type=float)
    train.add_argument("--no-feedback", dest="feedback", action="store_false", default=None)
    train.add_argument("--compositions", nargs="+", type=composition)
    train.add_argument("--held-out", nargs="+", type=composition)
    train.add_argument("--failure-window", nargs=2, type=int)

    evaluate = commands.add_parser("evaluate", help="evaluate a frozen SoftRole checkpoint")
    legacy = commands.add_parser("evaluate-hetnet", help="contextual typed HetNet composition reference")
    for command in (evaluate, legacy):
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--episodes", type=int, default=500,
                             help="number of scenarios PER composition")
        command.add_argument("--seed", type=int, default=0, help="evaluation scenario seed")
        command.add_argument("--compositions", nargs="+", type=composition)
        command.add_argument("--scenarios", type=Path, help="JSON list of explicit Scenario records")
        command.add_argument("--trace", action="store_true")
    evaluate.add_argument("--failure-prob", type=float, default=0)
    evaluate.add_argument("--protocol", type=Path, help="declared selection/distribution sidecar; requires --scenarios")
    evaluate.add_argument("--failure-window", type=int, nargs=2, default=(10, 30))
    evaluate.add_argument("--intervention", choices=("none", "freeze_affected", "freeze_all", "comm_off"),
                          default="none")
    evaluate.add_argument("--intervention-step", type=int)
    evaluate.add_argument("--sham", action="store_true",
                          help="retain scheduled event and intervention, but suppress physical failure")
    legacy.add_argument("--task", choices=("pp", "pcp", "fc"), default="pcp")
    legacy.add_argument("--variant", choices=("real", "binary"), default="binary")
    legacy.add_argument("--dim", type=int)
    legacy.add_argument("--vision", type=int)
    legacy.add_argument("--max-steps", type=int)
    legacy.add_argument("--msg-dim", type=int)
    legacy.add_argument("--comm-range", type=float)

    pilot = commands.add_parser("pilot-failure", help="paired nominal PCP 2P1A diagnostic pilot")
    pilot.add_argument("--checkpoints", nargs="+", type=Path, required=True,
                       help="one nominal shared and banked checkpoint per training seed")
    pilot.add_argument("--checkpoint-rule", required=True, help="predeclared selection rule, recorded verbatim")
    pilot.add_argument("--output", type=Path, required=True, help="fresh pilot directory")
    pilot.add_argument("--episodes", type=int, default=20, help="scenarios per policy and condition")
    pilot.add_argument("--seed", type=int, default=1700, help="evaluation scenario seed")

    report = commands.add_parser("summarize", help="seed-level uncertainty, stratified by composition")
    report.add_argument("reports", nargs="+", type=Path)
    report.add_argument("--output", type=Path)
    report.add_argument("--bootstrap-samples", type=int, default=10000)
    report.add_argument("--seed", type=int, default=0)
    report.add_argument("--protocol", type=Path, help="validate a common protocol sidecar, including older reports")
    return cli


def scenarios_for(args, config):
    from softrole.scenarios import Scenario, make_scenarios
    if args.scenarios:
        with args.scenarios.open() as stream:
            records = json.load(stream)
        return [Scenario(**record) for record in records]
    if args.episodes <= 0 or args.seed < 0:
        raise ValueError("episodes must be positive and the scenario seed nonnegative")
    compositions = args.compositions or config.held_out or ((config.num_p, config.num_a),)
    if len(set(compositions)) != len(compositions):
        raise ValueError("evaluation compositions must be unique")
    scenarios = []
    for p, a in compositions:
        config.validate_composition((p, a))
        # Composition-specific seed namespaces are order independent. Every model
        # receives the same scenarios when using the same evaluation seed.
        import numpy as np
        seed = int(np.random.SeedSequence([args.seed, p, a]).generate_state(1)[0])
        scenarios.extend(make_scenarios(seed, args.episodes, [(p, a)],
                         getattr(args, "failure_prob", 0), getattr(args, "failure_window", (10, 30))))
    return scenarios


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.command == "train":
            options = vars(args).copy()
            for key in ("command", "output", "resume", "dry_run", "task", "study", "episode_log"):
                options.pop(key)
            if args.resume:
                import torch
                if args.study is not None or args.task is not None:
                    raise ValueError("resume retains its checkpoint's task and study")
                saved = torch.load(args.resume, map_location="cpu", weights_only=False)
                values = saved["config"].copy()
                values.update({key: value for key, value in options.items() if value is not None})
                config = Config(**values)
            else:
                config = recipe(args.task or "pcp", args.study or "fixed", **options)
            if args.dry_run:
                print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
            else:
                from softrole.train import train
                train(config, args.output, args.resume, episode_log=args.episode_log)
        elif args.command == "evaluate":
            import torch
            from softrole.evaluate import evaluate_checkpoint
            saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            config = Config(**saved["config"])
            scenarios = scenarios_for(args, config)
            if args.protocol and not args.scenarios:
                raise ValueError("--protocol requires an explicit --scenarios file")
            import hashlib
            report = evaluate_checkpoint(args.checkpoint, scenarios, args.output,
                         args.intervention, args.intervention_step, args.trace, sham=args.sham,
                         protocol=json.loads(args.protocol.read_text()) if args.protocol else None,
                         scenarios_sha256=hashlib.sha256(args.scenarios.read_bytes()).hexdigest()
                             if args.scenarios else None)
            print(json.dumps({key: value for key, value in report.items()
                              if key not in ("per_episode", "scenarios", "config", "model_config", "evaluator")}, indent=2))
        elif args.command == "evaluate-hetnet":
            from softrole.hetnet import evaluate_hetnet
            config = recipe(args.task, **{name: getattr(args, name) for name in
                            ("dim", "vision", "max_steps", "msg_dim", "comm_range")})
            report = evaluate_hetnet(args.checkpoint, config, scenarios_for(args, config),
                                    args.output, use_binary=args.variant == "binary", trace=args.trace)
            print(json.dumps({key: value for key, value in report.items()
                              if key not in ("per_episode", "scenarios", "config", "model_config")}, indent=2))
        elif args.command == "pilot-failure":
            from softrole.pilot import run_pilot
            report = run_pilot(args.checkpoints, args.output, args.checkpoint_rule, args.episodes, args.seed)
            print(json.dumps(report, indent=2, allow_nan=False))
        else:
            from softrole.report import summarize_reports
            report = summarize_reports(args.reports, args.output, args.bootstrap_samples, args.seed,
                                       protocol=args.protocol)
            print(json.dumps(report, indent=2, allow_nan=False))
    except (ValueError, FileExistsError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    main()
