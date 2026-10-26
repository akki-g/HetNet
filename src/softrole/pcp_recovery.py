"""Locked six-run recovery experiment after the two observed nominal regressions.

This is a new experimental branch, not a protocol override of a research run.
Historical onset states require exact archived replay before any branch launches.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import fcntl
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from softrole.config import Config
from softrole.ledger import read_json, sha, canonical, committed_checkpoints, iter_episodes
from softrole.train import source_snapshot, write_json, train

STUDY = "pcp-onset-recovery-v1"
ARCHIVE_SOURCE = "e9a58193d6739283fc030999641acc0338032646c81063fd548bff9fe9aff5c7"
ONSETS = {4: (17537, 35273245, 1750, 1760), 5: (19588, 39432334, 1950, 1960)}
VARIANTS = (("control", .001, 0.), ("lower_lr", .0001, 0.), ("entropy", .001, .001))
ADDITIONAL_STEPS = 5_000_000
PANEL_SEED = 92704
PANEL_EPISODES = 500


def normalized(config):
    return json.loads(json.dumps(config.to_dict()))


def require_index(index, count):
    if type(index) is not int or not 0 <= index < count:
        raise ValueError(f"index must be 0..{count - 1}")


def parent_record(archive, seed):
    run = Path(archive).resolve() / "l1a_decomposed_beta0" / f"seed{seed}" / "training"
    update, steps, before, after = ONSETS[seed]
    from softrole.programme import programme_config, CAPCON_PROFILE
    cfg = Config(**read_json(run / "config.json"))
    if normalized(cfg) != normalized(programme_config("l1a_decomposed_beta0", seed, .001, profile=CAPCON_PROFILE)):
        raise ValueError("Historical configuration differs from the declared beta0 research run")
    source = read_json(run / "source_manifest.json")
    runtime = read_json(run / "run.json")
    if source["sha256"] != ARCHIVE_SOURCE or runtime["source_sha256"] != ARCHIVE_SOURCE:
        raise ValueError("Unknown historical source; this protocol is specific to array 1004964")
    identity = hashlib.sha256(json.dumps(source["files"], sort_keys=True).encode()).hexdigest()
    if identity != ARCHIVE_SOURCE:
        raise ValueError("Historical source manifest is malformed")
    archive_source = (run / "source").resolve()
    for name, digest in source["files"].items():
        file = (archive_source / name).resolve()
        if not file.is_relative_to(archive_source) or sha(file) != digest:
            raise ValueError("Historical source bytes changed")
    rows = {r["update"]: r for c in read_json(run / "episodes.index.json")["chunks"] for r in c["updates"]}
    if (any(rows[u]["successes"] != rows[u]["episodes"] for u in range(update - 100, update))
            or rows[update]["successes"] >= rows[update]["episodes"]):
        raise ValueError("Declared onset does not follow 100 all-success updates")
    # Independently recalculate the indexed onset window from immutable raw
    # chunks. Avoid streaming millions of unrelated historical episodes.
    from softrole.ledger import empty_stats, add_episode, _safe_path, _validate_episode
    raw_chunks = []
    for chunk in read_json(run / "episodes.index.json")["chunks"]:
        if chunk["last_update"] < update - 100 or chunk["first_update"] > update:
            continue
        raw = _safe_path(run, chunk["path"])
        if sha(raw) != chunk["sha256"] or raw.stat().st_size != chunk["bytes"]:
            raise ValueError("Historical onset episode chunk changed")
        actual = {}
        with gzip.open(raw, "rt") as stream:
            for line in stream:
                episode = json.loads(line)
                _validate_episode(episode, cfg)
                add_episode(actual.setdefault(episode["update"], empty_stats()), episode, cfg.max_steps)
        if [dict(update=u, **actual[u]) for u in sorted(actual)] != chunk["updates"]:
            raise ValueError("Historical onset index differs from raw episode evidence")
        raw_chunks.append(chunk["path"])
    observed = [json.loads(line) for line in (run / "updates.jsonl").open()]
    if next(r for r in observed if r["update"] == update)["total_steps"] != steps:
        raise ValueError("Historical onset count mismatch")
    names = ["config.json", "run.json", "source_manifest.json", "episodes.index.json", "updates.jsonl"] + raw_chunks
    anchors = [run / "checkpoints" / f"epoch{epoch:04d}.pt" for epoch in (before, after)]
    for checkpoint in anchors:
        marker = read_json(str(checkpoint) + ".complete.json")
        item = next(x for x in marker["files"] if x["path"] == str(checkpoint.relative_to(run)))
        if item["sha256"] != sha(checkpoint):
            raise ValueError("Historical checkpoint differs from its commit")
        names.extend(str(checkpoint.relative_to(run)) + suffix for suffix in ("", ".complete.json", ".signature.json"))
    return dict(seed=seed, run=str(run), update=update, total_steps=steps,
                anchor_before=str(anchors[0]), anchor_after=str(anchors[1]), config=normalized(cfg),
                runtime={k: runtime[k] for k in ("torch", "numpy")},
                inputs={name: sha(run / name) for name in names})


def branch_config(parent, variant, additional_steps=ADDITIONAL_STEPS):
    _, lr, entropy = variant
    config = Config(**parent["config"])
    # Retain ten updates per epoch; increase the administrative ceiling so each
    # branch receives its full new budget even when onset was near epoch 2000.
    max_updates = parent["update"] + math.ceil(additional_steps / (config.batch_steps * config.nprocesses)) + 1
    return replace(config, lr=lr, entropy_coeff=entropy,
                   total_steps=parent["total_steps"] + additional_steps,
                   epochs=max(config.epochs, math.ceil(max_updates / config.updates_per_epoch)))


def entries(parents, root, additional_steps):
    return [dict(index=i * 3 + j, seed=p["seed"], variant=v[0],
                 output=str(root / "branches" / f"seed{p['seed']}" / v[0]),
                 config=normalized(branch_config(p, v, additional_steps)))
            for i, p in enumerate(parents) for j, v in enumerate(VARIANTS)]


def panel():
    from softrole.scenarios import make_scenarios
    return [asdict(s) for s in make_scenarios(PANEL_SEED, PANEL_EPISODES, [(2, 1)])]


def prepare(archive_root, output, additional_steps=ADDITIONAL_STEPS):
    if type(additional_steps) is not int or additional_steps != ADDITIONAL_STEPS:
        raise ValueError("This protocol fixes 5M additional training steps per branch")
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(root)
    parents = [parent_record(archive_root, seed) for seed in ONSETS]
    manifest = dict(version=1, study=STUDY, root=str(root), archive_root=str(Path(archive_root).resolve()),
                    additional_steps=additional_steps, source_sha256=source_snapshot()["sha256"],
                    parents=parents, entries=entries(parents, root, additional_steps),
                    boundary="after the first failed batch and its optimizer step; no anticipatory intervention",
                    entropy_rule="50 * (policy_loss - entropy_coeff * sum_agent_time_entropy) + value_loss",
                    panel_seed=PANEL_SEED, panel_episodes=PANEL_EPISODES, panel_sha256=canonical(panel()),
                    interpretation="post-hoc two-seed recovery diagnostic; no final research selection")
    root.mkdir(parents=True)
    write_json(root / "panel.json", panel())
    write_json(root / "manifest.json", manifest)
    return manifest


def load_manifest(path, check_inputs=True):
    path = Path(path).resolve()
    m = read_json(path)
    if (m.get("version") != 1 or m.get("study") != STUDY or m.get("root") != str(path.parent)
            or m.get("additional_steps") != ADDITIONAL_STEPS or [p["seed"] for p in m["parents"]] != list(ONSETS)):
        raise ValueError("Malformed recovery manifest")
    for parent in m["parents"]:
        u, steps, _, _ = ONSETS[parent["seed"]]
        if parent["update"] != u or parent["total_steps"] != steps:
            raise ValueError("Manifest onset mismatch")
    if m["entries"] != entries(m["parents"], path.parent, m["additional_steps"]):
        raise ValueError("Manifest entry order/settings/output mismatch")
    if (m["panel_seed"] != PANEL_SEED or m["panel_episodes"] != PANEL_EPISODES
            or read_json(path.parent / "panel.json") != panel() or m["panel_sha256"] != canonical(panel())):
        raise ValueError("Recovery panel identity mismatch")
    if source_snapshot()["sha256"] != m["source_sha256"]:
        raise ValueError("Source differs from locked recovery manifest")
    if check_inputs and [parent_record(m["archive_root"], s) for s in ONSETS] != m["parents"]:
        raise ValueError("Historical inputs changed since manifest preparation")
    return m


def boundary_record(m, parent):
    directory = Path(m["root"]) / "boundaries" / f"seed{parent['seed']}"
    admission = read_json(directory / "admission.json")
    checkpoint = directory / "boundary.pt"
    if (admission.get("admitted") is not True or admission["boundary_update"] != parent["update"]
            or admission["total_steps"] != parent["total_steps"]
            or admission["source_sha256"] != ARCHIVE_SOURCE or admission["runtime"] != parent["runtime"]
            or admission["boundary_checkpoint_sha256"] != sha(checkpoint)
            or admission["anchor_before"] != parent["anchor_before"] or admission["anchor_after"] != parent["anchor_after"]):
        raise ValueError("Onset replay admission mismatch")
    for path, digest in admission["input_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Historical replay evidence changed")
    return checkpoint, admission


def reconstruct(manifest, index):
    require_index(index, 2)
    m = load_manifest(manifest)
    p = m["parents"][index]
    destination = Path(m["root"]) / "boundaries" / f"seed{p['seed']}"
    command = [sys.executable, "-B", str(Path(__file__).with_name("recovery_replay.py")),
               "--run-dir", p["run"], "--output", str(destination),
               "--boundary-update", str(p["update"]), "--anchor-before", p["anchor_before"],
               "--anchor-after", p["anchor_after"]]
    subprocess.run(command, check=True)
    boundary, _ = boundary_record(m, p)
    from softrole.evaluate import evaluate_checkpoint
    from softrole.scenarios import Scenario
    evaluate_checkpoint(boundary, [Scenario(**s) for s in panel()], destination / "nominal.json",
                        scenarios_sha256=m["panel_sha256"])
    return str(boundary)


def branch_payload(boundary, config, provenance):
    """Explicit derived initialization; preserve tensors/moments and change only the declared recipe."""
    import torch
    from softrole.optimizers import optimizer_spec, validate_optimizer_resume
    original = torch.load(boundary, map_location="cpu", weights_only=False)
    validate_optimizer_resume(Config(**original["config"]), original)
    old, new = Config(**original["config"]).to_dict(), config.to_dict()
    for key in ("lr", "entropy_coeff", "epochs", "total_steps"):
        old.pop(key, None)
        new.pop(key, None)
    if old != new:
        raise ValueError("Recovery branch may change only LR, entropy and administrative budget")
    saved = deepcopy(original)
    saved.update(config=config.to_dict(), optimizer=optimizer_spec(config),
                 source_sha256=provenance["source_sha256"], recovery_parent=provenance)
    for group in saved["optimizer_state"]["param_groups"]:
        group["lr"] = config.lr
    validate_optimizer_resume(config, saved)
    return saved


def initialize_branch(boundary, config, destination, provenance):
    import torch
    saved = branch_payload(boundary, config, provenance)
    with Path(destination).open("xb") as stream:
        torch.save(saved, stream)
    return saved


def audit_attempt(checkpoint, parent, config, source_id):
    """Audit a committed continuation prefix with nonzero inherited counts."""
    import torch
    from softrole.ledger import _safe_path, _validate_episode
    checkpoint = Path(checkpoint)
    run = checkpoint.parent.parent
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if normalized(Config(**saved["config"])) != normalized(config) or saved["source_sha256"] != source_id:
        raise ValueError("Continuation checkpoint config/source mismatch")
    marker = read_json(str(checkpoint) + ".complete.json")
    for item in marker["files"]:
        path = _safe_path(run, item["path"])
        if sha(path, item["bytes"]) != item["sha256"]:
            raise ValueError("Continuation committed evidence changed")
    index = read_json(run / "episodes.index.json")
    indexed = {c["path"]: c["sha256"] for c in index["chunks"] if c["first_update"] <= saved["updates"]}
    committed = {f["path"]: f["sha256"] for f in marker["files"] if f["path"].endswith(".gz")}
    if indexed != committed:
        raise ValueError("Continuation index differs from committed chunks")
    episodes = iter(iter_episodes(run, saved["updates"]))
    u, steps, count = (parent[k] for k in ("updates", "total_steps", "total_episodes"))
    update_file = next(f for f in marker["files"] if f["path"] == "updates.jsonl")
    with (run / "updates.jsonl").open("rb") as stream:
        lines = stream.read(update_file["bytes"]).splitlines()
    for line in lines:
        row = json.loads(line)
        if row["update"] > saved["updates"]:
            break
        u += 1
        if row["update"] != u:
            raise ValueError("Continuation has gapped/duplicate updates")
        batch_steps, collectors, identities = 0, set(), set()
        for _ in range(row["episodes"]):
            e = next(episodes)
            _validate_episode(e, config)
            identity = (e["collector"], e["scenario_id"])
            if e["update"] != u or identity in identities:
                raise ValueError("Continuation episode identity mismatch")
            identities.add(identity)
            collectors.add(e["collector"])
            batch_steps += e["steps"]
        steps += batch_steps
        count += row["episodes"]
        if collectors != set(range(config.nprocesses)) or row["steps"] != batch_steps or row["total_steps"] != steps:
            raise ValueError("Continuation episode/update counts mismatch")
    if next(episodes, None) is not None or (u, steps, count) != tuple(saved[k] for k in ("updates", "total_steps", "total_episodes")):
        raise ValueError("Continuation checkpoint counts mismatch")
    return saved


def accepted_branch(entry, manifest_hash, source_id, *, boundary=None, provenance=None):
    """Each new attempt explicitly supersedes its parent's unpublished tail."""
    import torch
    root = Path(entry["output"])
    config = Config(**entry["config"])
    initial = root / "initial.pt"
    identity = read_json(root / "initial.json")
    if identity != dict(manifest_sha256=manifest_hash, checkpoint_sha256=sha(initial)):
        raise ValueError("Branch initialization identity mismatch")
    parent_path = initial
    parent = torch.load(initial, map_location="cpu", weights_only=False)
    if boundary is not None:
        from softrole.recovery_replay import exact
        if not exact(parent, branch_payload(boundary, config, provenance)):
            raise ValueError("Branch initialization differs from the admitted onset state")
    accepted = []
    for number, path in enumerate(sorted((root / "attempts").glob("[0-9][0-9][0-9][0-9].json"))):
        metadata = read_json(path)
        expected = dict(number=number, manifest_sha256=manifest_hash, parent_checkpoint=str(parent_path.resolve()),
                        parent_sha256=sha(parent_path))
        if metadata != expected:
            raise ValueError("Invalid continuation attempt ancestry")
        directory = path.with_suffix("")
        records = committed_checkpoints(directory, verify_ledgers=False)
        if records:
            latest = records[-1]
            saved = audit_attempt(latest["checkpoint"], parent, config, source_id)
            if saved["updates"] <= parent["updates"]:
                raise ValueError("Continuation made no accepted progress")
            accepted.extend(records)
            parent_path, parent = Path(latest["checkpoint"]), saved
    return parent_path, parent, accepted


def evaluate_entry(m, entry, checkpoint):
    import numpy as np
    import torch
    from softrole.evaluate import evaluate_checkpoint
    from softrole.scenarios import Scenario
    destination = Path(entry["output"]) / "nominal.json"
    receipt = destination.with_suffix(".identity.json")
    if destination.exists():
        report = read_json(destination)
        if (report["checkpoint_sha256"] != sha(checkpoint) or report.get("scenarios_sha256") != m["panel_sha256"]
                or not receipt.exists() or read_json(receipt)["report_sha256"] != sha(destination)):
            raise ValueError("Existing nominal report has a different checkpoint/panel")
        return report
    report = evaluate_checkpoint(checkpoint, [Scenario(**s) for s in panel()], destination,
                                 scenarios_sha256=m["panel_sha256"])
    write_json(receipt, dict(report_sha256=sha(destination), runtime={"torch": torch.__version__, "numpy": np.__version__},
                            manifest_sha256=sha(Path(m["root"]) / "manifest.json")))
    return report


def branch_provenance(m, entry, boundary, manifest_hash):
    return dict(source_sha256=m["source_sha256"], manifest_sha256=manifest_hash,
                boundary_sha256=sha(boundary), admission_sha256=sha(boundary.parent / "admission.json"),
                variant=entry["variant"], seed=entry["seed"])


def launch(manifest, index, dry_run=False, resume=False):
    require_index(index, 6)
    m = load_manifest(manifest)
    entry, parent = m["entries"][index], m["parents"][index // 3]
    config = Config(**entry["config"])
    if dry_run:
        return entry
    import numpy as np
    import torch
    if parent["runtime"] != {"torch": torch.__version__, "numpy": np.__version__}:
        raise ValueError("Recovery requires the historical Torch/NumPy runtime")
    boundary, admission = boundary_record(m, parent)
    root = Path(entry["output"])
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / (root.name + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Recovery entry already running") from error
        manifest_hash = sha(manifest)
        provenance = branch_provenance(m, entry, boundary, manifest_hash)
        if not resume:
            root.mkdir()
            initialize_branch(boundary, config, root / "initial.pt", provenance)
            write_json(root / "initial.json", dict(manifest_sha256=manifest_hash, checkpoint_sha256=sha(root / "initial.pt")))
        parent_path, saved, _ = accepted_branch(entry, manifest_hash, m["source_sha256"], boundary=boundary, provenance=provenance)
        if saved["total_steps"] < config.total_steps:
            attempts = root / "attempts"
            attempts.mkdir(exist_ok=True)
            number = len(list(attempts.glob("[0-9][0-9][0-9][0-9].json")))
            destination = attempts / f"{number:04d}"
            write_json(destination.with_suffix(".json"), dict(number=number, manifest_sha256=manifest_hash,
                       parent_checkpoint=str(parent_path.resolve()), parent_sha256=sha(parent_path)))
            train(config, destination, resume=parent_path, episode_log="chunked-gzip",
                  checkpoint_commits=True, diagnostics_every=50)
            parent_path, saved, _ = accepted_branch(entry, manifest_hash, m["source_sha256"], boundary=boundary, provenance=provenance)
        if saved["total_steps"] < config.total_steps:
            raise ValueError("Recovery stopped before its additional-step budget")
        evaluate_entry(m, entry, parent_path)
        if sha(boundary) != admission["boundary_checkpoint_sha256"]:
            raise ValueError("Boundary changed during recovery")
    return entry


def summarize(manifest):
    m = load_manifest(manifest)
    results, missing = [], []
    for entry in m["entries"]:
        path = Path(entry["output"]) / "nominal.json"
        if not path.exists():
            missing.append(entry["index"])
            continue
        boundary, _ = boundary_record(m, m["parents"][entry["index"] // 3])
        checkpoint, saved, _ = accepted_branch(entry, sha(manifest), m["source_sha256"], boundary=boundary,
            provenance=branch_provenance(m, entry, boundary, sha(manifest)))
        report = read_json(path)
        receipt = read_json(path.with_suffix(".identity.json"))
        if receipt != dict(report_sha256=sha(path), runtime=m["parents"][entry["index"] // 3]["runtime"], manifest_sha256=sha(manifest)):
            raise ValueError("Evaluation receipt/runtime identity mismatch")
        if (saved["total_steps"] < entry["config"]["total_steps"] or report["checkpoint_sha256"] != sha(checkpoint)
                or report["scenarios"] != panel() or report.get("scenarios_sha256") != m["panel_sha256"]
                or normalized(Config(**report["config"])) != entry["config"]
                or report["intervention"] != "none" or report["sham"]
                or report["evaluator"]["source"]["sha256"] != m["source_sha256"]):
            raise ValueError("Substituted/incomplete recovery evaluation")
        episodes = report["per_episode"]
        if len(episodes) != PANEL_EPISODES or len({e["scenario_id"] for e in episodes}) != PANEL_EPISODES:
            raise ValueError("Incomplete/duplicate evaluation episodes")
        for e, scenario in zip(episodes, panel()):
            if any(e[k] != v for k, v in scenario.items()) or e["event_exposed"]:
                raise ValueError("Evaluation episode/panel mismatch")
        score = Fraction(sum(e["steps"] if e["success"] else 80 for e in episodes), len(episodes))
        results.append(dict(index=entry["index"], seed=entry["seed"], variant=entry["variant"],
                            success_rate=sum(e["success"] for e in episodes) / len(episodes),
                            capped_completion=float(score), exact_score=[score.numerator, score.denominator],
                            team_return=sum(e["team_return"] for e in episodes) / len(episodes),
                            checkpoint_sha256=sha(checkpoint), report_sha256=sha(path),
                            actual_additional_steps=saved["total_steps"] - ONSETS[entry["seed"]][1]))
    return dict(study=STUDY, complete=not missing, missing=missing, results=results,
                manifest_sha256=sha(manifest), panel_sha256=m["panel_sha256"],
                interpretation="Descriptive two-seed recovery results; no automatic configuration selection")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--archive-root", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--additional-steps", type=int, default=ADDITIONAL_STEPS)
    for name, limit in (("reconstruct", 2), ("run", 6)):
        command = commands.add_parser(name)
        command.add_argument("--manifest", required=True)
        command.add_argument("--index", type=int, choices=range(limit), required=True)
        if name == "run":
            command.add_argument("--dry-run", action="store_true")
            command.add_argument("--resume", action="store_true")
    summary = commands.add_parser("summarize")
    summary.add_argument("--manifest", required=True)
    args = vars(parser.parse_args(argv))
    name = args.pop("command")
    result = {"prepare": prepare, "reconstruct": reconstruct, "run": launch, "summarize": summarize}[name](**args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
