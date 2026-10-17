"""Transport previously admitted S1 evidence without changing canonical receipts.

stage --selection ORIGINAL.json --output FRESH_BUNDLE
verify --bundle BUNDLE --output FRESH_SELECTION.json [--inventory-sha256 DIGEST]

Verification hashes compressed bytes; it neither re-audits episode semantics nor
decodes the episode ledgers. Keep the printed inventory digest separately during
transfer. The regenerated selection relocates audit-file references only: original
run/input paths inside immutable receipts remain provenance, checked by relative
mapping to the staged trees. No programme manifest or cluster action is produced.
Bounded fixtures require --allow-bounded and remain explicitly bounded.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, payload):
    with Path(path).open("x") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def relative_path(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Expected a safe relative path: {name}")
    return path


def inventory(root):
    files = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Evidence trees must contain regular files, not symlinks: {path}")
        if path.is_file():
            files[str(path.relative_to(root))] = {"bytes": path.stat().st_size, "sha256": sha(path)}
        elif not path.is_dir():
            raise ValueError(f"Unsupported evidence file: {path}")
    return files


def staged_run(row):
    return Path("runs") / f"lr{row['lr']:g}" / f"seed{row['seed']}"


def check_assets(bundle, selection, files, allow_bounded):
    """Bind transported bytes to the original receipts and checkpoint/source identities."""
    from softrole.config import Config
    from softrole.ledger import validate_receipt
    from softrole.programme import S1_RULE, validate_selection

    if selection.get("rule") != S1_RULE or selection.get("metric") != "capped_completion":
        raise ValueError("Selection does not follow the fixed S1 rule")
    if selection.get("bounded_check") and not allow_bounded:
        raise ValueError("Bounded evidence requires --allow-bounded; it cannot admit research runs")
    relocated = copy.deepcopy(selection)
    audits = []
    for row in relocated.get("runs", []):
        base = staged_run(row)
        audit = bundle / base / "audit.json"
        if files[str(base / "audit.json")]["sha256"] != row["audit_sha256"]:
            raise ValueError("Copied audit bytes differ from the certified selection")
        receipt = validate_receipt(audit)
        if receipt["run"] != row["run"]:
            raise ValueError("Selection and receipt disagree about the original admitted run")
        for item in receipt["inputs"]:
            relative = relative_path(Path(item["path"]).relative_to(receipt["run"]))
            if item.get("decoded") or files[str(base / relative)] != {k: item[k] for k in ("bytes", "sha256")}:
                raise ValueError(f"Transported receipt input differs: {relative}")
        anchor = receipt["anchor"]
        checkpoint = relative_path(Path(anchor["checkpoint"]).relative_to(anchor["original_run"]))
        if files[str(base / checkpoint)]["sha256"] != anchor["sha256"]:
            raise ValueError("Copied checkpoint differs from the admitted anchor")
        if str(base / Path(str(checkpoint) + ".signature.json")) not in files:
            raise ValueError("Missing checkpoint signature sidecar")
        source = read_json(bundle / base / "source_manifest.json")
        identity = hashlib.sha256(json.dumps(source["files"], sort_keys=True).encode()).hexdigest()
        if identity != source["sha256"] or identity != anchor["source_sha256"]:
            raise ValueError("Source manifest differs from the admitted checkpoint identity")
        for name, digest in source["files"].items():
            if files[str(base / "source" / relative_path(name))]["sha256"] != digest:
                raise ValueError(f"Transported archived source differs: {name}")
        if (Config(**read_json(bundle / base / "config.json")).to_dict() != Config(**receipt["config"]).to_dict()
                or read_json(bundle / base / "run.json")["source_sha256"] != identity):
            raise ValueError("Transported configuration/run identity differs")
        row["audit"] = str(audit.resolve())
        audits.append(audit)
    # Production validation checks all six identities, anchors and window scores.
    validate_selection(relocated)
    return audits


def stage(selection, output, allow_bounded=False):
    from softrole.programme import validate_selection

    selection_path, output = Path(selection).resolve(), Path(output).resolve()
    canonical_bytes = selection_path.read_bytes()
    canonical = json.loads(canonical_bytes)
    validate_selection(canonical)
    if canonical.get("bounded_check") and not allow_bounded:
        raise ValueError("Bounded evidence requires --allow-bounded")
    # Snapshot source hashes before copying, so copied bytes cannot silently race
    # a changing input. A failed staging directory is retained without a receipt.
    expected = {"selection.canonical.json": {"bytes": len(canonical_bytes),
                                           "sha256": hashlib.sha256(canonical_bytes).hexdigest()}}
    sources = []
    for row in canonical["runs"]:
        source = Path(row["run"])
        if Path(row["audit"]).resolve() != (source / "audit.json").resolve():
            raise ValueError("Expected each canonical audit.json inside its admitted run")
        if output == source.resolve() or output.is_relative_to(source.resolve()):
            raise ValueError("The bundle must be outside its input evidence trees")
        base = staged_run(row)
        expected.update({str(base / relative): record for relative, record in inventory(source).items()})
        sources.append((source, base))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "selection.canonical.json").open("xb") as stream:
        stream.write(canonical_bytes)
    for source, base in sources:
        shutil.copytree(source, output / base)
    actual = inventory(output)
    if actual != expected:
        raise ValueError("Source/copy bytes differ; incomplete bundle has no inventory certificate")
    check_assets(output, canonical, actual, allow_bounded)
    record = {"version": 1, "scope": "transport identity only; prior scientific admission retained",
              "canonical_selection": "selection.canonical.json", "files": actual}
    write_json(output / "inventory.json", record)
    return {"bundle": str(output), "files": len(actual), "inventory_sha256": sha(output / "inventory.json"),
            "selected_lr": canonical["selected_lr"], "bounded_check": bool(canonical.get("bounded_check"))}


def verify(bundle, output, inventory_sha256=None, allow_bounded=False):
    from softrole.programme import select_learning_rate, validate_selection

    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if output.exists() or output.is_symlink() or output.is_relative_to(bundle):
        raise ValueError("The relocation selection must be a fresh file outside the immutable bundle")
    digest = sha(bundle / "inventory.json")
    if inventory_sha256 is not None and digest != inventory_sha256:
        raise ValueError("Transport inventory differs from the independently pinned digest")
    record = read_json(bundle / "inventory.json")
    if record.get("version") != 1 or record.get("canonical_selection") != "selection.canonical.json":
        raise ValueError("Unsupported transport inventory")
    actual = inventory(bundle)
    actual.pop("inventory.json")
    if actual != record["files"]:
        raise ValueError("Transported file inventory differs (missing, changed or extra files)")
    canonical = read_json(bundle / "selection.canonical.json")
    audits = check_assets(bundle, canonical, actual, allow_bounded)
    output.parent.mkdir(parents=True, exist_ok=True)
    # The same production implementation as `select-lr --audits`; canonical
    # receipt contents and their hashes are intentionally never rewritten.
    with tempfile.TemporaryDirectory(prefix=".s1-relocation-", dir=output.parent) as scratch:
        candidate = Path(scratch) / "selection.json"
        relocated = select_learning_rate(None, candidate, window=canonical["window"],
                                         bounded=bool(canonical.get("bounded_check")), audits=audits)
        validate_selection(relocated)
        comparable = copy.deepcopy(relocated)
        for row, original in zip(comparable["runs"], canonical["runs"]):
            row["audit"] = original["audit"]
        if comparable != canonical:
            raise ValueError("Regenerated selection differs beyond audit locations")
        with output.open("xb") as stream:
            stream.write(candidate.read_bytes())
    return {"selection": str(output), "selection_sha256": sha(output), "inventory_sha256": digest,
            "selected_lr": relocated["selected_lr"], "bounded_check": relocated["bounded_check"],
            "exact_scores_unchanged": True, "receipt_bytes_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    stage_parser = subcommands.add_parser("stage")
    stage_parser.add_argument("--selection", required=True)
    stage_parser.add_argument("--output", required=True)
    verify_parser = subcommands.add_parser("verify")
    verify_parser.add_argument("--bundle", required=True)
    verify_parser.add_argument("--output", required=True)
    verify_parser.add_argument("--inventory-sha256")
    for command in (stage_parser, verify_parser):
        command.add_argument("--allow-bounded", action="store_true")
    options = vars(parser.parse_args())
    command = options.pop("command")
    result = stage(**options) if command == "stage" else verify(**options)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
