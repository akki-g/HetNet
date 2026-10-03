#!/usr/bin/env python3
"""Refresh current reconstruction provenance without rewriting the dated audit."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "publication_reconstruction"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def refresh(output):
    output = Path(output).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Evidence output already exists: {output}")
    origins_path = PACKAGE / "ORIGINS.json"
    old_bytes = origins_path.read_bytes()
    origins = json.loads(old_bytes)
    old_files = origins["files"]
    paths = [p for p in sorted((PACKAGE / "runtime").rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    files, patches = {}, []
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for path in paths:
        name = path.relative_to(PACKAGE / "runtime").as_posix()
        payload = path.read_bytes()
        row = dict(old_files.get(name, {"source_repository": "local addition", "commit": None,
                                      "path": None, "original_sha256": None}))
        row.update(current_sha256=sha(payload), modified=sha(payload) != row["original_sha256"])
        files[name] = row
        repo_path = path.relative_to(ROOT).as_posix()
        original = subprocess.run(["git", "show", f"{head}:{repo_path}"], cwd=ROOT,
                                  capture_output=True).stdout
        patches.extend(difflib.unified_diff(original.decode().splitlines(True), payload.decode().splitlines(True),
                                           fromfile=f"{head}/{repo_path}", tofile=f"working/{repo_path}"))
    removed = sorted(set(old_files) - set(files))
    if removed:
        raise ValueError(f"Refusing silent removal of runtime provenance: {removed}")
    origins.update(files=files, current_audit={"base_commit": head, "evidence": str(output.relative_to(ROOT))
                                             if output.is_relative_to(ROOT) else str(output),
                                             "previous_origins_sha256": sha(old_bytes)})
    new_bytes = (json.dumps(origins, indent=2, sort_keys=True) + "\n").encode()
    output.mkdir(parents=True, exist_ok=False)
    (output / "previous_ORIGINS.json").write_bytes(old_bytes)
    (output / "runtime_changes.patch").write_text("".join(patches))
    (output / "ORIGINS.json").write_bytes(new_bytes)
    record = {"base_commit": head, "previous_origins_sha256": sha(old_bytes), "current_origins_sha256": sha(new_bytes),
              "historical_audit_preserved": "analysis/publication_reconstruction_2026-09-30/",
              "files": files, "patch_sha256": sha((output / "runtime_changes.patch").read_bytes())}
    (output / "audit.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary = origins_path.with_suffix(".json.tmp")
    with temporary.open("xb") as stream:
        stream.write(new_bytes)
    temporary.replace(origins_path)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="fresh evidence folder; required for refresh")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        import sys
        sys.path.insert(0, str(ROOT))
        from publication_reconstruction.__main__ import validate_source_origins
        payloads, _ = validate_source_origins()
        print(json.dumps({"valid": True, "runtime_files": len(payloads)}))
    elif args.output is None:
        parser.error("--output is required unless --check is used")
    else:
        result = refresh(args.output)
        print(json.dumps({"output": str(args.output), "current_origins_sha256": result["current_origins_sha256"]}))


if __name__ == "__main__":
    main()
