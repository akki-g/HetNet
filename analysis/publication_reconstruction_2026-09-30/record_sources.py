"""Record exact origins and reviewable diffs of the standalone reconstruction."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNTIME = ROOT / "publication_reconstruction/runtime"
MIRROR = ROOT / "analysis/upstream_history_2026-09-30/history.git"
UPSTREAM = "d57da0717d5564027df7e8ba75614feca8006960"
SCAFFOLD = "0ee9ceb38b6133866e9c1db40bed2c6caa3b842d"


def main():
    rows = {}
    diffs = []
    for path in sorted(RUNTIME.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        name = path.relative_to(RUNTIME).as_posix()
        upstream = name.startswith("envs/") or name == "WildFire_Simulate_Original.py"
        commit = UPSTREAM if upstream else SCAFFOLD
        command = ["git", f"--git-dir={MIRROR}"] if upstream else ["git", "-C", str(ROOT)]
        original = subprocess.check_output([*command, "show", f"{commit}:{name}"])
        current = path.read_bytes()
        rows[name] = {"source_repository": "CORE-Robotics-Lab/HetNet" if upstream else "local reproduction",
                      "commit": commit, "path": name, "original_sha256": hashlib.sha256(original).hexdigest(),
                      "current_sha256": hashlib.sha256(current).hexdigest(), "modified": original != current}
        if original != current:
            diffs.extend(difflib.unified_diff(original.decode().splitlines(keepends=True),
                         current.decode().splitlines(keepends=True), fromfile=f"{commit}/{name}",
                         tofile=f"publication_reconstruction/runtime/{name}"))
    output = {"schema_version": 1, "scope": "Exact file origins; not a verified publication training commit",
              "files": rows}
    (ROOT / "publication_reconstruction/ORIGINS.json").write_text(json.dumps(output, indent=2) + "\n")
    (HERE / "runtime_changes.patch").write_text("".join(diffs))
    print(json.dumps({"runtime_files": len(rows), "changed_files": [name for name, row in rows.items() if row["modified"]]}))


if __name__ == "__main__":
    main()
