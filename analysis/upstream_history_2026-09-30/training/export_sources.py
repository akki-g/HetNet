"""Export exact upstream blobs and record their identities without a checkout."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
MIRROR = ROOT.parent / "history.git"
COMMITS = ["3ca462b", "d57da07", "6b09d36", "f54ca5a"]
PATHS = ["main.py", "trainer.py", "multi_processing.py", "action_utils.py", "README.md", "utils.py",
         "hetgat/uavnet.py", "hetgat/policy.py", "hetgat/policy_entropy.py", "hetgat/utils.py",
         "hetgat/graph/fastreal.py", "hetgat/graph/fastbinary.py", "hetgat/graph/hetgatreal.py",
         "hetgat/graph/hetgata2c.py"]
def git(*args):
    return subprocess.check_output(["git", "--git-dir="+str(MIRROR), *args])

records = []
for short in COMMITS:
    commit = git("rev-parse", short).decode().strip()
    for path in PATHS:
        raw = git("show", commit+":"+path)
        target = ROOT / "snapshots" / short / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert target.read_bytes() == raw, target
        else:
            target.write_bytes(raw)
        records.append(dict(commit=commit, path=path,
                            local_path=str(target.relative_to(ROOT)),
                            blob=git("rev-parse", commit+":"+path).decode().strip(),
                            sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
(ROOT / "source_manifest.json").write_text(json.dumps(records, indent=2)+"\n")
print(json.dumps(dict(commits=len(COMMITS), exported_blobs=len(records))))
