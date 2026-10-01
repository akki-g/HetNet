"""Archive decisive diffs and locate candidate experiment artifacts in Git trees."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
MIRROR = HERE / "history.git"


def git(*args):
    return subprocess.check_output(["git", f"--git-dir={MIRROR}", *args])


def main():
    history = json.loads((HERE / "history.json").read_text())
    rows = []
    for commit in history["commits"]:
        files = git("ls-tree", "-r", "--name-only", commit["sha"]).decode().splitlines()
        candidates = [name for name in files if (
            Path(name).suffix.lower() in {".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".zip", ".csv", ".jsonl", ".lock"}
            or "initial_starts" in name.lower()
            or any(word in Path(name).name.lower() for word in ["requirements", "environment.yml", "train.log"]))]
        rows.append({"commit": commit["sha"], "author_date": commit["author_date"],
                     "on_main": commit["on_main"], "candidate_artifacts": candidates})
    url = "https://gatech.box.com/s/v737rbbwkfa4xop1n5qgmas6b6rwkgqn"
    request = urllib.request.Request(url, headers={"User-Agent": "HetNet-reproducibility-audit"})
    link = {"url": url, "checked_utc": datetime.now(timezone.utc).isoformat(),
            "introduced_on": "hetnet-ppo branch, e0bcaf3, June 14 2026",
            "scope": "HTTP access only; inaccessible contents are not characterized"}
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
            link.update(status=response.status, final_url=response.url,
                        response_sha256=hashlib.sha256(payload).hexdigest())
    except urllib.error.HTTPError as exc:
        link.update(status=exc.code, reason=str(exc.reason))
    except urllib.error.URLError as exc:
        link.update(error=str(exc.reason))
    content = {"scope": "Filename search across every reachable tree, not inspection of unavailable external contents",
               "trees": rows, "external_artifact_link": link}
    (HERE / "artifact_search.json").write_text(json.dumps(content, indent=2) + "\n")
    diff_dir = HERE / "diffs"
    diff_dir.mkdir(exist_ok=True)
    diffs = {
        "model_october2022.patch": ["6b09d36^", "6b09d36", "--", "hetgat/uavnet.py"],
        "fc_may2026.patch": ["bff9f7f^", "bff9f7f", "--", "WildFire_Simulate_Original.py", "envs/ic3net_envs/fire_commander_env.py"],
        "recipes_june2022.patch": ["394b57e^", "4179f48", "--", "README.md"],
        "recipes_october2022.patch": ["f54ca5a^", "f54ca5a", "--", "README.md"],
    }
    for name, args in diffs.items():
        (diff_dir / name).write_bytes(git("diff", *args))
    print(json.dumps({"trees": len(rows), "diffs": len(diffs), "external_link": link}))


if __name__ == "__main__":
    main()
