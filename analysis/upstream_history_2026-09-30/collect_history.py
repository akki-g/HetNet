"""Record public upstream history and exact relevant source snapshots.

Requires a full bare mirror at adjacent history.git. Does not alter the working
repository or execute historical training. Run in a fresh audit directory.
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

HERE = Path(__file__).resolve().parent
MIRROR = HERE / "history.git"
BASE = "https://api.github.com/repos/CORE-Robotics-Lab/HetNet"


def git(*args):
    return subprocess.check_output(["git", f"--git-dir={MIRROR}", *args])


def save_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(content, stream, indent=2)
        stream.write("\n")


def fetch(name, suffix):
    url = BASE + suffix
    request = urllib.request.Request(url, headers={"User-Agent": "HetNet-reproducibility-audit"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
        headers = dict(response.headers)
    path = HERE / "public_api" / (name + ".json")
    path.parent.mkdir(exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
    return json.loads(payload), {"url": url, "response_date": headers.get("Date"),
                                 "sha256": hashlib.sha256(payload).hexdigest(),
                                 "pagination": headers.get("Link")}


def main():
    fetches = {}
    for name, suffix in [("repository", ""), ("branches", "/branches?per_page=100"),
                         ("tags", "/tags?per_page=100"), ("releases", "/releases?per_page=100"),
                         ("issues", "/issues?state=all&per_page=100"),
                         ("forks", "/forks?per_page=100&sort=oldest"),
                         ("main_commits", "/commits?sha=main&per_page=100")]:
        content, metadata = fetch(name, suffix)
        assert metadata["pagination"] is None, "Add pagination before claiming exhaustive metadata"
        fetches[name] = metadata
        if name == "issues":
            for issue in content:
                _, detail = fetch(f"issue_{issue['number']}_comments", f"/issues/{issue['number']}/comments?per_page=100")
                assert detail["pagination"] is None
                fetches[f"issue_{issue['number']}_comments"] = detail
    commits = []
    for sha in git("rev-list", "--all", "--reverse", "--topo-order").decode().splitlines():
        raw = git("show", "-s", "--format=%H%x00%P%x00%aI%x00%cI%x00%an%x00%s", sha).decode().strip().split("\0")
        full, parents, author_date, committer_date, author, subject = raw
        commits.append({"sha": full, "parents": parents.split(), "author_date": author_date,
                        "committer_date": committer_date, "author": author, "subject": subject,
                        "on_main": subprocess.run(["git", f"--git-dir={MIRROR}", "merge-base", "--is-ancestor", sha, "refs/heads/main"], capture_output=True).returncode == 0,
                        "changed_files": git("diff-tree", "--root", "--no-commit-id", "--name-status", "-r", sha).decode().splitlines()})
    refs = git("for-each-ref", "--format=%(refname) %(objectname)").decode().splitlines()
    chronology = {"scope": "All commits reachable from advertised upstream refs at fetch; deleted/private/unadvertised history is not recovered",
                  "refs": refs, "commits": commits, "api_fetches": fetches}
    save_json(HERE / "history.json", chronology)
    with (HERE / "timeline.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sha", "author_date", "committer_date", "author", "subject", "on_main"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in commits)
    tracked = ["hetgat/uavnet.py", "hetgat/policy.py", "hetgat/graph/fastreal.py", "hetgat/graph/fastbinary.py",
               "main.py", "trainer.py", "multi_processing.py", "action_utils.py", "eval_trainer.py", "README.md",
               "WildFire_Simulate_Original.py", "envs/ic3net_envs/predator_capture_env.py", "envs/ic3net_envs/fire_commander_env.py"]
    matrix = []
    for commit in commits:
        if not commit["on_main"]:
            continue
        names = set(git("ls-tree", "-r", "--name-only", commit["sha"]).decode().splitlines())
        blobs = {}
        for path in tracked:
            name = path if path in names else path.replace("envs/", "ic3net-envs/", 1)
            blobs[path] = {"historical_path": name, "blob": git("rev-parse", f"{commit['sha']}:{name}").decode().strip()} if name in names else None
        matrix.append({"commit": commit["sha"], "files": blobs})
    save_json(HERE / "file_history.json", matrix)
    snapshots = {}
    for short in ["3ca462b", "59d5983", "d57da07", "6b09d36", "f54ca5a", "bff9f7f"]:
        sha = git("rev-parse", short).decode().strip()
        names = git("ls-tree", "-r", "--name-only", sha).decode().splitlines()
        files = {}
        for name in names:
            if not (name.endswith(".py") or name == "README.md"):
                continue
            if name.startswith(".idea/"):
                continue
            data = git("show", f"{sha}:{name}")
            target = HERE / "snapshots" / short / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
            files[name] = hashlib.sha256(data).hexdigest()
        snapshots[short] = {"sha": sha, "files": files}
    save_json(HERE / "snapshot_manifest.json", snapshots)
    earliest_supplement = git("show", "8b28c6f:AAMAS_22___HetNet_Supplementary.pdf")
    with (HERE / "papers/supplement_jan2022.pdf").open("xb") as stream:
        stream.write(earliest_supplement)
    print(json.dumps({"commits": len(commits), "main_commits": sum(c["on_main"] for c in commits),
                      "refs": len(refs), "snapshots": len(snapshots),
                      "supplement_sha256": hashlib.sha256(earliest_supplement).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
