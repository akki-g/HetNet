"""Streaming, independently verified episode evidence and immutable run attempts.

The chunk index is an accelerator, never an admission certificate. ``audit_run``
reads the raw records and checks its statistics. Cached receipts explicitly pin
immutable evidence; callers request another audit to check those bytes again.
"""
from contextlib import contextmanager
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

VERSION = 1
INDEX = "episodes.index.json"


def sha(path, limit=None):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        remaining = limit
        while remaining is None or remaining:
            data = stream.read(1024 * 1024 if remaining is None else min(remaining, 1024 * 1024))
            if not data:
                if remaining:
                    raise ValueError(f"Truncated evidence: {path}")
                break
            h.update(data)
            if remaining is not None:
                remaining -= len(data)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def empty_stats():
    return dict(episodes=0, steps=0, successes=0, team_return=0., mean_agent_return=0.,
                capped_completion_sum=0, event_exposed=0)


def add_episode(stats, row, horizon):
    stats["episodes"] += 1
    stats["steps"] += row["steps"]
    stats["successes"] += bool(row["success"])
    stats["team_return"] += row["team_return"]
    stats["mean_agent_return"] += (row["mean_agent_return"] if row.get("mean_agent_return") is not None
                                    else row["team_return"] / row["num_agents"])
    stats["capped_completion_sum"] += row["steps"] if row["success"] else horizon
    stats["event_exposed"] += bool(row.get("event_exposed"))


class ChunkedEpisodes:
    """Append complete updates; publish gzip chunks only at explicit seal points."""
    def __init__(self, run, horizon, interval=100):
        self.run, self.horizon, self.interval = Path(run), horizon, interval
        self.directory = self.run / "episodes"
        self.directory.mkdir()
        self.index = {"version": VERSION, "compression": "gzip", "compresslevel": 1, "chunks": []}
        self.stream = self.raw = None
        self.updates = []
        atomic_json(self.run / INDEX, self.index)

    def append(self, rows, update):
        if self.stream is None:
            self.temporary = self.directory / "pending.jsonl.gz.tmp"
            self.raw = self.temporary.open("wb")
            self.stream = gzip.GzipFile(filename="", fileobj=self.raw, mode="wb", compresslevel=1, mtime=0)
        stats = empty_stats()
        for row in rows:
            self.stream.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())
            add_episode(stats, row, self.horizon)
        self.updates.append(dict(update=update, **stats))
        if update % self.interval == 0:
            self.seal()

    def seal(self):
        if self.stream is None:
            return
        self.stream.close()
        self.raw.flush()
        os.fsync(self.raw.fileno())
        self.raw.close()
        first, last = self.updates[0]["update"], self.updates[-1]["update"]
        path = self.directory / f"updates{first:08d}-{last:08d}.jsonl.gz"
        if path.exists():
            raise FileExistsError(path)
        self.temporary.replace(path)
        self.index["chunks"].append({"path": str(path.relative_to(self.run)), "sha256": sha(path),
                                      "bytes": path.stat().st_size, "first_update": first, "last_update": last,
                                      "episodes": sum(x["episodes"] for x in self.updates),
                                      "steps": sum(x["steps"] for x in self.updates), "updates": self.updates})
        atomic_json(self.run / INDEX, self.index)
        self.stream = self.raw = None
        self.updates = []

    def close(self):
        # An exception deliberately leaves an unpublished tail. Recovery uses only
        # committed checkpoints and never admits this tail as completed evidence.
        if self.stream is not None:
            self.stream.close()
            self.raw.close()
            self.stream = self.raw = None


def _safe_path(run, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Evidence path escapes its run")
    return Path(run) / relative


def ledger_files(run):
    run = Path(run)
    if (run / INDEX).exists():
        index = read_json(run / INDEX)
        if index.get("version") != VERSION:
            raise ValueError("Unsupported episode index")
        return [run / INDEX] + [_safe_path(run, c["path"]) for c in index["chunks"]]
    return [run / "episodes.jsonl"]


def _json_rows(path, through_update=None, expected_rows=None, evidence=None, compressed=False, prefixes=None):
    digest, size, count = hashlib.sha256(), 0, 0
    opener = gzip.open if compressed else open
    with opener(path, "rb") as stream:
        while expected_rows is None or count < expected_rows:
            line = stream.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError(f"Invalid committed ledger record: {path}") from error
            if through_update is not None and row.get("update", 0) > through_update:
                break
            digest.update(line)
            size += len(line)
            count += 1
            if prefixes is not None:
                prefixes[str(row["update"])] = {"bytes": size, "sha256": digest.hexdigest()}
            yield row
    if expected_rows is not None and count != expected_rows:
        raise ValueError(f"Missing committed ledger records: {path}")
    if evidence is not None:
        evidence.append({"path": str(Path(path).resolve()), "bytes": size,
                         "sha256": digest.hexdigest(), "decoded": compressed})


def iter_episodes(run, through_update=None):
    """Read a single attempt's published records, verifying chunk hashes."""
    run = Path(run)
    if not (run / INDEX).exists():
        yield from _json_rows(run / "episodes.jsonl", through_update)
        return
    index = read_json(run / INDEX)
    if index.get("version") != VERSION:
        raise ValueError("Unsupported episode index")
    previous = None
    horizon = read_json(run / "config.json")["max_steps"]
    for chunk in index["chunks"]:
        first, last = chunk["first_update"], chunk["last_update"]
        if previous is not None and first != previous + 1:
            raise ValueError("Gapped or duplicate chunk update ranges")
        previous = last
        if through_update is not None and first > through_update:
            break
        path = _safe_path(run, chunk["path"])
        if sha(path) != chunk["sha256"] or path.stat().st_size != chunk["bytes"]:
            raise ValueError(f"Corrupt episode chunk: {path}")
        statistics = {}
        first_seen = last_seen = None
        for row in _json_rows(path, compressed=True):
            update = row.get("update")
            if type(update) is not int or not first <= update <= last:
                raise ValueError("Episode outside indexed chunk range")
            if last_seen is not None and update not in (last_seen, last_seen + 1):
                raise ValueError("Gapped or unordered chunk episodes")
            if first_seen is None:
                first_seen = update
            last_seen = update
            add_episode(statistics.setdefault(update, empty_stats()), row, horizon)
            if through_update is None or update <= through_update:
                yield row
        actual = [dict(update=u, **statistics[u]) for u in sorted(statistics)]
        if (first_seen != first or last_seen != last or actual != chunk["updates"]
                or sum(x["episodes"] for x in actual) != chunk["episodes"]
                or sum(x["steps"] for x in actual) != chunk["steps"]):
            raise ValueError("Forged episode index statistics or range")


def logical_segments(run):
    """Accepted ancestry of the latest immutable attempt; abandoned tails stay on disk."""
    run = Path(run).resolve()
    candidates = sorted((run / "attempts").glob("[0-9][0-9][0-9][0-9]/attempt.json"))
    latest = candidates[-1].parent if candidates else run
    result, seen, limit = [], set(), None
    current = latest
    while True:
        if current in seen or not current.is_relative_to(run):
            raise ValueError("Invalid attempt ancestry")
        seen.add(current)
        metadata = read_json(current / "attempt.json") if (current / "attempt.json").exists() else None
        if metadata is not None:
            from softrole.config import Config
            original = read_json(run.parent / (run.name + ".programme.json"))
            config = json.loads(json.dumps(Config(**read_json(run / "config.json")).to_dict()))
            config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            if (metadata.get("version") != VERSION or metadata.get("logical_run") != str(run)
                    or metadata.get("manifest_sha256") != original["manifest_sha256"]
                    or metadata.get("source_sha256") != original["source_sha256"]
                    or metadata.get("config_sha256") != config_id
                    or metadata.get("attempt") != int(current.name)):
                raise ValueError("Attempt manifest/source/configuration identity mismatch")
        result.append({"run": str(current), "through_update": limit, "attempt": metadata})
        if metadata is None:
            break
        parent = metadata.get("parent_checkpoint")
        if parent is None:
            if any(metadata.get(k) != 0 for k in ("accepted_updates", "accepted_total_steps", "accepted_total_episodes")):
                raise ValueError("Restart attempt has nonzero parent counts")
            break
        path = Path(parent)
        if not path.exists():
            raise ValueError("Orphan attempt parent checkpoint")
        if sha(path) != metadata["parent_checkpoint_sha256"]:
            raise ValueError("Attempt parent checkpoint changed")
        import torch
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if any(metadata.get("accepted_" + name) != saved[name] for name in ("updates", "total_steps", "total_episodes")):
            raise ValueError("Attempt parent counts mismatch")
        current, limit = path.parent.parent.resolve(), metadata["accepted_updates"]
    return list(reversed(result))


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid {label}")
    return value


def _validate_episode(row, config):
    for name in ("update", "steps", "num_agents"):
        _integer(row.get(name), f"episode {name}", 1)
    _integer(row.get("collector"), "collector")
    if row["collector"] >= config.nprocesses or row["steps"] > config.max_steps:
        raise ValueError("Invalid collector or episode horizon")
    if not isinstance(row.get("scenario_id"), str) or not row["scenario_id"]:
        raise ValueError("Missing episode scenario identity")
    composition = (row.get("num_p"), row.get("num_a"))
    if (any(type(x) is not int or x < 0 for x in composition)
            or composition not in config.training_compositions or sum(composition) != row["num_agents"]
            or row.get("composition", list(composition)) != list(composition)):
        raise ValueError("Invalid episode roster")
    if type(row.get("success")) is not bool:
        raise ValueError("Invalid episode success flag")
    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(v) for v in value)
        return not isinstance(value, (int, float)) or isinstance(value, bool) or math.isfinite(value)
    if not finite(row):
        raise ValueError("Nonfinite episode metric")
    if not isinstance(row.get("team_return"), (int, float)):
        raise ValueError("Missing episode team return")


def audit_run(run, through_update=None):
    """Audit raw accepted updates/episodes, returning compact per-update evidence.

    ``through_update`` pins a checkpoint boundary and ignores all later bytes,
    including a torn final append. No cached index statistic is trusted.
    """
    from softrole.config import Config
    root = Path(run).resolve()
    config = Config(**read_json(root / "config.json"))
    normalized = json.loads(json.dumps(config.to_dict()))
    rows, inputs, previous, total_steps, total_episodes = [], [], 0, 0, 0
    prefix_records = []
    for segment in logical_segments(root):
        path = Path(segment["run"])
        limit = segment["through_update"]
        if through_update is not None:
            limit = through_update if limit is None else min(limit, through_update)
        if limit is not None and limit <= previous:
            continue
        if Config(**read_json(path / "config.json")).to_dict() != config.to_dict():
            raise ValueError("Attempt configuration differs from logical run")
        updates = []
        if not (path / "updates.jsonl").exists():
            if limit is not None:
                raise ValueError("Missing committed update ledger")
            continue
        update_prefixes = {}
        for row in _json_rows(path / "updates.jsonl", limit,
                              expected_rows=(limit - previous) if limit is not None else None, evidence=inputs, prefixes=update_prefixes):
            update = _integer(row.get("update"), "update", 1)
            if update != previous + len(updates) + 1:
                raise ValueError("Gapped or duplicate updates")
            for name in ("episodes", "steps", "total_steps"):
                _integer(row.get(name), f"update {name}", 1)
            if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
                raise ValueError("Nonfinite update metric")
            updates.append(row)
        if limit is not None and (not updates or updates[-1]["update"] != limit):
            raise ValueError("Missing committed updates")
        if not updates:
            continue
        prefix_records.append({"path": str((path / "updates.jsonl").resolve()), "updates": update_prefixes})
        expected = sum(row["episodes"] for row in updates)
        episode_prefixes = {}
        indexed = (path / INDEX).exists()
        episodes = (iter_episodes(path, updates[-1]["update"] if limit is not None else None) if indexed else
                    _json_rows(path / "episodes.jsonl", updates[-1]["update"] if limit is not None else None,
                               expected_rows=expected if limit is not None else None, evidence=inputs, prefixes=episode_prefixes))
        iterator = iter(episodes)
        chunk_stats = {}
        for update in updates:
            stats, seen, collectors = empty_stats(), set(), set()
            for _ in range(update["episodes"]):
                try:
                    episode = next(iterator)
                except StopIteration as error:
                    raise ValueError("Missing committed episode records") from error
                _validate_episode(episode, config)
                if episode["update"] != update["update"]:
                    raise ValueError("Episode/update counts disagree")
                identity = (episode["collector"], episode["scenario_id"])
                if identity in seen:
                    raise ValueError("Duplicate episode identity")
                seen.add(identity)
                collectors.add(episode["collector"])
                add_episode(stats, episode, config.max_steps)
            if collectors != set(range(config.nprocesses)):
                raise ValueError("Missing collector episodes")
            if stats["steps"] != update["steps"] or total_steps + stats["steps"] != update["total_steps"]:
                raise ValueError("Episode/update step counts disagree")
            total_steps += stats["steps"]
            total_episodes += stats["episodes"]
            if "total_episodes" in update and update["total_episodes"] != total_episodes:
                raise ValueError("Update cumulative episode count disagrees")
            row = dict(update=update["update"], total_steps=total_steps, total_episodes=total_episodes, **stats)
            rows.append(row)
            chunk_stats[row["update"]] = dict(update=row["update"], **stats)
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise ValueError("Unexpected episode records in committed updates")
        if not indexed:
            prefix_records.append({"path": str((path / "episodes.jsonl").resolve()), "updates": episode_prefixes})
        if indexed:
            index = read_json(path / INDEX)
            for chunk in index["chunks"]:
                if chunk["first_update"] > updates[-1]["update"]:
                    break
                actual = [chunk_stats[u] for u in range(chunk["first_update"], min(chunk["last_update"], updates[-1]["update"]) + 1)]
                claimed = [x for x in chunk["updates"] if x["update"] <= updates[-1]["update"]]
                if actual != claimed:
                    raise ValueError("Forged episode index statistics")
                if chunk["last_update"] <= updates[-1]["update"] and (
                    sum(x["episodes"] for x in actual) != chunk["episodes"] or sum(x["steps"] for x in actual) != chunk["steps"]):
                    raise ValueError("Forged episode chunk totals")
                inputs.append({"path": str((path / chunk["path"]).resolve()), "sha256": chunk["sha256"], "bytes": chunk["bytes"], "decoded": False})
        previous = updates[-1]["update"]
        if through_update is not None and previous >= through_update:
            break
    if not rows:
        raise ValueError(f"{root} has no recorded updates")
    if through_update is not None and previous != through_update:
        raise ValueError("Missing committed updates")
    payload = {"version": VERSION, "kind": "softrole-ledger-audit", "run": str(root), "config": normalized,
               "logical_run_sha256": canonical({"config": normalized, "run": str(root)}), "inputs": inputs, "prefixes": prefix_records,
               "updates": rows, "covered_updates": [rows[0]["update"], previous],
               "total_steps": total_steps, "total_episodes": total_episodes}
    payload["audit_sha256"] = canonical(payload)
    return payload


def validate_receipt(receipt, reaudit=False):
    receipt = read_json(receipt) if isinstance(receipt, (str, Path)) else receipt
    payload = {k: v for k, v in receipt.items() if k != "audit_sha256"}
    if receipt.get("version") != VERSION or receipt.get("kind") != "softrole-ledger-audit" or canonical(payload) != receipt.get("audit_sha256"):
        raise ValueError("Invalid ledger audit receipt")
    if reaudit:
        fresh = audit_run(receipt["run"], receipt["covered_updates"][1])
        if fresh["updates"] != receipt["updates"] or fresh["inputs"] != receipt["inputs"]:
            raise ValueError("Evidence changed since independent audit")
    return receipt


def stats_from_audit(receipt, windows):
    receipt = validate_receipt(receipt)
    result = {}
    for lo, hi in windows:
        if type(lo) is not int or type(hi) is not int or lo < 0 or hi <= lo:
            raise ValueError("Windows must have start < end")
        stats = empty_stats()
        for row in receipt["updates"]:
            if lo < row["total_steps"] <= hi:
                for key in stats:
                    stats[key] += row[key]
        n = stats["episodes"]
        result[f"{lo}-{hi}"] = dict(start_exclusive=lo, end_inclusive=hi, episodes=n, steps=stats["steps"],
            complete=receipt["total_steps"] >= hi, event_exposed_episodes=stats["event_exposed"],
            capped_completion_sum=stats["capped_completion_sum"],
            success_rate=stats["successes"] / n if n else None,
            capped_completion=stats["capped_completion_sum"] / n if n else None,
            team_return=stats["team_return"] / n if n else None,
            mean_agent_return=stats["mean_agent_return"] / n if n else None)
    return result, receipt["total_steps"]


def checkpoint_commit(path, run, payload):
    """Publish last: checkpoint, signature, and the exact required ledger prefix."""
    run, path = Path(run), Path(path)
    files = []
    for candidate in [path, Path(str(path) + ".signature.json"), run / "updates.jsonl"]:
        files.append({"path": str(candidate.relative_to(run)), "bytes": candidate.stat().st_size, "sha256": sha(candidate)})
    if (run / INDEX).exists():
        for chunk in read_json(run / INDEX)["chunks"]:
            files.append({"path": chunk["path"], "bytes": chunk["bytes"], "sha256": chunk["sha256"]})
    elif (run / "episodes.jsonl").exists():
        candidate = run / "episodes.jsonl"
        files.append({"path": "episodes.jsonl", "bytes": candidate.stat().st_size, "sha256": sha(candidate)})
    record = {"version": VERSION, "checkpoint": path.name, "files": files,
              **{key: payload[key] for key in ("updates", "total_steps", "total_episodes", "epoch", "source_sha256")}}
    atomic_json(str(path) + ".complete.json", record)


def committed_checkpoints(run, verify_ledgers=True, stop_at_steps=None):
    """Validate completion records and return published checkpoints in one attempt."""
    run = Path(run)
    result = []
    required = read_json(run / "run.json").get("checkpoint_commits", False) if (run / "run.json").exists() else False
    for marker in (run / "checkpoints").glob("epoch*.pt.complete.json"):
        if not Path(str(marker).removesuffix(".complete.json")).exists():
            raise ValueError("Orphan checkpoint completion record")
    for path in sorted((run / "checkpoints").glob("epoch*.pt")):
        marker = Path(str(path) + ".complete.json")
        if required and not marker.exists():
            continue
        if marker.exists():
            record = read_json(marker)
            if record.get("version") != VERSION or record.get("checkpoint") != path.name:
                raise ValueError("Invalid checkpoint completion record")
            for item in record["files"]:
                if not verify_ledgers and item["path"] not in (path.name, str(path.relative_to(run)), str(path.relative_to(run)) + ".signature.json"):
                    continue
                candidate = _safe_path(run, item["path"])
                if (candidate.suffix in (".pt", ".gz") and candidate.stat().st_size != item["bytes"]) or sha(candidate, item["bytes"]) != item["sha256"]:
                    raise ValueError(f"Corrupt committed checkpoint evidence: {candidate}")
        else:
            if not Path(str(path) + ".signature.json").exists():
                continue
        import torch
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            raise ValueError(f"Invalid checkpoint: {path}") from error
        from softrole import CHECKPOINT_VERSION
        from softrole.config import Config
        from softrole.model import SoftRoleNet
        from hetnet_ext.signatures import model_signature
        if payload.get("format_version") != CHECKPOINT_VERSION:
            raise ValueError("Unsupported checkpoint format")
        with torch.random.fork_rng():
            model = SoftRoleNet(**Config(**payload["config"]).model_kwargs()).double()
        model.load_state_dict(payload["model_state"], strict=True)
        if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
            raise ValueError("Nonfinite checkpoint model state")
        if model_signature(model) != payload["signature"]:
            raise ValueError("Checkpoint model-state signature mismatch")
        def finite_tensors(value):
            if isinstance(value, torch.Tensor):
                return bool(torch.isfinite(value).all())
            if isinstance(value, dict):
                return all(finite_tensors(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return all(finite_tensors(v) for v in value)
            return True
        if not finite_tensors(payload["optimizer_state"]):
            raise ValueError("Nonfinite checkpoint optimizer state")
        if read_json(str(path) + ".signature.json") != payload["signature"]:
            raise ValueError("Checkpoint signature sidecar mismatch")
        if marker.exists() and any(record[k] != payload[k] for k in ("updates", "total_steps", "total_episodes", "epoch", "source_sha256")):
            raise ValueError("Checkpoint completion counts differ")
        result.append({"checkpoint": str(path.resolve()), "sha256": sha(path),
                       **{k: payload[k] for k in ("epoch", "updates", "total_steps", "total_episodes", "source_sha256")}})
        if stop_at_steps is not None and payload["total_steps"] >= stop_at_steps:
            break
    return result
