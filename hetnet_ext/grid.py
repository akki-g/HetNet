"""Declarative Phase A grid and scientific-code provenance (no Torch imports)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID = ROOT / "configs" / "run1_grid.json"


@dataclass(frozen=True)
class GridEntry:
    index: int
    composition: str
    nfriendly_P: int
    nfriendly_A: int
    seed: int
    priority: str

    def as_dict(self) -> dict:
        return asdict(self)


def _read(path: str | Path | None = None) -> dict:
    data = json.loads(Path(path or DEFAULT_GRID).read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Grid schema_version must be 1")
    return data


def _integer(value: object, label: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def load_grid(path: str | Path | None = None) -> list[GridEntry]:
    """Expand file order then seed order; reject unsupported singleton-P teams."""
    data = _read(path)
    entries = []
    seen_names: set[str] = set()
    seen_pairs: set[tuple[int, int, int]] = set()
    for comp in data["compositions"]:
        n_p = _integer(comp["nfriendly_P"], "nfriendly_P (singleton P excluded)", 2)
        n_a = _integer(comp["nfriendly_A"], "nfriendly_A", 1)
        name = comp["name"]
        if name != f"{n_p}P{n_a}A" or name in seen_names:
            raise ValueError("Composition names must be unique canonical P/A names")
        seen_names.add(name)
        if not comp["seeds"]:
            raise ValueError("Every composition must contain at least one seed")
        for seed in comp["seeds"]:
            seed = _integer(seed, "seed", 0)
            key = (n_p, n_a, seed)
            if key in seen_pairs:
                raise ValueError(f"Duplicate composition/seed: {key}")
            seen_pairs.add(key)
            entries.append(GridEntry(len(entries), name, n_p, n_a, seed, comp["priority"]))
    if not entries:
        raise ValueError("Grid must not be empty")
    return entries


def calibration_entries(path: str | Path | None = None) -> list[GridEntry]:
    data = _read(path)
    all_entries = load_grid(path)
    settings = data["calibration"]
    seed = _integer(settings["seed"], "calibration seed", 0)
    selected = []
    for name in settings["compositions"]:
        matches = [e for e in all_entries if e.composition == name and e.seed == seed]
        if len(matches) != 1 or any(e.composition == name for e in selected):
            raise ValueError(f"Invalid calibration selection: {name}, seed {seed}")
        match = matches[0]
        selected.append(GridEntry(len(selected), name, match.nfriendly_P,
                                  match.nfriendly_A, seed, match.priority))
    return selected


def entries_for_mode(mode: str, path: str | Path | None = None) -> list[GridEntry]:
    if mode == "train":
        return load_grid(path)
    if mode == "calibration":
        return calibration_entries(path)
    raise ValueError(f"Unknown mode: {mode}")


def run_directory(entry: GridEntry, mode: str, output_root: str | Path | None = None) -> Path:
    prefix = "run1_train" if mode == "train" else "calibration"
    return Path(output_root or ROOT / "runs" / prefix).resolve() / entry.composition / f"seed{entry.seed}"


def build_command(entry: GridEntry, run_dir: str | Path, *, mode: str = "train",
                  python: str = sys.executable, path: str | Path | None = None) -> list[str]:
    data = _read(path)
    if entry.nfriendly_P < 2:
        raise ValueError("Singleton-P compositions are excluded")
    recipe = dict(data["recipe"])
    if mode == "calibration":
        recipe["num_epochs"] = data["calibration"]["num_epochs"]
    elif mode != "train":
        raise ValueError(f"Unknown mode: {mode}")
    command = [python, "-u", str(ROOT / "main.py")]
    for name, value in recipe.items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{name}")
        else:
            command.extend((f"--{name}", str(value)))
    run_dir = Path(run_dir).resolve()
    command.extend(("--nfriendly_P", str(entry.nfriendly_P), "--nfriendly_A", str(entry.nfriendly_A),
                    "--seed", str(entry.seed), "--save_dir", str(run_dir / "checkpoints"),
                    "--experiment_name", f"{entry.composition}_s{entry.seed}",
                    "--metrics_file", str(run_dir / "metrics.jsonl")))
    return command


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_inventory(root: str | Path = ROOT) -> list[dict[str, str]]:
    """Hash tracked source plus new study code/config/tests; ignore documents/results.

    Including untracked scientific code closes a dirty-tree loophole without making
    a later notes/evidence commit invalidate an otherwise identical test gate.
    """
    root = Path(root).resolve()
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    paths = {Path(p) for p in tracked if p and (p.endswith((".py", ".sbatch", ".sh"))
             or p in {"pyproject.toml", "uv.lock"})}
    for prefix in ("hetnet_ext", "tests", "configs", "slurm"):
        base = root / prefix
        if base.exists():
            paths.update(p.relative_to(root) for p in base.rglob("*") if p.is_file()
                         and p.suffix in {".py", ".json", ".sh", ".sbatch"})
    for name in ("pyproject.toml", "uv.lock"):
        if (root / name).is_file():
            paths.add(Path(name))
    return [{"path": str(p), "sha256": file_sha256(root / p)} for p in sorted(paths)
            if (root / p).is_file()]


def code_sha256(root: str | Path = ROOT) -> str:
    raw = json.dumps(code_inventory(root), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", required=True,
                        help="Print commands only; creates no files or processes")
    parser.add_argument("--mode", choices=("train", "calibration"), default="train")
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--index", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    entries = entries_for_mode(args.mode, args.grid)
    if args.index is not None:
        if not 0 <= args.index < len(entries):
            parser.error(f"index must be between 0 and {len(entries) - 1}")
        entries = [entries[args.index]]
    for entry in entries:
        directory = run_directory(entry, args.mode, args.output_root)
        print(json.dumps({**entry.as_dict(), "mode": args.mode, "run_dir": str(directory),
                          "command": build_command(entry, directory, mode=args.mode, path=args.grid)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
