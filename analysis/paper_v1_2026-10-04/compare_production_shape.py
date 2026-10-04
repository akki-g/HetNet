"""Read-only numerical audit of the bounded production-shaped backend pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from compare_integration import compare


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from publication_reconstruction.artifacts import load_checkpoint
    from publication_reconstruction.runtime.hetnet_ext.signatures import tree_signature
    validation_path = args.runs / "validation.json"
    validation = json.loads(validation_path.read_text())
    assert validation["passed"] and validation["source_unchanged"]
    rows = validation["rows"]
    assert len(rows) == 8
    results = []
    for index in range(0, len(rows), 2):
        left, right = rows[index:index + 2]
        assert (left["backend"], right["backend"]) == ("dgl", "torch-v1")
        assert all(left[key] == right[key] for key in ("task", "variant", "initial_identity", "source_manifest_sha256"))
        states = []
        for row in (left, right):
            assert digest(row["checkpoint"]) == row["checkpoint_sha256"]
            states.append(load_checkpoint(Path(row["run"]), row["checkpoint"]))
        result = dict(task=left["task"], variant=left["variant"], collectors=4,
            model_max_absolute_error={}, optimizer_max_absolute_error={},
            final_rng_equal=tree_signature(states[0]["recovery"]["rng_states"]) == tree_signature(states[1]["recovery"]["rng_states"]),
            counts_equal=left["counts"] == right["counts"],
            episode_semantics_equal=left["episode_semantic_sha256"] == right["episode_semantic_sha256"],
            full_trajectory_identity_available=left["trajectory_identity_complete"] and right["trajectory_identity_complete"],
            checkpoint_sha256=[row["checkpoint_sha256"] for row in (left, right)])
        try:
            compare(states[0]["policy_net"], states[1]["policy_net"], result["model_max_absolute_error"])
            compare(states[0]["trainer"], states[1]["trainer"], result["optimizer_max_absolute_error"])
            result["model_optimizer_within_tolerance"] = True
        except AssertionError as error:
            result["model_optimizer_within_tolerance"] = False
            result["error"] = str(error)
        results.append(result)
    report = dict(schema_version=1, passed=all(row["model_optimizer_within_tolerance"] and row["counts_equal"] and row["final_rng_equal"] for row in results),
        pairs=results, tolerances={"torch.float32": {"atol": 1e-6, "rtol": 1e-5}, "other": {"atol": 1e-10, "rtol": 1e-8}},
        validation_sha256=digest(validation_path), script_sha256=digest(__file__),
        comparison_implementation_sha256=digest(Path(__file__).with_name("compare_integration.py")),
        limitation="One update per backend/workload. Equal episode summaries and RNG states do not prove action/observation trajectory equality; those full traces were not recorded. No throughput or learning-quality conclusion.")
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
