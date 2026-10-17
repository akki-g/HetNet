"""Cross-arm contrasts and difference-in-differences on hand-built strata."""
import pytest

from softrole.report import METRICS, contrast_groups, difference_in_differences

A = {"model": "shared", "communication": "real", "max_steps": 80}
B = {"model": "shared", "communication": "binary", "max_steps": 80}


def rows(values):
    return [{"training_seed": seed, "episodes": 4, "scenario_ids": ["0", "1", "2", "3"],
             "panel_sha256": "panel", "checkpoint": {"checkpoint_sha256": f"seed{seed}", "model_signature": f"seed{seed}",
                                                        "progress": {"epoch": 1, "updates": 1, "total_steps": 20_000_100, "total_episodes": 4}},
             **{m: value + i for i, m in enumerate(METRICS)}}
            for seed, value in values.items()]


def stratum(config, intervention="none", sham=False, failure=False, values=None, **extra):
    per_seed = rows(values)
    return dict(config=config, composition=[3, 2], environment_version="corrected-observation-v1",
                source_sha256="s", evaluation_version=2, evaluator_source_sha256="e",
                scheduled_sensor_failure=failure, sham=sham, intervention=intervention,
                training_seeds=len(per_seed), episodes=4 * len(per_seed), per_seed=per_seed,
                metrics={}, **extra)


def test_contrasts_match_strata_on_everything_but_the_configuration():
    groups = [stratum(A, values={0: 1., 1: 2., 2: 6.}), stratum(B, values={0: .5, 1: .5, 2: .5}),
              stratum(A, intervention="comm_off", values={0: 3., 1: 3., 2: 3.}, intervention_offset_from_event=0),
              stratum(B, intervention="comm_off", values={0: 1., 1: 1., 2: 1.}, intervention_offset_from_event=0),
              stratum(A, failure=True, values={0: 9., 1: 9., 2: 9.})]  # no B partner
    result = contrast_groups(groups, A, B, bootstrap_samples=200, seed=1, label="A_minus_B")
    assert [r["intervention"] for r in result] == ["none", "comm_off"]
    nominal = result[0]
    assert nominal["contrast"] == "A_minus_B" and nominal["left_config"] == A and nominal["right_config"] == B
    assert [r["success_rate"] for r in nominal["per_seed"]] == [.5, 1.5, 5.5]
    assert nominal["metrics"]["success_rate"]["mean"] == pytest.approx(2.5)
    assert nominal["metrics"]["team_return"]["mean"] == pytest.approx(2.5)  # offsets cancel inside the difference
    low, high = nominal["metrics"]["success_rate"]["ci95"]
    assert .5 <= low <= 2.5 <= high <= 5.5
    assert result[1]["metrics"]["success_rate"]["mean"] == pytest.approx(2.)
    assert "sham" in nominal and "per_seed" in nominal and "config" not in nominal
    with pytest.raises(ValueError, match="identical nonempty training-seed sets"):
        contrast_groups([stratum(A, values={0: 1.}), stratum(B, values={0: 1., 1: 1.})], A, B, 10)
    with pytest.raises(ValueError, match="different configurations"):
        contrast_groups(groups, A, A, 10)
    with pytest.raises(ValueError, match="Duplicate"):
        contrast_groups(groups + [stratum(B, values={0: 0.})], A, B, 10)


def paired(config, intervention, values, offset=0):
    group = stratum(config, intervention=intervention, failure=True, values=values)
    group.pop("sham")
    group.update(contrast="failure_minus_sham", intervention_offset_from_event=offset if intervention != "none" else None)
    return group


def test_difference_in_differences_subtracts_the_uninterrupted_pair_within_seed():
    pairs = [paired(A, "none", {0: -1., 1: -2., 2: -3.}), paired(A, "comm_off", {0: -4., 1: -4., 2: -4.}),
             paired(A, "freeze_all", {0: -1., 1: -2., 2: -3.}), paired(B, "comm_off", {0: 0., 1: 0., 2: 0.})]
    result = difference_in_differences(pairs, bootstrap_samples=100, seed=0)
    assert [(r["intervention"], r["config"]) for r in result] == [("comm_off", A), ("freeze_all", A)]
    comm_off = result[0]
    assert comm_off["contrast"] == "difference_in_differences" and comm_off["intervention_offset_from_event"] == 0
    assert [r["success_rate"] for r in comm_off["per_seed"]] == [-3., -2., -1.]
    assert comm_off["metrics"]["success_rate"]["mean"] == pytest.approx(-2.)
    assert all(r["completion_steps_horizon_capped"] == 0. for r in result[1]["per_seed"])
    assert result[1]["metrics"]["ci95"] if False else result[1]["metrics"]["success_rate"]["ci95"] == [0., 0.]
    with pytest.raises(ValueError, match="Duplicate paired stratum"):
        difference_in_differences(pairs + [paired(A, "none", {0: 0., 1: 0., 2: 0.})], 10)
    contrasted = contrast_groups(result + difference_in_differences([paired(B, "none", {0: 0., 1: 0., 2: 0.}), pairs[3]], 10),
                                 A, B, 10, label="A_minus_B")
    assert len(contrasted) == 1 and contrasted[0]["intervention"] == "comm_off"
    assert contrasted[0]["metrics"]["success_rate"]["mean"] == pytest.approx(-2.)


@pytest.mark.parametrize("damage", ["checkpoint", "panel", "ids", "duplicate", "missing_identity"])
def test_difference_in_differences_rejects_unpaired_seed_evidence(damage):
    reference = paired(A, "none", {0: 0.})
    intervention = paired(A, "comm_off", {0: 1.})
    row = intervention["per_seed"][0]
    if damage == "checkpoint":
        row["checkpoint"]["progress"]["total_steps"] = 21_000_000
    elif damage == "panel":
        row["panel_sha256"] = "other"
    elif damage == "ids":
        row["scenario_ids"] = ["0", "1", "2", "4"]
    elif damage == "duplicate":
        intervention["per_seed"].append(dict(row))
    else:
        row.pop("panel_sha256")
    with pytest.raises(ValueError, match="checkpoint|panels|duplicate"):
        difference_in_differences([reference, intervention], 10)


def test_arm_contrasts_allow_distinct_checkpoints_but_retain_both():
    left, right = stratum(A, values={0: 2.}), stratum(B, values={0: 1.})
    right["per_seed"][0]["checkpoint"]["checkpoint_sha256"] = "binary-weights"
    row = contrast_groups([left, right], A, B, 10)[0]["per_seed"][0]
    assert row["left_checkpoint"] != row["right_checkpoint"]
    assert row["panel_sha256"] == "panel"
    with pytest.raises(ValueError, match="Duplicate left"):
        contrast_groups([left, left, right], A, B, 10)
