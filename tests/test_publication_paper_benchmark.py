"""Integrity gates for a performance result, independently of its speed."""
from copy import deepcopy

import pytest

from publication_reconstruction.benchmark import compare_runs, validate_episode_ledger
from test_publication_paper_launcher import fake_rows


def ledger():
    updates = [dict(update=i, epoch=1, steps=24, episodes=8) for i in (1, 2)]
    episodes = [dict(update=u, epoch=1, collector=c, collector_episode=e,
                     episode=(u - 1) * 8 + c * 2 + e + 1, steps=3)
                for u in (1, 2) for c in range(4) for e in (0, 1)]
    return episodes, updates


def validate(episodes, updates):
    return validate_episode_ledger(episodes, updates, collectors=4, floor=5, horizon=3)


def test_complete_numeric_ledger_still_requires_trajectory_replay():
    episodes, updates = ledger()
    assert not validate(episodes, updates)
    for row in episodes:
        row['trajectory_sha256'] = 'a' * 64
    assert validate(episodes, updates)
    del episodes[-1]['trajectory_sha256']
    assert not validate(episodes, updates)


@pytest.mark.parametrize('mutation', ['empty', 'missing', 'duplicate', 'wrong_update',
                                     'wrong_collector', 'wrong_local_index', 'over_horizon',
                                     'under_floor', 'bad_digest', 'wrong_steps'])
def test_missing_or_inconsistent_work_is_rejected(mutation):
    episodes, updates = ledger()
    if mutation == 'empty':
        episodes = []
    elif mutation == 'missing':
        episodes.pop()
    elif mutation == 'duplicate':
        episodes[-1] = deepcopy(episodes[-2])
    elif mutation == 'wrong_update':
        episodes[-1]['update'] = 1
    elif mutation == 'wrong_collector':
        episodes[-1]['collector'] = 4
    elif mutation == 'wrong_local_index':
        episodes[-1]['collector_episode'] = 2
    elif mutation == 'over_horizon':
        episodes[-1]['steps'] = 4
    elif mutation == 'under_floor':
        episodes[-1]['steps'] = 1
        updates[-1]['steps'] -= 2
    elif mutation == 'bad_digest':
        episodes[-1]['trajectory_sha256'] = ''
    elif mutation == 'wrong_steps':
        updates[-1]['steps'] += 1
    with pytest.raises(ValueError):
        validate(episodes, updates)


def test_segment_gate_uses_ratio_of_medians_not_median_of_pair_ratios(tmp_path):
    rows = fake_rows(tmp_path)
    # Pairwise ratios have median 2, but typical realized segment throughput
    # falls from 100 to 20: that is not the specified segment-level gain.
    for row in rows:
        rates = (1, 100, 1000) if row['backend'] == 'dgl' else (2, 200, 20)
        row['segment_steps_per_second'] = rates[row['pair'] - 1]
    for result in compare_runs(rows):
        assert result['ratio_of_median_segment_throughput'] == .2
        assert not result['throughput_gate']
