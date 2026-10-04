"""Mocked collector timing; no policy training or environment rollout."""
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "publication_reconstruction/runtime"


def test_episode_timing_preserves_rollout_records_and_random_streams():
    # Isolate the runtime's historical import names from the repository modules.
    code = r'''
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, sys.argv[1])
import numpy as np
import torch
import trainer as runtime

def seed():
    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)

def draw():
    return [random.random(), float(np.random.random()), float(torch.rand(()))]

seed()
expected_rewards = [draw(), draw()]
expected_next = draw()
seed()

trainer = runtime.Trainer.__new__(runtime.Trainer)
trainer.args = SimpleNamespace(eval=False, hetcomm=False, batch_size=4, max_steps=3)
trainer.last_step = False
calls = []
def episode(epoch):
    calls.append(epoch)
    index = len(calls) - 1
    steps = (2, 3)[index]
    reward = draw()
    trainer.last_episode_record = dict(collector=2, steps=steps, success=index == 1,
        terminated=index == 1, num_agents=3, reward_per_agent=reward,
        team_return=sum(reward), mean_agent_return=sum(reward) / 3)
    transition = runtime.Transition(*([None] * len(runtime.Transition._fields)))
    return [transition] * steps, dict(num_steps=steps, reward=np.array(reward),
                                     success=int(index == 1))
trainer.get_episode = episode
ticks = iter((10.0, 10.125, 20.0, 20.75))
runtime.time = SimpleNamespace(monotonic=lambda: next(ticks))

batch, stats = trainer.run_batch(9)
assert calls == [9, 9]
assert stats['num_steps'] == 5 and stats['num_episodes'] == 2
assert len(batch.state) == 5 and not trainer.last_step
records = stats['_episode_records']
assert [r['rollout_wall_time_seconds'] for r in records] == [0.125, 0.75]
assert [r['steps'] for r in records] == [2, 3]
assert [r['collector_episode'] for r in records] == [0, 1]
assert [r['success'] for r in records] == [False, True]
for row, rewards in zip(records, expected_rewards):
    assert row['collector'] == 2 and row['reward_per_agent'] == rewards
    assert row['team_return'] == sum(rewards)
    assert row['mean_agent_return'] == sum(rewards) / 3
np.testing.assert_array_equal(stats['reward'], np.array(expected_rewards).sum(axis=0))
assert draw() == expected_next
'''
    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    result = subprocess.run([sys.executable, "-c", code, str(RUNTIME)], cwd=ROOT,
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
