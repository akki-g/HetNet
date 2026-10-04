"""Read-only synthetic loss probe; does not collect episodes or step an optimizer."""
import json
import os
from pathlib import Path
import sys
import warnings

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['DGLBACKEND'] = 'pytorch'
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'publication_reconstruction/runtime'))

import numpy as np
import torch
from hetgat.policy import A2CPolicy

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.manual_seed(810)
policy = A2CPolicy.__new__(A2CPolicy)
policy.per_class_critic = True
policy.num_A = 1
policy.device = torch.device('cpu')
policy.gamma, policy.lmbda, policy.max_grad_norm = 1, .95, .75
policy.model = torch.nn.Linear(1, 1)
policy.optimizer = torch.optim.Adam(policy.model.parameters(), lr=.001)
value = policy.model(torch.ones(1, 1))
policy.batch_rewards = [[np.array([-.05, -.05, -.05])]]
policy.batch_P_critics, policy.batch_A_critics = [[value]], [[value]]
policy.batch_P_log_probs = [[value.reshape(1).repeat(2)]]
policy.batch_A_log_probs = [[value.reshape(1)]]
with warnings.catch_warnings(record=True) as caught:
    losses = policy.batch_finish_per_class(1, 1, num_P=2, num_A=1)
result = {
    'scope': 'Actual loss method with synthetic one-episode, one-step 2P1A buffers; no policy rollout or optimizer step',
    'loss_is_finite': {key: bool(np.isfinite(value)) for key, value in losses.items()},
    'gradient_is_finite': {name: bool(torch.isfinite(parameter.grad).all())
                           for name, parameter in policy.model.named_parameters()},
    'warnings': [str(warning.message) for warning in caught],
}
assert not result['loss_is_finite']['total']
assert not any(result['gradient_is_finite'].values())
print(json.dumps(result, indent=2, allow_nan=False))
