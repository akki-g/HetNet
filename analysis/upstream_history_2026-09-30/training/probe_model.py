"""Isolated historical model/loss checks; no simulator or optimizer updates.

Run with the repository's pinned Python, passing snapshot directory and output.
Imports resolve exclusively to exported historical source for HetNet modules.
Synthetic all-zero observations test shape/control flow, not task performance.
"""
import ast
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

snapshot = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(snapshot))
import numpy as np
import torch
import dgl

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
from hetgat.policy import A2CPolicy
from action_utils import select_action

rows = []
for mode in ("real", "binary"):
    for task, num_p, num_a, vision in (("pp", 3, 0, 2), ("pcp", 2, 1, 2), ("fc", 2, 1, 1)):
        torch.manual_seed(100)
        row = dict(mode=mode, task=task, num_p=num_p, num_a=num_a, vision=vision)
        phase = "construct"
        try:
            raw = dict(vision=vision, P=29, A=25, state=4)
            model_dims = dict(P=29, A=25, state=4)
            policy = A2CPolicy(raw, model_dims, dict(P=16, A=16, state=16),
                               dict(P=5, A=6, state=8), num_p, num_a,
                               num_heads=4, msg_dim=16, device=torch.device("cpu"),
                               gamma=1, lr=0.0001, weight_decay=0,
                               use_real=(mode == "real"), use_CNN=False,
                               per_class_critic=True, with_two_state=True,
                               obs=(2*vision+1)**2, tensor_obs=False, action_vision=-1)
            before = {name: param.detach().clone() for name, param in policy.model.named_parameters()}
            hidden = policy.model.init_hidden(1)
            row["parameters"] = sum(p.numel() for p in policy.model.parameters())
            row["top_modules"] = list(dict(policy.model.named_children()))
            row["recurrence"] = {
                "observation": [policy.model.f_module_obs.input_size, policy.model.f_module_obs.hidden_size],
                "position": [policy.model.f_module_stat.input_size, policy.model.f_module_stat.hidden_size],
            }
            row["policy_optimizer_class"] = type(policy.optimizer).__name__
            row["policy_optimizer_steps"] = 0
            def forbidden_step(*args, **kwargs):
                row["policy_optimizer_steps"] += 1
                raise AssertionError("Historical policy tried to step its optimizer")
            policy.optimizer.step = forbidden_step
            for t in range(3):
                phase = "forward"
                obs = torch.zeros(1, num_p+num_a, 29*(2*vision+1)**2)
                logits, value, hidden = policy.batch_select_action_universal([obs, hidden], 0)
                row["logit_shapes"] = {k: list(v.shape) for k, v in logits.items()}
                phase = "select_action"
                action = select_action(SimpleNamespace(hetgat=True, nfriendly_P=num_p, nfriendly_A=num_a), logits)
                actual = [action[0, :, 0].numpy()]
                phase = "append_logprobs"
                policy.append_log_probs_properly(actual)
                policy.batch_rewards[0].append(np.full(num_p+num_a, -.1))
            phase = "loss_backward"
            loss = policy.batch_finish_per_class(1, sim_time=3, num_P=num_p, num_A=num_a)
            row["loss"] = {k: float(v) for k, v in loss.items()}
            grads = [p.grad for p in policy.model.parameters() if p.grad is not None]
            row["finite_gradients"] = all(bool(torch.isfinite(g).all()) for g in grads)
            row["gradient_norm_after_policy"] = float(torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])))
            row["weights_unchanged"] = all(torch.equal(before[name], p) for name, p in policy.model.named_parameters())
            row["status"] = "pass"
        except Exception as error:
            row.update(status="error", phase=phase, exception=type(error).__name__, message=str(error),
                       traceback=traceback.format_exc())
        rows.append(row)

# Execute the exact historical Trainer.train_batch body on an instrumented dummy
# collector. RecordingOptimizer observes the step, but never updates any weight.
tree = ast.parse((snapshot / "trainer.py").read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "train_batch")
module = ast.Module(body=[method], type_ignores=[])
scope = dict(np=np, merge_stat=lambda src, dest: dest.update(src))
exec(compile(ast.fix_missing_locations(module), str(snapshot / "trainer.py"), "exec"), scope)
parameter = torch.nn.Parameter(torch.tensor(2.0))
events = []
class RecordingOptimizer:
    def zero_grad(self):
        events.append("trainer_optimizer.zero_grad")
    def step(self):
        events.append("trainer_optimizer.step")
        events.append({"gradient_at_step": float(parameter.grad)})
def compute_grad(batch):
    parameter.grad = torch.tensor(.5)
    events.append("compute_grad")
    return {}
dummy = SimpleNamespace(reset_memory_peak=lambda: None,
                        run_batch=lambda epoch: (None, {"num_steps": 500}),
                        args=SimpleNamespace(hetcomm=False), optimizer=RecordingOptimizer(),
                        compute_grad=compute_grad, params=[parameter],
                        cpu_memory_peak=0, gpu_memory_peak=0)
scope["train_batch"](dummy, 0)

record = dict(snapshot=str(snapshot), python=sys.version, torch=torch.__version__,
              dgl=dgl.__version__, numpy=np.__version__, models=rows,
              trainer_instrumented_events=events,
              notes=["No simulator execution or optimizer parameter update performed.",
                     "Runtime compatibility is tested under the current pinned environment, not an unavailable 2022 package lock.",
                     "The historical trainer optimizer constructor is RMSprop; the policy owns Adam but does not step it."])
output.write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps({"snapshot":snapshot.name,"models":[{k:r[k] for k in ("task","mode","status","phase","message") if k in r} for r in rows],"events":events}))
