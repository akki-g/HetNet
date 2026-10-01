"""Bounded probes of exact Git source, with no import from the working tree.

Run with .venv/bin/python from the repository root. Class/function AST nodes are
compiled unchanged directly from git show. They receive only standard external
library globals. Historical np.int is supplied as Python int through a namespace
proxy; this compatibility shim does not change source or numerical algorithms.
No policies are trained and no historical source is repaired by these probes.
"""
import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import random
import subprocess
import time
from types import SimpleNamespace

import gym
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
GIT = OUT.parent / "history.git"


def git(*args):
    return subprocess.check_output(["git", f"--git-dir={GIT}", *args])


def source(commit, path):
    return git("show", f"{commit}:{path}").decode()


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class OldNumpy:
    def __getattr__(self, key):
        return int if key == "int" else getattr(np, key)


def compile_node(text, class_name, method=None):
    cls = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = cls if method is None else next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    ns = dict(torch=torch, np=OldNumpy(), gym=gym, spaces=gym.spaces,
              time=time, math=math, random=random, copy=copy)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<exact historical Git AST>", "exec"), ns)
    return ns[class_name if method is None else method], ns


def extraction(text):
    f, _ = compile_node(text, "UAVNetA2CEasy", "get_obs_features")
    position, _ = compile_node(text, "UAVNetA2CEasy", "remove_excess_action_features_from_all")
    rows = []
    for task, cells, num_p, num_a in [("pp", 25, 3, 0), ("pcp", 25, 2, 1), ("fc", 9, 2, 1)]:
        schema = SimpleNamespace(num_P=num_p, num_A=num_a, in_dim={"P": 29, "A": 25, "state": 4}, P_s=25, obs_squares=cells)
        x = torch.arange((num_p + num_a) * cells * 29).reshape(1, num_p + num_a, -1).double()
        actual = f(schema, x).reshape(num_p, cells, 4)
        expected = x.reshape(num_p + num_a, cells, 29)[:num_p, :, 25:]
        visible = []
        for cell in range(cells):
            obs = torch.zeros_like(x)
            obs[0, 0, cell * 29 + 27] = 1
            if torch.any(f(schema, obs)):
                visible.append(cell)
            assert not torch.any(position(schema, obs)[0])
        rows.append(dict(task=task, relative_cells=cells,
                         misplaced_entries_per_P=int((actual[0] != expected[0]).sum()),
                         target_cells_reaching_model=visible,
                         output_shape=list(f(schema, x).shape)))
    return rows


def setup_env(text, classname, vision):
    cls, namespace = compile_node(text, classname)
    env = cls()
    parser = argparse.ArgumentParser()
    env.init_args(parser)
    args = parser.parse_args([])
    args.max_steps = 300
    args.eval = False
    args.vision = vision
    env.multi_agent_init(args)
    return env, namespace


def pcp(text):
    env, _ = setup_env(text, "PredatorCaptureEnv", 2)
    np.random.seed(501)
    env.reset()
    env.predator_loc[:] = [[2, 1], [4, 4]]
    env.predator_capture_loc[:] = [2, 2]
    env.prey_loc[:] = [2, 2]
    blinded = env._get_obs()
    env.action_blind = False
    env.A_agents_have_vision = False
    unmasked = env._get_obs()
    # Only compare the P view: allowing A vision must not change a P sensor.
    b = int(blinded[0, 2, 3, env.PREY_CLASS])
    u = int(unmasked[0, 2, 3, env.PREY_CLASS])
    assert (b, u) == (-1, 1)
    # Same source supplies PP via PCP with no action agents: no alias masking.
    env.predator_capture_loc = np.empty((0, 2), dtype=int)
    env.action_blind = True
    pp = env._get_obs()
    assert pp[0, 2, 3, env.PREY_CLASS] == 1
    return dict(P_target_entry_with_blind_A=b, P_target_entry_without_A_mask=u,
                PP_no_A_has_target_entry=int(pp[0, 2, 3, env.PREY_CLASS]),
                mechanism="A view writes -1 into the shared base grid and existing P views")


def fc(env_text, wildfire_text):
    env, ns = setup_env(env_text, "FireCommanderEnv", 1)
    result = {}
    if wildfire_text is None:
        result["reset"] = dict(status="dependency absent", missing="WildFire_Simulate_Original.py")
        return result
    wildfire, _ = compile_node(wildfire_text, "WildFire")
    ns["WildFire"] = wildfire
    np.random.seed(501)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        try:
            env.reset()
            result["reset"] = dict(status="ok")
        except Exception as e:
            result["reset"] = dict(status="error", exception=type(e).__name__, message=str(e))
    result["reset_stdout"] = log.getvalue()

    # Probe exact propagation and reward methods with explicit physical state.
    # This bypasses the failing historical initializer; it is not a claim that
    # that historical full training command can execute.
    env.predator_loc = np.array([[4, 4], [3, 4]])
    env.predator_capture_loc = np.array([[0, 0]])
    env.fire_loc = np.array([[0, 2]])
    env.fire_mdl = wildfire(terrain_sizes=[5, 5], hotspot_areas=[[0, 0, 2, 2]], num_ign_points=1, duration=300)
    env.ign_points_all = np.array([[0., 2., 1.]])
    env.previous_terrain_map = env.ign_points_all.copy()
    env.geo_phys_info = dict(spread_rate=np.ones((5, 5)), wind_speed=np.ones((5, 1)), wind_direction=np.ones((5, 1)))
    env.pruned_list = []
    env.discovered_fire = []
    env.nfire = 1
    env.fire_out = False
    env.episode_over = False
    env.stat = dict(enemy_count=[])
    env._set_grid()
    # Call unchanged physical methods directly. Historical _get_obs also uses
    # NumPy membership broadcasting that errors for an empty discovered array
    # under NumPy 1.26; that independent compatibility issue is not repaired.
    env._fire_propagation()
    fires = env.fire_loc.tolist()
    front = env.ign_points_all.tolist()
    env.false_water_drop = np.zeros(1)
    env.fire_extinguished = np.zeros(1)
    env.extinguishing = np.zeros(1)
    env.just_discovered_source = np.zeros(2)
    env.just_discovered_nonsource = np.zeros(2)
    env._take_action(2, 5)
    env._fire_propagation()
    reward = env._get_reward()
    done = env.episode_over
    assert [0, 0] in fires and front == [[0., 0., 0.]]
    assert reward.tolist() == [-.1, -.1, 10.] and not done
    assert env.fire_loc.tolist() == [[0, 2]]
    result["manually_initialized_phantom_probe"] = dict(
        initial_fire=[0, 2], fires_after_stay=fires, front_after_stay=front,
        action_agent_water_drop=[0, 0], rewards=reward.tolist(), done=bool(done),
        residual_fires=env.fire_loc.tolist(),
        execution="exact _fire_propagation, _take_action, _get_reward; initialized physical state and step event arrays explicitly")

    # Inspect FC observation isolation, again with explicit states.
    env.predator_loc[:] = [[2, 1], [4, 4]]
    env.predator_capture_loc[:] = [2, 2]
    env.fire_loc = np.array([[2, 2]])
    env.ign_points_all = np.array([[2., 2., 1.]])
    env.discovered_fire = [np.array([2, 2])]
    env.just_discovered_source = np.zeros(2)
    env.just_discovered_nonsource = np.zeros(2)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        blinded = env._get_obs()
    # Historical FC debug references local c only when action_blind is True;
    # avoid changing the branch. Nonoverlap A gives an independent P baseline.
    env.predator_capture_loc[:] = [0, 4]
    with contextlib.redirect_stdout(log):
        unmasked = env._get_obs()
    b = int(blinded[0, 1, 2, env.FIRE_CLASS])
    u = int(unmasked[0, 1, 2, env.FIRE_CLASS])
    assert (b, u) == (0, 1)
    result["aliasing"] = dict(P_fire_with_overlapping_A=b, P_fire_with_distant_A=u,
                              debug_stdout=log.getvalue())
    return result


def main():
    torch.set_num_threads(1)
    commits = git("rev-list", "--all", "--reverse", "--topo-order").decode().splitlines()
    rows = []
    unique = {}
    for commit in commits:
        names = git("ls-tree", "-r", "--name-only", commit).decode().splitlines()
        selected = [p for p in names if p in ["hetgat/uavnet.py", "WildFire_Simulate_Original.py", "multi_processing.py"]
                    or p.endswith("/ic3net_envs/predator_capture_env.py")
                    or p.endswith("/ic3net_envs/fire_commander_env.py")]
        record = dict(commit=commit, metadata=git("show", "-s", "--format=%aI%n%cI%n%s", commit).decode().splitlines(), files={})
        for path in selected:
            txt = source(commit, path)
            digest = sha(txt)
            record["files"][path] = digest
            unique.setdefault(digest, dict(first_encounter_commit=commit, path=path, text=txt))
        rows.append(record)
    results = {}
    for digest, entry in unique.items():
        txt, path = entry["text"], entry["path"]
        p = OUT / "source_blobs" / f"{digest}.py"
        p.parent.mkdir(exist_ok=True)
        p.write_text(txt)
        if path.endswith("uavnet.py"):
            results[digest] = dict(probe="sensory_extraction", rows=extraction(txt))
        elif path.endswith("predator_capture_env.py"):
            results[digest] = dict(probe="pcp_observation_aliasing", **pcp(txt))
        elif path == "multi_processing.py":
            results[digest] = dict(probe="static_only", cached_gradient_pointers="if self.grads is None:" in txt,
                                   zero_grad_without_set_to_none="optimizer.zero_grad()" in txt,
                                   caveat="Torch-version-dependent issue; not asserted to have affected original training")
    # FC combinations by source hash cover all distinct same-commit physics.
    pairs = {}
    for record in rows:
        f = record["files"]
        ep = next((p for p in f if p.endswith("fire_commander_env.py")), None)
        if ep:
            key = f[ep] + ":" + f.get("WildFire_Simulate_Original.py", "missing")
            if key not in pairs:
                pairs[key] = dict(commit=record["commit"], env_path=ep,
                                  results=fc(unique[f[ep]]["text"], unique[f["WildFire_Simulate_Original.py"]]["text"] if "WildFire_Simulate_Original.py" in f else None))
            record["fc_pair"] = key
    output = dict(methodology=__doc__, versions=dict(torch=torch.__version__, numpy=np.__version__, gym=gym.__version__),
                  script_sha256=sha(Path(__file__).read_text()), commits=rows,
                  unique_blobs={h: {k:v for k,v in e.items() if k != "text"} for h,e in unique.items()},
                  probes_by_blob=results, fc_pairs=pairs)
    (OUT / "findings.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(dict(commits=len(rows), unique_blobs=len(unique), probes=len(results), fc_pairs=len(pairs)), indent=2))


if __name__ == "__main__":
    main()
