"""Additional seeds: recipe identity, fresh outputs, provenance and shell routing.

No policy rollout, full training, scheduler or network operation is performed.
"""
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from softrole import pcp_additional_seeds as extension
from softrole import programme
from publication_reconstruction.__main__ import parser, resolve
from publication_reconstruction.study import train_arguments

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "slurm/softrole_pcp_additional_seeds.sbatch"


@pytest.mark.parametrize("index", range(15))
def test_seed_extension_changes_only_seed_and_output_of_existing_recipes(tmp_path, index):
    result = extension.entry(index, tmp_path / "fresh paths ' $(unused)")
    seed = (3, 4, 5)[index % 3]
    assert result["seed"] == seed and result["resume"] is None
    assert result["starting_counts"] == dict(env_steps=0, episodes=0, updates=0)
    if index < 3:
        args = train_arguments(3, tmp_path, reconstruction_spec="paper-v1", message_backend="torch-v1")
        args[args.index("--seed") + 1] = str(seed)
        args[args.index("--output") + 1] = result["output"]
        assert result["arguments"] == args
        assert result["resolved"] == resolve(parser().parse_args(args))
        assert "--resume_checkpoint" not in result["resolved"]["command"]
        assert result["resolved"]["max_env_steps"] == 40_000_000
        assert result["resolved"]["wall_seconds"] == 165600
    else:
        arm = ("l1a_decomposed_beta0", "l1a_decomposed_beta05",
               "c1_real64", "l1b_decomposed_beta1")[(index // 3) - 1]
        baseline = next(e["config"] for e in programme.entries(.001, profile=programme.CAPCON_PROFILE)
                        if e["arm"] == arm)
        assert result["arm"] == arm
        assert result["resolved"] == dict(baseline, seed=seed)
        config = result["resolved"]
        assert (config["actor_coeff"], config["value_coeff"]) == (50., 1.)
        assert (config["optimizer"], config["lr"], config["nprocesses"],
                config["batch_steps"], config["max_steps"]) == ("adam", .001, 4, 500, 80)
        assert (config["critic_mode"], config["credit_beta"]) == {
            "l1a_decomposed_beta0": ("decomposed", 0.),
            "l1a_decomposed_beta05": ("decomposed", .5),
            "c1_real64": ("team", 1.),
            "l1b_decomposed_beta1": ("decomposed", 1.),
        }[arm]
        assert config["nprocesses"] * config["batch_steps"] * config["updates_per_epoch"] * config["epochs"] == config["total_steps"] == 40_000_000
        assert result["episode_log"] == "chunked-gzip" and result["diagnostics_every"] == 50
    assert programme.SEEDS == (0, 1, 2)  # Original research/evaluation default is intact.
    assert not tmp_path.joinpath("fresh paths ' $(unused)").exists()


@pytest.mark.parametrize("value", ["-1", "42", "100", "010", "03", "3.0", "+3", " 3", "3 ", "3;echo", "", None])
def test_parser_rejects_noncanonical_or_out_of_scope_indices(value):
    with pytest.raises(Exception, match="index must"):
        extension.index_value(value)


def test_all_forty_two_outputs_are_unique_and_outside_original_runs(tmp_path):
    paths = [extension.entry(i, tmp_path)["output"] for i in range(42)]
    assert len(set(paths)) == 42
    assert all("seed0" not in p and "seed1" not in p and "seed2" not in p for p in paths)


@pytest.mark.parametrize("index", range(42))
def test_cli_dry_run_never_prepares_or_trains(tmp_path, monkeypatch, capsys, index):
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run reached an execution/preparation gate")
    monkeypatch.setattr(extension, "environment_identity", forbidden)
    monkeypatch.setattr(extension, "prepare", forbidden)
    monkeypatch.setattr(extension, "run_entry", forbidden)
    out = tmp_path / "output"
    assert extension.main(["--index", str(index), "--run-root", str(out), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out) == extension.entry(index, out)
    assert not out.exists()


def test_cli_real_launch_requires_slurm_before_environment_or_files(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm"):
        extension.main(["--index", "0", "--run-root", str(tmp_path / "output")])
    assert not (tmp_path / "output").exists()


def test_capcon_collector_seed_uses_new_root_seed_and_is_work_indexed(monkeypatch, tmp_path):
    import softrole.train as training
    from softrole.config import Config
    observed = []
    monkeypatch.setattr(training, "make_env", lambda *args: None)
    monkeypatch.setattr(training, "make_scenarios", lambda seed, *args: observed.append(seed) or [])
    model = torch.nn.Linear(1, 1).double()
    coordinates = [(seed, worker, update) for seed in (3,4,5) for worker in range(4) for update in (0,13)]
    def sample(order):
        result = {}
        for seed, worker, update in order:
            config = Config(**extension.entry(seed, tmp_path)["resolved"])  # indices 3..5 are beta0
            assert config.seed == seed
            training.collect_batch(model, config, worker, update)
            result[seed,worker,update] = observed[-1]
        return result
    first = sample(coordinates)
    assert first == sample(reversed(coordinates))
    assert len(set(first.values())) == len(coordinates)


def test_hetnet_root_seeds_reach_existing_independent_worker_streams():
    path = ROOT / "publication_reconstruction/runtime/hetnet_ext/recovery.py"
    spec = importlib.util.spec_from_file_location("additional_seed_recovery", path)
    recovery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recovery)
    outer = recovery.capture_rng()
    try:
        def sample(seed, worker):
            recovery.seed_stream(seed, worker)
            return random.random(), float(np.random.random()), float(torch.rand(()))
        coordinates = [(seed, worker) for seed in (3,4,5) for worker in (None,0,1,2,3)]
        values = {key:sample(*key) for key in coordinates}
        assert values == {key:sample(*key) for key in reversed(coordinates)}
        assert len(set(values.values())) == len(coordinates)
    finally:
        recovery.restore_rng(outer)


def baseline_fixture():
    definition = programme.profile_definition(programme.CAPCON_PROFILE)
    return dict(version=programme.MANIFEST_VERSION, programme=programme.CAPCON_PROFILE,
                lr=.001, bounded_check=False, diagnostics_every=50, source_sha256="baseline-source",
                lr_selection_sha256="selection-hash", lr_selection=dict(selected_lr=.001, rule=programme.S1_RULE, bounded_check=False),
                common=definition["common"], presets=json.loads(json.dumps(definition["presets"])),
                arms=programme.arm_table(programme.CAPCON_PROFILE)["arms"],
                entries=programme.entries(.001,profile=programme.CAPCON_PROFILE),
                runtime_override_keys=list(programme.RUNTIME_KEYS), checkpoint_policy=definition["checkpoint_policy"],
                evaluation_policy=definition["evaluation_policy"])


def test_baseline_uses_existing_manifest_and_s1_validation(tmp_path, monkeypatch):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline_fixture()))
    calls = []
    monkeypatch.setattr(programme, "validate_selection", lambda s:calls.append(s))
    result = extension.baseline_manifest(path)
    assert calls and result["original_source_sha256"] == "baseline-source"
    def rejected(selection):
        raise ValueError("Pinned receipt changed")
    monkeypatch.setattr(programme, "validate_selection", rejected)
    with pytest.raises(ValueError, match="receipt"):
        extension.baseline_manifest(path)


@pytest.mark.parametrize("field,value", [("bounded_check",True),("diagnostics_every",49),("lr",.0003)])
def test_rejects_engineering_or_changed_capcon_recipe(tmp_path, monkeypatch, field, value):
    path=tmp_path/"baseline.json"
    baseline=baseline_fixture();baseline[field]=value
    path.write_text(json.dumps(baseline))
    monkeypatch.setattr(programme,"validate_selection",lambda s:s)
    with pytest.raises(ValueError):extension.baseline_manifest(path)


def test_shared_manifest_reuses_exact_identity_and_rejects_checkout_change(tmp_path, monkeypatch):
    import softrole.train as training
    identity = dict(sha256="one",files={},git_commit="commit",git_status="dirty")
    monkeypatch.setattr(training,"source_snapshot",lambda output=None:dict(identity))
    monkeypatch.setattr(extension,"git_identity",lambda:({"commit":"commit","diff_sha256":"x","untracked_files":{}},b"diff"))
    root=tmp_path/"fresh"
    first=extension.prepare(root,{"sha256":"baseline"},{"python":"3.12"})
    assert extension.prepare(root,{"sha256":"baseline"},{"python":"3.12"})==first
    assert (root/"provenance/working_tree.patch").read_bytes()==b"diff"
    original=(root/"manifest.json").read_bytes()
    identity["sha256"]="two"
    with pytest.raises(ValueError,match="changed"):extension.prepare(root,{"sha256":"baseline"},{"python":"3.12"})
    assert (root/"manifest.json").read_bytes()==original


@pytest.mark.parametrize("index",[0,3,6,9,12,15,24,41])
def test_actual_dispatch_preserves_full_recipe_and_rejects_duplicate_output(tmp_path,monkeypatch,index):
    import softrole.train as training
    import publication_reconstruction.__main__ as reconstruction
    selected=extension.entry(index,tmp_path)
    monkeypatch.setattr(training,"source_snapshot",lambda:{"sha256":"source"})
    manifest=tmp_path/"manifest.json";manifest.write_text(json.dumps({"source":{"sha256":"source"}}))
    calls=[]
    monkeypatch.setattr(reconstruction,"main",lambda args:calls.append(args) or 0)
    monkeypatch.setattr(training,"train",lambda config,output,**kwargs:calls.append((programme.normalized(config),str(output),kwargs)))
    assert extension.run_entry(selected,manifest,{"python":"3.12"})==0
    assert len(calls)==1
    if index==0:assert calls[0]==selected["arguments"]
    else:assert calls[0]==(selected["resolved"],selected["output"],dict(episode_log="chunked-gzip",diagnostics_every=50,checkpoint_commits=True))
    with pytest.raises(FileExistsError):extension.run_entry(selected,manifest,{"python":"3.12"})
    assert len(calls)==1


def test_wrong_dependency_activation_fails_before_compiled_imports(tmp_path,monkeypatch):
    (tmp_path/"pyproject.toml").write_text('[project]\ndependencies=["numpy==1.26.4"]\n')
    monkeypatch.setattr(extension,"ROOT",tmp_path)
    monkeypatch.setattr(extension.importlib.metadata,"version",lambda name:"2.0.0")
    with pytest.raises(RuntimeError,match="Wrong environment"):extension.environment_identity()


@pytest.fixture
def shell(tmp_path):
    tools=tmp_path/"tools";tools.mkdir()
    submit=tmp_path/"repo ' $(never-executed)";submit.mkdir()
    capture=tmp_path/"calls.jsonl"
    def tool(name,body):
        path=tools/name;path.write_text(f"#!{sys.executable}\n"+body);path.chmod(0o755);return path
    prefix="import os,sys,json\nwith open(os.environ['CAPTURE'],'a') as f:f.write(json.dumps({'tool':os.path.basename(sys.argv[0]),'args':sys.argv[1:],'threads':os.environ.get('OMP_NUM_THREADS'),'conda':os.environ.get('CONDA_SHLVL')})+'\\n')\n"
    tool("module",prefix+"assert os.environ['CONDA_SHLVL']=='0'\nassert 'CONDA_PREFIX_1' not in os.environ\nraise SystemExit(int(os.environ.get('MODULE_EXIT','0')))\n")
    tool("srun",prefix+"raise SystemExit(int(os.environ.get('RUN_EXIT','0')))\n")
    interpreter=tool("python with spaces",prefix)
    env={**os.environ,"PATH":str(tools)+os.pathsep+os.environ['PATH'],"SLURM_SUBMIT_DIR":str(submit),
         "SLURM_ARRAY_JOB_ID":"123456","SLURM_ARRAY_TASK_ID":"0","HETNET_PYTHON":str(interpreter),
         "CAPTURE":str(capture),"CONDA_SHLVL":"4","CONDA_PREFIX_1":"/stale"}
    env.pop("BASH_ENV",None);env.pop("ENV",None)
    def invoke(arguments=(),**changes):
        result=subprocess.run(['bash',str(SCRIPT),*arguments],env={**env,**changes},capture_output=True,text=True,timeout=10)
        rows=[json.loads(l) for l in capture.read_text().splitlines()] if capture.exists() else []
        return result,rows,submit,interpreter
    return invoke


@pytest.mark.parametrize("index",range(42))
def test_slurm_routes_each_index_without_shell_interpolation(shell,index):
    result,rows,submit,interpreter=shell(SLURM_ARRAY_TASK_ID=str(index))
    assert result.returncode==0,result.stderr
    assert [r['tool'] for r in rows]==['module','srun']
    assert rows[0]['args']==['load','anaconda/anaconda-2024.10']
    assert rows[-1]['args']==['--cpu-bind=cores',str(interpreter),'-u','-m','softrole.pcp_additional_seeds',
                             '--index',str(index),'--run-root',str(submit/'runs/pcp_additional_seeds_v3_123456'),
                             '--capcon-manifest',str(submit/'runs/capcon_l1_40m_20261017/manifest.json')]
    assert rows[-1]['threads']=='1' and rows[-1]['conda']=='0'


@pytest.mark.parametrize("value",['','42','100','03','010','+10','10.0',' 10','10 ','-1','3;echo'])
def test_bad_slurm_indices_never_activate_or_launch(shell,value):
    result,rows,*_=shell(SLURM_ARRAY_TASK_ID=value)
    assert result.returncode==2 and not rows


def test_module_failure_propagates_without_srun(shell):
    result,rows,*_=shell(MODULE_EXIT='23')
    assert result.returncode==23 and [r['tool'] for r in rows]==['module']


def test_srun_failure_propagates(shell):
    result,rows,*_=shell(RUN_EXIT='37')
    assert result.returncode==37 and rows[-1]['tool']=='srun'


def test_unrequested_script_arguments_never_launch(shell):
    result,rows,*_=shell(['--resume'])
    assert result.returncode==2 and not rows


def test_slurm_directives_preserve_resource_recipe():
    subprocess.run(['bash','-n',str(SCRIPT)],check=True)
    content=SCRIPT.read_text()
    directives=dict(l[len('#SBATCH --'):].split('=',1) for l in content.splitlines() if l.startswith('#SBATCH --'))
    assert {k:directives[k] for k in ['account','partition','constraint','nodes','ntasks','cpus-per-task','mem','time','array']} == dict(
        account='cenyioha',partition='normal',constraint='sapphirerapids',nodes='1',ntasks='1',**{'cpus-per-task':'4','mem':'16G','time':'48:00:00','array':'6-41%36'})
    assert content.startswith('#!/bin/bash\n')


def test_pinned_version_accepts_standard_local_build_label(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\ndependencies=["numpy==1.26.4"]\n')
    monkeypatch.setattr(extension, "ROOT", tmp_path)
    monkeypatch.setattr(extension.importlib.metadata, "version", lambda name: "1.26.4+local")
    assert extension.environment_identity()["packages"] == {"numpy": "1.26.4+local"}


def test_actual_paper_worker_branch_does_not_use_adjacent_additive_seeds():
    import ast
    from types import SimpleNamespace
    path = ROOT / "publication_reconstruction/runtime/multi_processing.py"
    tree = ast.parse(path.read_text())
    worker = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MultiProcessWorker")
    method = next(n for n in worker.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    namespace = {"seed_stream": lambda seed, collector: calls.append((seed, collector))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    calls = []
    for seed in (3, 4, 5):
        for worker_id in range(3):
            fake = SimpleNamespace(seed=seed, id=worker_id,
                                   trainer=SimpleNamespace(args=SimpleNamespace(model_spec="paper-v1")),
                                   comm=SimpleNamespace(recv=lambda: "quit"))
            namespace["run"](fake)
    assert calls == [(seed, collector) for seed in (3, 4, 5) for collector in (1, 2, 3)]


@pytest.mark.parametrize("index", range(42))
def test_parser_accepts_canonical_multi_digit_indices(index):
    assert extension.index_value(str(index)) == index


def test_v3_manifest_locks_all_forty_two_mappings(tmp_path, monkeypatch):
    import softrole.train as training
    monkeypatch.setattr(training, "source_snapshot", lambda output=None: {"sha256": "source"})
    monkeypatch.setattr(extension, "git_identity", lambda: ({"untracked_files": {}}, b"diff"))
    plan = extension.prepare(tmp_path, {}, {})
    assert plan["study"] == "pcp-additional-seeds-v3"
    assert [(e["index"], e["arm"], e["seed"]) for e in plan["entries"]] == [
        (3 * group + offset, arm, seed)
        for group, arm in enumerate(("hetnet_pcp_real", "l1a_decomposed_beta0",
                                     "l1a_decomposed_beta05", "c1_real64", "l1b_decomposed_beta1"))
        for offset, seed in enumerate((3, 4, 5))
    ] + [
        (15 + 9 * ki + 3 * bi + si, f"banked_k{k}_decomposed_beta{beta}", seed)
        for ki, k in enumerate((2, 3, 4))
        for bi, beta in enumerate(("1", "05", "0"))
        for si, seed in enumerate((3, 4, 5))
    ]
    assert "not independent certification of banked actors" in plan["scope"]


def test_incomplete_provenance_is_preserved_and_cannot_be_reused(tmp_path, monkeypatch):
    import softrole.train as training
    monkeypatch.setattr(training, "source_snapshot", lambda output=None: {"sha256": "source"})
    monkeypatch.setattr(extension, "git_identity", lambda: ({"untracked_files": {}}, b"diff"))
    archive = tmp_path / "provenance"
    archive.mkdir()
    evidence = archive / "partial-source"
    evidence.write_bytes(b"preserve incomplete capture")
    with pytest.raises(FileExistsError):
        extension.prepare(tmp_path, {}, {})
    assert evidence.read_bytes() == b"preserve incomplete capture"
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("index", [6, 9, 12, 15, 24, 41])
def test_new_wave_dispatch_rejects_changed_source_before_training(tmp_path, monkeypatch, index):
    import softrole.train as training
    selected = extension.entry(index, tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"source": {"sha256": "original"}}))
    monkeypatch.setattr(training, "source_snapshot", lambda: {"sha256": "changed"})
    def forbidden(*args, **kwargs):
        pytest.fail("changed source reached training")
    monkeypatch.setattr(training, "train", forbidden)
    with pytest.raises(ValueError, match="Checkout changed"):
        extension.run_entry(selected, manifest, {})
    assert not Path(selected["output"]).exists()
    assert not (Path(selected["output"]).parent / "launch.json").exists()


@pytest.mark.parametrize("index", range(15, 42))
def test_banked_factorial_changes_only_actor_and_expert_count(tmp_path, index):
    from softrole.config import Config
    k = (2, 3, 4)[(index - 15) // 9]
    beta, shared_start = ((1., 12), (.5, 6), (0., 3))[((index - 15) % 9) // 3]
    offset = (index - 15) % 3
    selected = extension.entry(index, tmp_path)
    shared = extension.entry(shared_start + offset, tmp_path)
    config = selected["resolved"]
    assert config == dict(shared["resolved"], model="banked", experts=k)
    assert (config["critic_mode"], config["credit_beta"]) == ("decomposed", beta)
    Config(**config).validate()
    assert selected["kind"] == "capcon"
    assert selected["episode_log"] == shared["episode_log"] == "chunked-gzip"
    assert selected["diagnostics_every"] == shared["diagnostics_every"] == 50
    assert selected["fresh"] and selected["resume"] is None
    assert selected["starting_counts"] == dict(env_steps=0, episodes=0, updates=0)
    assert selected["output"] != shared["output"]
    assert not Path(selected["output"]).exists()
