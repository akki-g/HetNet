"""Load the paper simulator from verified bytes, without changing global imports.

Training uses its own immutable source archive. Frozen evaluation uses the
checkpoint's archive, never whichever simulator happens to be installed/imported.
The only redirected import is the FC simulator's sibling wildfire module.
"""
import builtins
import hashlib
import json
from pathlib import Path
from types import ModuleType


SIMULATOR_FILES = (
    "publication_reconstruction/runtime/WildFire_Simulate_Original.py",
    "publication_reconstruction/runtime/envs/ic3net_envs/fire_commander_env.py",
    "publication_reconstruction/runtime/envs/ic3net_envs/predator_capture_env.py",
)
_CLASSES = {}


def _digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def simulator_identity(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
             for name in SIMULATOR_FILES}
    return {"sha256": _digest(files), "files": files}


def source_binding(root, identity=None):
    """Bind an explicit source root to a complete, validated simulator identity."""
    root = Path(root).resolve()
    current = simulator_identity(root)
    if identity is not None and current != identity:
        raise ValueError("Paper simulator source bytes differ from the recorded identity")
    return {"root": str(root), **current}


def checkpoint_binding(checkpoint, saved):
    """Verify a paper checkpoint's source-manifest binding before executing code."""
    source = Path(checkpoint).resolve().parent.parent
    manifest = json.loads((source / "source_manifest.json").read_text())
    files = manifest.get("files", {})
    if (_digest(files) != manifest.get("sha256")
            or manifest.get("sha256") != saved.get("source_sha256")):
        raise ValueError("Checkpoint source manifest identity is inconsistent")
    identity = saved.get("simulator_source")
    if (not isinstance(identity, dict) or set(identity.get("files", {})) != set(SIMULATOR_FILES)
            or _digest(identity["files"]) != identity.get("sha256")
            or any(files.get(name) != digest for name, digest in identity["files"].items())):
        raise ValueError("Checkpoint paper simulator identity is missing or inconsistent")
    return source_binding(source / "source", identity)


def load_environment_class(task, binding=None):
    if task not in ("pp", "pcp", "fc"):
        raise ValueError("Unsupported paper environment task")
    if binding is None:
        binding = source_binding(Path(__file__).resolve().parents[1])
    root = Path(binding["root"]).resolve()
    identity = {key: binding[key] for key in ("sha256", "files")}
    if set(identity["files"]) != set(SIMULATOR_FILES) or _digest(identity["files"]) != identity["sha256"]:
        raise ValueError("Invalid paper simulator source identity")
    # Read once, verify, then compile these same bytes (no check/load race).
    payloads = {}
    for name, digest in identity["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Paper simulator source escapes its archive")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Paper simulator source changed after binding")
        payloads[name] = data
    key = (str(root), identity["sha256"], task == "fc")
    if key not in _CLASSES:
        wildfire = ModuleType("_softrole_paper_wildfire_" + identity["sha256"])
        wildfire.__file__ = str(root / SIMULATOR_FILES[0])
        exec(compile(payloads[SIMULATOR_FILES[0]], wildfire.__file__, "exec"), wildfire.__dict__)

        def simulator_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "WildFire_Simulate_Original" and level == 0:
                return wildfire
            return builtins.__import__(name, globals, locals, fromlist, level)

        filename = SIMULATOR_FILES[1 if task == "fc" else 2]
        module = ModuleType("_softrole_paper_env_" + identity["sha256"])
        module.__file__ = str(root / filename)
        module.__dict__["__builtins__"] = {**vars(builtins), "__import__": simulator_import}
        exec(compile(payloads[filename], module.__file__, "exec"), module.__dict__)
        _CLASSES[key] = getattr(module, "FireCommanderEnv" if task == "fc" else "PredatorCaptureEnv")
    return _CLASSES[key]
