"""Locate live checkout files and both recorded source-archive layouts."""
from pathlib import Path


def repository_root():
    """Return the checkout root, keeping outputs/configuration outside ``src``."""
    root = Path(__file__).resolve().parents[1]
    if root.name == "src":
        root = root.parent
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError("This command requires the HetNet source checkout")
    return root


def code_root(root):
    """Locate imports without changing names recorded in historical manifests."""
    root = Path(root).resolve()
    if (root / "src" / "softrole" / "__init__.py").is_file():
        return root / "src"
    return root
