"""Generate environment.yml from pyproject.toml.

Routes conda-native packages (GDAL, numpy, scipy, etc.) to the conda
dependencies section under conda-forge, and everything else to pip.

Usage:
    python scripts/sync_env_yml.py          # runtime deps only
    python scripts/sync_env_yml.py --dev    # include dev deps
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
ENV_YML = ROOT / "environment.yml"

CONDA_NATIVE = {
    "gdal",
    "libgdal-hdf5",
    "matplotlib",
    "numpy",
    "pillow",
    "scipy",
    "scikit-image",
    "scikit-learn",
    "tqdm",
}

CONDA_EXTRAS = [
    "libgdal-hdf5",
]


def _strip_version(spec: str) -> str:
    return re.split(r"[<>=!~\[]", spec, maxsplit=1)[0].strip().lower()


def _to_conda_spec(spec: str) -> str:
    return spec.replace("==", "=")


def generate(include_dev: bool = False) -> str:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]
    name = project["name"]
    python_req = project.get("requires-python", ">=3.10")

    deps: list[str] = list(project.get("dependencies", []))
    if include_dev:
        dev_group = data.get("dependency-groups", {}).get("dev", [])
        deps.extend(dev_group)

    conda_deps: list[str] = []
    pip_deps: list[str] = []

    for spec in deps:
        pkg = _strip_version(spec)
        if pkg in CONDA_NATIVE:
            conda_deps.append(_to_conda_spec(spec))
        else:
            pip_deps.append(spec)

    for extra in CONDA_EXTRAS:
        if not any(_strip_version(d) == extra for d in conda_deps):
            conda_deps.append(extra)

    conda_deps.sort(key=str.lower)

    lines = [
        f"# Auto-generated from pyproject.toml — do not edit by hand.",
        f"# Regenerate: python scripts/sync_env_yml.py{'  --dev' if include_dev else ''}",
        f"",
        f"name: {name}",
        f"channels:",
        f"  - conda-forge",
        f"  - defaults",
        f"dependencies:",
        f"  - python{python_req}",
    ]
    for dep in conda_deps:
        lines.append(f"  - {dep}")

    lines.append(f"  - pip:")
    lines.append(f"    - -e .")
    for dep in sorted(pip_deps, key=str.lower):
        lines.append(f"    - {dep}")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync environment.yml from pyproject.toml")
    parser.add_argument("--dev", action="store_true", help="Include dev dependencies")
    args = parser.parse_args()

    content = generate(include_dev=args.dev)
    ENV_YML.write_text(content, encoding="utf-8")
    print(f"Wrote {ENV_YML.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
