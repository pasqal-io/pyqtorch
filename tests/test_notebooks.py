"""Test examples scripts."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

expected_fail = {}  # type: ignore
skip = {}  # type: ignore


def get_ipynb_files(dir: Path) -> List[Path]:
    files = []

    for it in dir.iterdir():
        if it.suffix == ".ipynb":
            files.append(it)
        elif it.is_dir() and it.name != "deprecated":
            files.extend(get_ipynb_files(it))
    return files


# FIXME: refactor choice of notebooks folders
notebooks_dir = Path(__file__).parent.parent.joinpath("docs").resolve()
assert notebooks_dir.exists()
notebooks = get_ipynb_files(notebooks_dir)
notebooks_names = [f"{example.relative_to(notebooks_dir)}" for example in notebooks]
for example, reason in expected_fail.items():
    try:
        notebooks[notebooks_names.index(example)] = pytest.param(  # type: ignore
            example, marks=pytest.mark.xfail(reason=reason)
        )
    except ValueError:
        pass

for example, reason in skip.items():
    try:
        notebooks[notebooks_names.index(example)] = pytest.param(  # type: ignore
            example, marks=pytest.mark.skip(reason=reason)
        )
    except ValueError:
        pass


@pytest.mark.parametrize("notebook", notebooks, ids=notebooks_names)
def test_notebooks(notebook: Path) -> None:
    """Execute docs notebooks as a test, passes if it returns 0."""
    jupyter_cmd = ["-m", "jupyter", "nbconvert", "--to", "python", "--execute"]
    cmd = [sys.executable, *jupyter_cmd, notebook]
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**os.environ}  # type: ignore
    ) as run_example:
        stdout, stderr = run_example.communicate()
        error_string = (
            f"Notebook {notebook.name} failed\n"
            f"stdout:{stdout.decode()}\n"
            f"stderr: {stderr.decode()}"
        )

    if run_example.returncode != 0:
        raise Exception(error_string)
