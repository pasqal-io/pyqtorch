# pyqtorch

**pyqtorch** is a [PyTorch](https://pytorch.org/)-based state vector simulator designed for quantum machine learning.
It acts as the main backend for [`Qadence`](https://github.com/pasqal-io/qadence), a digital-analog quantum programming interface.
`pyqtorch` allows for writing fully differentiable quantum programs using both digital and analog operations; enabled via a intuitive, torch-based syntax.

[![Linting / Tests/ Documentation](https://github.com/pasqal-io/pyqtorch/actions/workflows/test.yml/badge.svg)](https://github.com/pasqal-io/pyqtorch/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Pypi](https://badge.fury.io/py/pyqtorch.svg)](https://pypi.org/project/pyqtorch/)
![Coverage](https://img.shields.io/codecov/c/github/pasqal-io/pyqtorch?style=flat-square)


## Installation guide

`pyqtorch` can be installed from PyPI with `pip` as follows:

```bash
pip install pyqtorch
```

## Install from source

We recommend to use the [`hatch`](https://hatch.pypa.io/latest/) environment manager to install `pyqtorch` from source:

```bash
python -m pip install hatch

# get into a shell with all the dependencies
python -m hatch shell

# run a command within the virtual environment with all the dependencies
python -m hatch run python my_script.py
```

Please note that `hatch` will not combine nicely with other environment managers such Conda. If you want to use Conda, install `pyqtorch` from source using `pip`:

```bash
# within the Conda environment
python -m pip install -e .
```

## Contributing

Please refer to [CONTRIBUTING](CONTRIBUTING.md) to learn how to contribute to `pyqtorch`.
