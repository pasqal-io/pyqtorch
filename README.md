# PyQ

A fast large scale emulator for quantum machine learning on a PyTorch backend.


## Installation

To install the library for development, you can go into any virtual environment of your
choice and install it normally with `pip` (including extra dependencies for development):

```
python -m pip install -e .[dev,converters]
```

Available extras:

* `dev`: development dependencies needed only if you want to contribute to PyQ
* `converters`: dependencies needed by the quantum circuit converters to other SDKs


## Contribute

If you want to contribute to the package, make sure to execute tests and MyPy checks
otherwise the automatic pipeline will not pass. To do so, the recommended way is
to use `hatch` for managing the environments:

```shell
hatch env create tests
hatch run --env tests python -m pytest -vvv --cov pyqtorch tests
hatch run --env tests python -m mypy pyqtorch tests
```

If you don't want to use `hatch`, you can use the environment manager of your choice (e.g. Conda) and execute
the following:

```shell
pip install -e .[dev]
pytest -vvv --cov pyqtorch tests
mypy pyqtorch tests
```
