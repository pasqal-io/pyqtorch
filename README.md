# PyQ

A fast large scale emulator for quantum machine learning on a PyTorch backend.

## Installation

To install the library for development, you can go into any virtual environment of your
choice and install it normally with `pip` (including extra dependencies for development):

```
python -m pip install -e .[dev]
```

The recommended way of managing virtual environments is using the `hatch` packaging tool. If
you use it, you can create a virtual environment for testing and install all the dependencies
in the following way:

```
python -m pip install hatch

# either
# get into the virtual environment and install normally with pip
python -m hatch env create
python -m hatch shell
python -m pip install -e .[dev]

# or
# use directly the hatch development environment
python -m hatch env create pyq_test
python -m hatch shell 
# or python -m hatch run <command>
```
