# How to Contribute

We're grateful for your interest in participating in pyqtorch! Please follow our guidelines to ensure a smooth contribution process.

## Reporting an Issue or Proposing a Feature

Your course of action will depend on your objective, but generally, you should start by creating an issue. If you've discovered a bug or have a feature you'd like to see added to **PyQ**, feel free to create an issue on [pyqtorch's GitHub issue tracker](https://github.com/pasqal-io/pyqtorch/issues). Here are some steps to take:

1. Quickly search the existing issues using relevant keywords to ensure your issue hasn't been addressed already.
2. If your issue is not listed, create a new one. Try to be as detailed and clear as possible in your description.

- If you're merely suggesting an improvement or reporting a bug, that's already excellent! We thank you for it. Your issue will be listed and, hopefully, addressed at some point.
- However, if you're willing to be the one solving the issue, that would be even better! In such instances, you would proceed by preparing a [Pull Request](#submitting-a-pull-request).

## Submitting a Pull Request

We're excited that you're eager to contribute to pyqtorch! To contribute, fork the `main` branch of pyqtorch repository and once you are satisfied with your feature and all the tests pass create a [Pull Request](https://github.com/pasqal-io/pyqtorch/pulls).

Here's the process for making a contribution:

Click the "Fork" button at the upper right corner of the [repo page](https://github.com/pasqal-io/pyqtorch) to create a new GitHub repo at `https://github.com/USERNAME/pyqtorch`, where `USERNAME` is your GitHub ID. Then, `cd` into the directory where you want to place your new fork and clone it:

```shell
git clone https://github.com/USERNAME/pyqtorch.git
```

Next, navigate to your new pyqtorch fork directory and mark the main pyqtorch repository as the `upstream`:

```shell
git remote add upstream https://github.com/pasqal-io/pyqtorch.git
```

## Setting up your development environment

We recommended to use `hatch` for managing environments:

To develop within pyqtorch, use:
```shell
pip install hatch
hatch -v shell
```

To run pyqtorch tests, use:

```shell
hatch -e tests run test
```

If you don't want to use `hatch`, you can use the environment manager of your
choice (e.g. Conda) and execute the following:

```shell
pip install pytest

pip install -e .
pytest
```

### Useful Things for your workflow: Linting and Testing

Use `pre-commit` hooks to make sure that the code is properly linted before pushing a new commit. Make sure that the unit tests and type checks are passing since the merge request will not be accepted if the automatic CI/CD pipeline do not pass.

Without `hatch`:

```shell
pip install pytest

pip install -e .
pip install pre-commit
pre-commit install
pre-commit run --all-files
pytest
```

And with `hatch`:

```shell
hatch -e tests run pre-commit run --all-files
hatch -e tests run test
```

Make sure your docs build too!

With `hatch`:

```shell
hatch -e docs run mkdocs build --clean --strict
```

Without `hatch`, `pip` install those libraries first:
"mkdocs",
"mkdocs-material",
"mkdocstrings",
"mkdocstrings-python",
"mkdocs-section-index",
"mkdocs-jupyter",
"mkdocs-exclude",
"markdown-exec"


And then:

```shell
 mkdocs build --clean --strict
```
