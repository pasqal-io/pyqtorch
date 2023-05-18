# How to Participate

We're grateful for your interest in participating in PyQ! Please follow our guidelines to ensure a smooth contribution process.

## Reporting an Issue or Proposing a Feature

Your course of action will depend on your objective, but generally, you should start by creating an issue. If you've discovered a bug or have a feature you'd like to see added to **PyQ**, feel free to create an issue on [PyQ's GitHub issue tracker](https://github.com/pasqal-io/PyQ/issues). Here are some steps to take:

1. Quickly search the existing issues using relevant keywords to ensure your issue hasn't been addressed already.
2. If your issue is not listed, create a new one. Try to be as detailed and clear as possible in your description.

- If you're merely suggesting an improvement or reporting a bug, that's already excellent! We thank you for it. Your issue will be listed and, hopefully, addressed at some point.
- However, if you're willing to be the one solving the issue, that would be even better! In such instances, you would proceed by preparing a [Pull Request](#submitting-a-pull-request).

## Submitting a Pull Request

We're excited that you're eager to contribute to PyQ! For general contributions, we combine two Git workflows: the [Forking workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow) and the [Gitflow workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow). If you're unfamiliar with these concepts, don't worry, you can still contribute by following the detailed instructions below. Essentially, this workflow involves making a fork from the main PyQ repository and working on a branch off `develop` (**not** `master`). Thus, you'll start your branch from `develop` and end with a pull request that merges your branch back to `develop`. The PyQ development team will handle any `hotfix`, which is the only exception to this rule.

Here's the process for making a contribution:

0. Fork the PyQ repository and mark the main PyQ repository as the `upstream`. You only need to do this once. Click the "Fork" button at the upper right corner of the [repo page](https://github.com/pasqal-io/PyQ) to create a new GitHub repo at `https://github.com/USERNAME/PyQ`, where `USERNAME` is your GitHub ID. Then, `cd` into the directory where you want to place your new fork and clone it:

    ```bash
    git clone https://github.com/USERNAME/PyQ.git
    ```

    **Note**: Replace `USERNAME` with your own GitHub ID.

   Next, navigate to your new PyQ fork directory and mark the main PyQ repository as the `upstream`:

   ```bash
   git remote add upstream https://github.com/pasqal-io/PyQ.git

## SETUP

Make sure to execute tests and MyPy/ruff checks
otherwise the automatic pipeline will not pass. To do so, the recommended way is
to use `hatch` for managing the environments:

```shell
pip install hatch
hatch -v shell
hatch run test
```

If you don't want to use `hatch`, you can use the environment manager of your
choice (e.g. Conda) and execute the following:

```shell
pip install pytest
pip install -e .[dev]
pytest
```

### Useful Things for your workflow

1. Use `pre-commit` hooks to make sure that the code is properly linted before pushing a new commit.

```shell
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

2.  Make sure that the unit tests and type checks are passing since the merge request will not be accepted if the automatic CI/CD pipeline do not pass.

```shell
hatch run test
```
Or without hatch:

```shell
pytest
```
