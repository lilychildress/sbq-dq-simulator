# Bias Field-Free Simulator : A Simulator of Double Quantum Experiments

## Poetry
- Install `poetry` following the [documentation instructions](https://python-poetry.org/docs/).
- `poetry install` to use the commited `poetry.lock` file.

- `poetry update` when you want to install by first updating and rewriting the `poetry.lock` file. Note that you should `poetry cache clear --all .` to make sure all `*.latest` versions are properly accounted for.

## Test code coverage
```
poetry run coverage run -m unittest
```
```
poetry run coverage report
```
Edit configuration file `.coveragerc` if needed

## General

- The simulations in this repository rely on QuTip for generating fundamental elements. However, some solvers ultimately depend solely on Numpy once the mathematical framework is established.

- This repository offers basic examples on how to use the different solvers in various situations.

## Licensing
This repository is licensed under the GNU GENERAL PUBLIC LICENSE Version 2.0
- For more details on licensing and what you are permitted/not permitted to do, see [license details](LICENSE.md).
