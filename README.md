# Graphene: MLX Training Framework

Graphene is a robust training framework designed to simplify the development of machine learning applications using Apple's MLX framework. By reducing boilerplate code, Graphene allows developers to focus more on the core logic of their applications and less on the repetitive setup often required in machine learning workflows.

## Features

- **Simplified API**: Graphene provides a simplified API that abstracts away many of the complexities associated with the MLX framework. This allows developers to get started quickly without needing to understand all the underlying details.

- **Boilerplate Reduction**: Graphene is designed to reduce the amount of boilerplate code that developers need to write when using the MLX framework. This leads to cleaner, more readable code and faster development times.

- **Flexible and Extensible**: Graphene is built with flexibility in mind. It can be easily extended and customized to fit your specific needs, making it a versatile tool for a wide range of machine learning tasks.

## Getting Started

To get started with Graphene, you can install it via pip:

```bash
pip install graphene-mlx
```

Once installed, you can import it in your Python scripts as follows:

```python
import graphene
```

For more detailed information on how to use Graphene, please refer to the [official documentation](https://graphene-mlx.readthedocs.io).

## Dev Installation

```bash
git clone git@github.com:radulescupetru/graphene.git
```

Create a conda environment

```bash
conda create -n my_awesome_conda_env
conda activate my_awesome_conda_env
```

Lock the requirements for your specific setup and install them

```bash
cd graphene
./scripts/lock-requirements.sh && ./scripts/install.sh
```

Install dev requirements

```bash
pip install -r requirements/requirements.dev.in
pre-commit install
pre-commit run --all-files
```
