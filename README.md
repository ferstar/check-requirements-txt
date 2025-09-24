# check-requirements-txt

A tool (and also a pre-commit hook) to automatically check the missing packages in requirements.txt and pyproject.toml.

[![PyPI - Version](https://img.shields.io/pypi/v/check-requirements-txt.svg)](https://pypi.org/project/check-requirements-txt)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/check-requirements-txt.svg)](https://pypi.org/project/check-requirements-txt)
[![CI](https://github.com/ferstar/check-requirements-txt/workflows/CI/badge.svg)](https://github.com/ferstar/check-requirements-txt/actions/workflows/ci.yml)
[![Lint](https://github.com/ferstar/check-requirements-txt/workflows/Lint/badge.svg)](https://github.com/ferstar/check-requirements-txt/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/ferstar/check-requirements-txt/branch/master/graph/badge.svg)](https://codecov.io/gh/ferstar/check-requirements-txt)

-----

**Table of Contents**

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Pre-commit Hook](#pre-commit-hook)
- [Output Sample](#output-sample)
- [License](#license)

## Features

- ✅ **Multiple dependency formats**: Supports both `requirements.txt` and `pyproject.toml`
- ✅ **Modern Python projects**: Full support for uv package manager and PEP 735 dependency groups
- ✅ **Comprehensive TOML parsing**:
  - `project.dependencies` - Main project dependencies
  - `project.optional-dependencies` - Optional dependencies/extras
  - `dependency-groups` - PEP 735 dependency groups with include-group support
  - `tool.uv.dev-dependencies` - Legacy uv development dependencies
- ✅ **Auto-detection**: Automatically finds pyproject.toml and requirements.txt files
- ✅ **Python 3.10-3.13**: Full compatibility across all modern Python versions
- ✅ **Pre-commit integration**: Works seamlessly as a pre-commit hook
- ✅ **Type-safe**: Fully typed with pyright/mypy support

## Installation

Install using pip:

```console
pip install check-requirements-txt
```

Or using uv:

```console
uv add --dev check-requirements-txt
```

## Usage

### Command Line

Check dependencies in your project:

```console
# Auto-detect pyproject.toml or requirements.txt files
check-requirements-txt src/

# Specify a requirements file
check-requirements-txt src/ -r requirements.txt

# Specify a pyproject.toml file
check-requirements-txt src/ -r pyproject.toml

# Check specific Python files
check-requirements-txt main.py utils.py -r pyproject.toml
```

### Supported File Formats

**pyproject.toml** (recommended for modern Python projects):
```toml
[project]
dependencies = ["requests>=2.25.0", "packaging"]

[project.optional-dependencies]
dev = ["pytest", "black"]

[dependency-groups]
lint = ["ruff", "mypy"]
test = ["coverage", "pytest-cov"]

[tool.uv]
dev-dependencies = ["pre-commit"]
```

**requirements.txt** (traditional format):
```
requests>=2.25.0
packaging
pytest  # dev dependency
```

## Pre-commit Hook

Set up pre-commit hooks for automatic checking:

See [pre-commit](https://github.com/pre-commit/pre-commit) for instructions

Sample `.pre-commit-config.yaml`:

> **Note**: Due to the pre-commit isolated environment, this package can't act as a normal git repo pre-commit hook.
> If your dependency files don't match the default patterns (`*requirement*.txt` or `pyproject.toml`), specify them explicitly.

```yaml
default_stages: [commit]

repos:
  - repo: local
    hooks:
      - id: check-requirements-txt
        name: check-requirements-txt
        description: Check missing packages in requirements.txt or pyproject.toml
        entry: check-requirements-txt
        args: ['--dst_dir', '.', '--ignore', 'pip,whatever,modules,you,want,to,ignore,with,comma,separated']
        language: python
        types: [python]
```

For more options, see `check-requirements-txt --help`.

## Output Sample

When missing dependencies are detected:

```shell
Bad import detected: "bs4", check your requirements.txt please.
/Users/ferstar/PycharmProjects/xxx_demo/xxx_spider.py:12
Bad import detected: "requests", check your requirements.txt please.
/Users/ferstar/PycharmProjects/xxx_demo/xxx_handler.py:17
"numpy" required by: {'numpy', 'scikit-learn', 'tensorflow', 'pandas'}

# Exit code indicates number of missing dependencies
$ echo $?
2
```

When using pyproject.toml:

```shell
# All dependencies found - no output, exit code 0
$ check-requirements-txt src/ -r pyproject.toml
$ echo $?
0

# Missing dependency detected
$ check-requirements-txt src/ -r pyproject.toml
Bad import detected: "missing_package", check your pyproject.toml please.
/path/to/file.py:5
$ echo $?
1
```

## License

`check-requirements-txt` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
