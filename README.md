# check-requirements-txt

A tool (and also a pre-commit hook) to automatically check the missing packages in requirements.txt.

[![PyPI - Version](https://img.shields.io/pypi/v/check-requirements-txt.svg)](https://pypi.org/project/check-requirements-txt)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/check-requirements-txt.svg)](https://pypi.org/project/check-requirements-txt)

-----

**Table of Contents**

- [Installation](#Installation)
- [License](#License)

## Installation

First install this package into current python env

```console
pip install check-requirements-txt
```

Then set up `pre-commit` hooks

See [pre-commit](https://github.com/pre-commit/pre-commit) for instructions

Sample `.pre-commit-config.yaml`:

> NOTE: 
> 
> Due to the pre-commit isolated pyenv runtime, this package can't be act as a normal git repo pre-commit hooks.
> 
> If the project's requirements.txt does not match pattern `*requirement*.txt`, you'll need to specify it.

```yaml
default_stages: [commit]

repos:
  - repo: local
    hooks:
      - id: check-requirements-txt
        name: check-requirements-txt
        description: Check the missing packages in requirements.txt.
        entry: check-requirements-txt
        args: ['--dst_dir', '.', '--ignore', 'pip,whatever,modules,you,want,to,ignore,with,comma,separated']
        language: python
        types: [python]
```

`check-requirements-txt` can be used as a normal cli tool, see `check-requirements-txt --help` for more details.

## Output sample

```shell
Bad import detected: "bs4"
/Users/ferstar/PycharmProjects/xxx_demo/xxx_spider.py:12
Bad import detected: "requests"
/Users/ferstar/PycharmProjects/xxx_demo/xxx_handler.py:17
"numpy" required by: {'numpy', 'scikit-learn', 'tensorflow', 'pandas'}
# NOTE: the output of cli is the total bad import count
~ echo $?
~ 2
```

## License

`check-requirements-txt` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
