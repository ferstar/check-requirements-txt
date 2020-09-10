check-requirements-txt
==================

A tool (and also a pre-commit hook) to automatically check the missing packages in requirements.txt.

## Install
First install this package into current python env

`pip install check-requirements-txt`

Then set up `pre-commit` hooks

See [pre-commit](https://github.com/pre-commit/pre-commit) for instructions

Sample `.pre-commit-config.yaml`:

> NOTE: Due to the pre-commit isolated pyenv runtime, this package can't be act as a normal git repo pre-commit hooks

```yaml
default_stages: [commit]

repos:
  - repo: local
    hooks:
      - id: check-requirements-txt
        name: check-requirements-txt
        description: Check the missing packages in requirements.txt.
        entry: check-requirements-txt
        args: ['--ignore', 'pip,']
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
```
