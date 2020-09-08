check-requirements-txt
==================

A tool (and pre-commit hook) to automatically check the missing packages in requirements.txt.

## Install
First install this package into current python env

`python setup.py install`

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
