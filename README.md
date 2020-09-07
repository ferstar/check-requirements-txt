check-requirements-txt
==================

A tool (and pre-commit hook) to automatically check the missing packages in requirements.txt.

## Install as a pre-commit hook

See [pre-commit](https://github.com/pre-commit/pre-commit) for instructions

Sample `.pre-commit-config.yaml`:

```yaml
-   repo: https://github.com/ferstar/check-requirements-txt
    rev: v1.0.0
    hooks:
    -   id: check-requirements-txt
```
