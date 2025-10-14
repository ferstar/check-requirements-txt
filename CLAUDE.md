# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`check-requirements-txt` is a Python tool that automatically checks for missing packages in `requirements.txt` and `pyproject.toml` files. It can be used as a standalone CLI tool or as a pre-commit hook.

## Development Commands

This project uses **Hatch** for dependency management and task running.

### Testing
```bash
# Run tests
hatch run test

# Run tests with coverage
hatch run cov

# Run tests with coverage and generate XML report
hatch run cov-xml

# Run specific test file
hatch run test tests/test_core.py::TestParseRequirements::test_basic_requirements

# Run tests with specific pattern
hatch run test -k "test_pyproject"
```

### Linting and Formatting
```bash
# Run all linting checks (ruff + pyright)
hatch run lint:all

# Run style checks only (ruff)
hatch run lint:style

# Run type checking only (pyright)
hatch run lint:typing

# Auto-format code
hatch run lint:fmt

# Fix style issues automatically
hatch run lint:fix
```

### Building
```bash
# Build distribution packages
hatch build

# Build and install in development mode
pip install -e .
```

### Running the Tool
```bash
# Run from source
python -m check_requirements_txt <path> [options]

# Or after installation
check-requirements-txt <path> [options]

# Common usage patterns
check-requirements-txt src/                    # Check directory
check-requirements-txt main.py               # Check specific files
check-requirements-txt -d .                # Check current directory
check-requirements-txt -r requirements.txt   # Specify requirements file
check-requirements-txt -r pyproject.toml   # Specify pyproject.toml
```

## Project Architecture

### Core Module: `src/check_requirements_txt/__init__.py`

The entire tool is implemented in a single module with the following key components:

**Main Functions:**
- `run(argv)` - Entry point that orchestrates the entire checking process
- `load_req_modules(req_path)` - Loads and resolves all packages from requirements files
- `get_imports(paths)` - Scans Python files to extract all import statements
- `find_real_modules(package_name)` - Maps package names to actual module names (e.g., `Pillow` → `PIL`)
- `find_depends(package_name, extras)` - Recursively resolves package dependencies

**Parsing Functions:**
- `parse_requirements(path)` - Parses traditional `requirements.txt` files
- `parse_pyproject_toml(path)` - Parses modern `pyproject.toml` files
  - Supports: `project.dependencies`, `project.optional-dependencies`, `dependency-groups` (PEP 735), `tool.uv.dev-dependencies`

**Module Discovery:**
The tool uses `importlib.metadata` to discover:
1. Top-level module names from `top_level.txt` metadata
2. Package files to infer module names when metadata is missing
3. Transitive dependencies with marker evaluation for extras

**Project Module Detection:**
The tool automatically identifies local project modules by:
- Scanning directories for Python files
- Building a set of project module names from relative paths
- Excluding these from missing package checks

**Color and Output:**
- `supports_color()` - Detects terminal color support
- `colorize()`, `red()`, `yellow()` - Colorized output functions
- `param_as_set()` - Utility for parsing comma-separated parameters

**Global State:**
- `project_modules` - Set of detected project modules (excluded from checks)
- `MODULE_IMPORT_P`, `MODULE_FROM_P` - Regex patterns for import detection
- `DROP_LINE_P` - Regex for skipping certain lines

### Test Suite: `tests/test_core.py`

Comprehensive test coverage including:
- Module import behavior (tomllib fallback)
- Requirements parsing (basic, extras, git URLs, nested files, encoding)
- Pyproject.toml parsing (all dependency sections, PEP 735 syntax)
- Color terminal support detection
- Import detection and project module filtering
- Integration tests with mocked importlib.metadata
- Complex extras parsing (uvicorn[standard], fastapi[all], etc.)
- Configuration auto-discovery and duplicate handling

## Python Version Support

Supports Python 3.10-3.14. Uses conditional imports for `tomllib` (Python 3.11+) with fallback to `tomli`.

## Key Implementation Details

1. **Package Name Normalization**: All package names are normalized to lowercase with hyphens (e.g., `my_package` → `my-package`)

2. **Module-to-Package Mapping**: The tool handles packages where the module name differs from the package name (e.g., `Pillow` package provides `PIL` module)

3. **Extras Support**: Full support for package extras (e.g., `coverage[toml]`) with marker evaluation using `packaging.requirements.Requirement`

4. **Nested Dependencies**: Recursively resolves all transitive dependencies to avoid false positives

5. **Configuration File Detection**: Auto-discovers `pyproject.toml` and `*requirement*.txt` files if not explicitly specified

6. **Project Module Filtering**: Automatically detects and excludes local project modules from missing package checks

7. **Error Handling**: Graceful handling of invalid requirements, encoding issues, and missing packages

8. **Color Output**: ANSI color support with fallback for non-color terminals

## CI/CD

- **CI** (`.github/workflows/ci.yml`): Tests on Python 3.10-3.14, runs coverage, linting, and builds packages
- **Lint** (`.github/workflows/lint.yml`): Automated linting with optional auto-fix on workflow dispatch
- **Manual Test** (`.github/workflows/manual-test.yml`): Manual testing workflow
- **Release** (`.github/workflows/release.yml`): Automated GitHub Releases

## Common Development Tasks

1. **Adding new parsing features**: Extend `parse_requirements()` or `parse_pyproject_toml()` functions
2. **Improving module discovery**: Enhance `find_real_modules()` or `find_depends()` functions
3. **Adding test coverage**: All new features should have comprehensive tests in `tests/test_core.py`
4. **Updating dependencies**: Modify `hatch.toml` for project dependencies
5. **Pre-commit integration**: The tool works as a pre-commit hook (see README for configuration)

## Important Notes

- The tool uses `packaging.requirements.Requirement` for parsing requirement strings
- Import detection uses regex patterns for `import` and `from ... import` statements
- Project modules are detected by scanning directory structure and file paths
- Error messages suggest the appropriate config file format (requirements.txt vs pyproject.toml)
- The `run()` function returns exit code equal to number of missing dependencies
- Tests extensively use mocking for `importlib.metadata` to avoid dependency on installed packages
