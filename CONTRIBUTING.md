# Contributing to check-requirements-txt

Thank you for your interest in contributing to check-requirements-txt!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/ferstar/check-requirements-txt.git
cd check-requirements-txt
```

2. Install development dependencies using hatch:
```bash
pip install hatch
```

3. Run tests:
```bash
hatch run test
```

4. Run linting:
```bash
hatch run lint:all
```

## Testing

We use pytest for testing and support Python 3.10-3.13:

```bash
# Run tests on current Python version
hatch run test

# Run tests on all supported Python versions
hatch run all:test

# Run tests with coverage
hatch run test-cov
hatch run cov-report
```

### GitHub Actions Manual Testing

You can manually trigger CI workflows from the GitHub Actions tab:

1. **CI Workflow**: Test on all or specific Python versions
2. **Manual Test**: Advanced testing with custom options:
   - Select specific Python versions
   - Enable/disable linting
   - Enable/disable package building
   - Test package installation
3. **Lint Workflow**: Quick code quality checks with auto-fix option

## Code Quality

We use ruff for linting and formatting, and pyright for type checking:

```bash
# Check code style
hatch run lint:style

# Check types
hatch run lint:typing

# Run all linting checks
hatch run lint:all

# Auto-fix issues
hatch run lint:fmt
```

## Releasing

### For Maintainers

1. Use the release script:
```bash
./scripts/release.sh
```

2. Push changes and create a tag:
```bash
git push
git tag v1.3.0
git push origin v1.3.0
```

3. The GitHub Action will automatically publish to PyPI.

### PyPI Trusted Publishing Setup

This project uses PyPI's trusted publishing feature. To set it up:

1. Go to PyPI project settings
2. Add a new "trusted publisher"
3. Configure:
   - Repository: `ferstar/check-requirements-txt`
   - Workflow: `release.yml`
   - Environment: `pypi`

## Pull Request Guidelines

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for public functions
- Keep functions small and focused
- Add tests for new features

## Supported Features

When contributing, please ensure compatibility with:
- Python 3.10-3.13
- Both requirements.txt and pyproject.toml formats
- All dependency group types (project.dependencies, optional-dependencies, dependency-groups, tool.uv.dev-dependencies)
