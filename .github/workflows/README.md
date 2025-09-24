# GitHub Actions Workflows

This directory contains the CI/CD workflows for check-requirements-txt.

## Workflows

### 1. CI (`ci.yml`)
**Triggers**: Push to master/main, Pull requests, Manual dispatch
- Tests on Python 3.10-3.13
- Runs linting checks
- Manual trigger allows selecting specific Python versions

### 2. Release (`release.yml`)
**Triggers**: Git tags starting with `v*`
- Runs full test suite
- Builds package
- Publishes to PyPI using trusted publishing

### 3. Manual Publish (`publish.yml`)
**Triggers**: Manual dispatch only
- Allows manual PyPI publishing
- Optional version update
- Full test suite before publishing

### 4. Manual Test (`manual-test.yml`)
**Triggers**: Manual dispatch only
- Advanced testing options
- Custom Python version selection
- Optional package installation testing
- Configurable linting and building

### 5. Lint (`lint.yml`)
**Triggers**: Push/PR with Python files, Manual dispatch
- Quick code quality checks
- Optional auto-fix mode
- Automatic commit of fixes (manual trigger only)

## Manual Triggers

All workflows support manual triggering from the GitHub Actions tab:

1. Go to Actions tab in GitHub repository
2. Select the workflow you want to run
3. Click "Run workflow"
4. Configure options as needed
5. Click "Run workflow" button

## PyPI Publishing Setup

To enable automatic PyPI publishing:

1. Go to PyPI project settings
2. Add trusted publisher:
   - Repository: `ferstar/check-requirements-txt`
   - Workflow: `release.yml`
   - Environment: `pypi`

## Release Process

1. Use the release script: `./scripts/release.sh`
2. Push changes: `git push`
3. Create and push tag: `git tag v1.3.0 && git push origin v1.3.0`
4. GitHub Actions will automatically publish to PyPI
