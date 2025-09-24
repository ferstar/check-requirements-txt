#!/bin/bash
# Release script for check-requirements-txt

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're on the master branch
current_branch=$(git branch --show-current)
if [ "$current_branch" != "master" ]; then
    print_error "Must be on master branch to release. Current branch: $current_branch"
    exit 1
fi

# Check if working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    print_error "Working directory is not clean. Please commit or stash changes."
    exit 1
fi

# Get current version
current_version=$(hatch version)
print_status "Current version: $current_version"

# Ask for new version
echo -n "Enter new version (or press Enter to keep current): "
read new_version

if [ -n "$new_version" ]; then
    print_status "Updating version to $new_version"
    hatch version "$new_version"
    
    # Commit version change
    git add src/check_requirements_txt/__about__.py
    git commit -m "chore: bump version to $new_version"
    
    current_version="$new_version"
fi

# Run tests
print_status "Running tests..."
hatch run test

# Run linting
print_status "Running linting..."
hatch run lint:all

# Build package
print_status "Building package..."
hatch build --clean

print_status "Package built successfully!"
print_status "To release:"
print_status "1. Push changes: git push"
print_status "2. Create and push tag: git tag v$current_version && git push origin v$current_version"
print_status "3. Or use GitHub's manual publish workflow"

print_warning "Make sure to set up PyPI trusted publishing in GitHub repository settings!"
