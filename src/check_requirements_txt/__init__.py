# SPDX-FileCopyrightText: 2023-present ferstar <zhangjianfei3@gmail.com>
#
# SPDX-License-Identifier: MIT
import argparse
import concurrent.futures
import importlib.metadata
import locale
import os
import re
import sys
from collections import defaultdict
from collections.abc import Generator, Iterable, Sequence
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]

from packaging.requirements import Requirement

MODULE_IMPORT_P = re.compile(r"^\s*?import\s+(?P<module>[a-zA-Z0-9_]+)")
MODULE_FROM_P = re.compile(r"^\s*?from\s+(?P<module>[a-zA-Z0-9_]+).*?\simport")
DROP_LINE_P = re.compile(r"^\w+:/+", re.I)

project_modules = set()


def supports_color() -> bool:
    """Check if the terminal supports color output."""
    # Check environment variables first
    if os.environ.get("NO_COLOR"):
        return False

    if os.environ.get("FORCE_COLOR"):
        return True

    # Check if output is redirected
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False

    # Check TERM environment variable
    term = os.environ.get("TERM", "") or ""
    if term.lower() in ("dumb", "unknown"):
        return False

    return True


def colorize(text: str, color_code: str) -> str:
    """Add color to text if terminal supports it."""
    if supports_color():
        return f"\033[{color_code}m{text}\033[0m"
    return text


def red(text: str) -> str:
    """Make text red if terminal supports color."""
    return colorize(text, "91")


def yellow(text: str) -> str:
    """Make text yellow if terminal supports color."""
    return colorize(text, "93")


def stdlibs() -> list[str]:
    return list(set(list(sys.stdlib_module_names) + list(sys.builtin_module_names)))


def find_depends(package_name: str, extras: set[str] | None = None) -> list[str]:
    """Find all dependencies for a package, optionally including extras.

    Args:
        package_name: The name of the package
        extras: Optional set of extras to include when resolving dependencies

    Returns:
        List of all required package names
    """
    requires = set()
    to_process = {package_name}
    processed = set()

    while to_process:
        current_package = to_process.pop()

        if current_package in processed:
            continue

        processed.add(current_package)

        try:
            dist = importlib.metadata.distribution(current_package)
        except importlib.metadata.PackageNotFoundError:
            # If the package itself is not found, add it to requirements.
            # This could happen if a requirement is specified but not installed.
            requires.add(current_package)
            continue

        requires.add(dist.metadata["Name"])  # Use normalized name

        if dist.requires:
            for req_str in dist.requires:
                req = Requirement(req_str)
                # Skip requirements with markers that don't evaluate to True
                if req.marker:
                    if extras:
                        # For each extra, check if this requirement should be included
                        should_include = False
                        for extra in extras:
                            env_with_extra = {"extra": extra}
                            if req.marker.evaluate(env_with_extra):
                                should_include = True
                                break
                        if not should_include:
                            # Also check if it evaluates to True without any extra
                            if not req.marker.evaluate({}):  # pragma: no branch
                                continue
                    # No extras specified, evaluate with empty environment
                    elif not req.marker.evaluate({}):  # pragma: no branch
                        continue
                to_process.add(req.name)
    return list(requires)


def find_real_modules(package_name: str) -> list[str]:
    modules = set()
    # Normalize package name for lookup
    normalized_package_name = package_name.replace("-", "_").lower()

    try:
        dist = importlib.metadata.distribution(package_name)
    except importlib.metadata.PackageNotFoundError:
        # If package not found, assume the package name is the module name
        modules.add(normalized_package_name)
        return list(modules)

    # Try to find top_level.txt
    if dist.files:
        for file_path in dist.files:
            if file_path.name == "top_level.txt":
                content = dist.read_text(file_path.name)
                if content:
                    for line in content.splitlines():
                        module_name = line.strip().lower()
                        if module_name:
                            modules.add(module_name)
                    # If top_level.txt is found and processed, we can often return early
                    if modules:  # pragma: no branch
                        return list(modules)

    # If top_level.txt is not found or empty, infer from file paths
    if dist.files:
        for file_path in dist.files:
            path_str = str(file_path)
            # Common patterns for module files or directories
            if path_str.endswith(".py") and "/" in path_str:  # part of a package
                module_name = path_str.split("/", 1)[0].lower()
                modules.add(module_name.replace("-", "_"))
            elif path_str.endswith(".py"):  # a top-level .py file
                module_name = file_path.name[:-3].lower()  # remove .py
                modules.add(module_name.replace("-", "_"))
            elif "__init__.py" in path_str:  # an __init__.py file indicates a package
                module_name = Path(path_str).parent.name.lower()
                modules.add(module_name.replace("-", "_"))

    # If no modules were found through file parsing, fall back to package name
    if not modules:
        modules.add(normalized_package_name)
    return list(modules)


def parse_pyproject_toml(path: Path) -> Iterable[tuple[str, set[str]]]:
    """Parse dependencies from pyproject.toml file.

    Supports:
    - project.dependencies
    - project.optional-dependencies
    - dependency-groups
    - tool.uv.dev-dependencies (legacy)
    """
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as e:
        print(f"Warning: Failed to parse {path}: {e}")
        return

    # Parse project.dependencies
    project = data.get("project", {})
    dependencies = project.get("dependencies", [])
    for dep in dependencies:
        try:
            req = Requirement(dep)
            package_name = req.name.lower().replace("_", "-")
            extras = {extra.lower().replace("_", "-") for extra in req.extras}
            yield (package_name, extras)
        except ValueError:
            # Invalid requirement, skip
            continue

    # Parse project.optional-dependencies
    optional_deps = project.get("optional-dependencies", {})
    for _extra_name, deps in optional_deps.items():
        for dep in deps:
            try:
                req = Requirement(dep)
                package_name = req.name.lower().replace("_", "-")
                extras = {extra.lower().replace("_", "-") for extra in req.extras}
                yield (package_name, extras)
            except ValueError:
                continue

    # Parse dependency-groups (PEP 735)
    dependency_groups = data.get("dependency-groups", {})
    for _group_name, deps in dependency_groups.items():
        for dep in deps:
            # Handle include-group syntax
            if isinstance(dep, dict) and "include-group" in dep:
                # Skip include-group entries, they reference other groups
                continue
            if isinstance(dep, str):
                try:
                    req = Requirement(dep)
                    package_name = req.name.lower().replace("_", "-")
                    extras = {extra.lower().replace("_", "-") for extra in req.extras}
                    yield (package_name, extras)
                except ValueError:
                    continue

    # Parse tool.uv.dev-dependencies (legacy)
    tool_uv = data.get("tool", {}).get("uv", {})
    dev_deps = tool_uv.get("dev-dependencies", [])
    for dep in dev_deps:
        try:
            req = Requirement(dep)
            package_name = req.name.lower().replace("_", "-")
            extras = {extra.lower().replace("_", "-") for extra in req.extras}
            yield (package_name, extras)
        except ValueError:
            continue


def parse_requirements(path: Path) -> Iterable[tuple[str, set[str]]]:
    """Parse requirements file and return tuples of (package_name, extras)."""
    system_encoding = locale.getpreferredencoding()
    supported_encodings = ["utf-8", "ISO-8859-1", "utf-16"]

    if system_encoding not in supported_encodings:
        supported_encodings.insert(1, system_encoding)

    last_error = None
    for encoding in supported_encodings:
        try:
            with open(path, encoding=encoding) as req_file:
                for raw_line in req_file:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("-r"):
                        # nested requirements path: "-r another-path.txt"
                        nested_path = Path(line.replace("-r", "", 1).split("#", 1)[0].strip())
                        if not nested_path.exists():
                            nested_path = path.parent.joinpath(nested_path)
                        yield from parse_requirements(nested_path)
                        continue
                    if line.startswith("-") or DROP_LINE_P.search(line):
                        continue
                    if line.startswith("git+https") and "#egg=" in line:
                        package_name = line.rsplit("#egg=", maxsplit=1)[-1].strip().lower().replace("_", "-")
                        yield (package_name, set())
                        continue

                    # Remove inline comments
                    clean_line = line.split("#", 1)[0].strip()
                    if not clean_line:  # pragma: no cover
                        continue

                    try:
                        req = Requirement(clean_line)
                        package_name = req.name.lower().replace("_", "-")  # Normalize to lowercase and hyphenated
                        extras = {extra.lower().replace("_", "-") for extra in req.extras}
                        yield (package_name, extras)
                    except ValueError:  # Catches RequirementParseError from packaging.requirements
                        # Invalid requirement line, skip
                        continue
            return
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue
        except Exception as e:
            raise e

    msg = f"Failed to decode {path} with any supported encoding: {supported_encodings}. Last error: {last_error}"
    encoding = "unknown"
    raise UnicodeDecodeError(encoding, b"", 0, 1, msg)


def load_req_modules(req_path: Path | str) -> dict[str, set[str]]:
    modules = defaultdict(set)
    if isinstance(req_path, str):
        req_path = Path(req_path)

    # Determine parser based on file extension
    if req_path.name.endswith((".toml", "pyproject.toml")):
        parser = parse_pyproject_toml
    else:
        parser = parse_requirements

    for package_name, extras in parser(req_path):
        for module in find_real_modules(package_name):
            modules[module].add(package_name)
        for pack in find_depends(package_name, extras):
            for mod in find_real_modules(pack):
                modules[mod].add(pack)  # Fixed: add the dependency package name, not the parent package
    return modules


def load_all_packages(req_path: Path | str, include_transitive: bool = False) -> set[str]:
    """Load all package names from requirements file.

    Args:
        req_path: Path to requirements file or pyproject.toml
        include_transitive: If True, include transitive dependencies. If False, only direct dependencies.

    Returns:
        Set of package names
    """
    packages = set()
    if isinstance(req_path, str):
        req_path = Path(req_path)

    # Determine parser based on file extension
    if req_path.name.endswith((".toml", "pyproject.toml")):
        parser = parse_pyproject_toml
    else:
        parser = parse_requirements

    for package_name, extras in parser(req_path):
        packages.add(package_name)
        # Optionally add all transitive dependencies
        if include_transitive:
            for dep in find_depends(package_name, extras):
                packages.add(dep)
    return packages


def load_package_dependencies(req_path: Path | str) -> dict[str, set[str]]:
    """Load package dependency mapping from requirements file.

    Returns:
        Dictionary mapping each directly declared package to its dependencies (including itself)
    """
    dependencies = {}
    if isinstance(req_path, str):
        req_path = Path(req_path)

    # Determine parser based on file extension
    if req_path.name.endswith((".toml", "pyproject.toml")):
        parser = parse_pyproject_toml
    else:
        parser = parse_requirements

    for package_name, extras in parser(req_path):
        # Find all dependencies for this package (including itself)
        deps = set(find_depends(package_name, extras))
        dependencies[package_name] = deps

    return dependencies


def get_imports_parallel(paths: Generator[Path, None, None] | list[Path], max_workers: int = 4) -> dict[str, set[str]]:
    """Get imports using parallel processing for better performance with many files."""
    modules: dict[str, set[str]] = defaultdict(set)

    def process_single_path(p: Path) -> dict[str, set[str]]:
        """Process a single path and return its imports."""
        path_modules: dict[str, set[str]] = defaultdict(set)

        def process_path_recursively(p: Path):
            if p.is_file() and p.suffix.lower() == ".py":
                try:
                    with open(p) as file_obj:
                        for idx, line in enumerate(file_obj, 1):
                            match = MODULE_IMPORT_P.search(line) or MODULE_FROM_P.search(line)
                            if not match:
                                continue
                            module = match.group("module").lower()
                            if module not in project_modules:
                                path_modules[module].add(f"{p}:{idx}")
                except (UnicodeDecodeError, OSError) as e:
                    # Skip files that can't be read, but log the error if verbose
                    import warnings
                    warnings.warn(f"Failed to read {p}: {e}", stacklevel=2)
            elif p.is_dir() and not p.name.startswith("."):
                for item in p.iterdir():
                    process_path_recursively(item)

        process_path_recursively(p)
        return path_modules

    # Collect all paths to process
    all_paths = list(paths)

    # Use ThreadPoolExecutor for I/O bound tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(process_single_path, path): path for path in all_paths}

        for future in concurrent.futures.as_completed(future_to_path):
            try:
                path_result = future.result()
                # Merge results
                for module, locations in path_result.items():
                    modules[module].update(locations)
            except Exception as e:
                # Handle any exceptions in individual file processing
                # Individual file errors shouldn't stop the entire process
                import warnings
                path = future_to_path[future]
                warnings.warn(f"Failed to process {path}: {e}", stacklevel=2)

    return modules


def get_imports(
    paths: Generator[Path, None, None] | list[Path],
    *,
    use_parallel: bool = False,
    max_workers: int = 4,
) -> dict[str, set[str]]:
    """Get imports from Python files."""
    if use_parallel:
        return get_imports_parallel(paths, max_workers)
    else:
        # Original sequential implementation
        modules: dict[str, set[str]] = defaultdict(set)

        def process_path(p: Path):
            if p.is_file() and p.suffix.lower() == ".py":
                with open(p) as file_obj:
                    for idx, line in enumerate(file_obj, 1):
                        match = MODULE_IMPORT_P.search(line) or MODULE_FROM_P.search(line)
                        if not match:
                            continue
                        module = match.group("module").lower()
                        if module not in project_modules:
                            modules[module].add(f"{p}:{idx}")
            elif p.is_dir() and not p.name.startswith("."):
                for item in p.iterdir():
                    process_path(item)

        for path in paths:
            process_path(path)
        return modules


def param_as_set(value: str) -> set[str]:
    return {v.strip() for v in value.split(",") if v}


def run(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    parser.add_argument(
        "-i",
        "--ignore",
        type=param_as_set,
        default="pip",
        help="ignore,modules,with,comma,separated",
    )
    parser.add_argument("-d", "--dst_dir", default="", help="destination directory you want to check")
    parser.add_argument(
        "-r",
        "--req-txt-path",
        dest="req_paths",
        type=param_as_set,
        default="",
        help="path of your requirements file or pyproject.toml (with comma separated)",
    )
    parser.add_argument(
        "--unused",
        action="store_true",
        help="check for unused dependencies in requirements files",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="use parallel processing for faster file scanning",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="maximum number of worker threads for parallel processing (default: 4)",
    )
    args = parser.parse_args(argv)

    # Validate max_workers parameter
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    if args.max_workers > 64:
        parser.error("--max-workers should not exceed 64")

    if not argv and not any([args.filenames, args.dst_dir, args.req_paths]):
        parser.print_help()
        sys.exit(0)

    builtin_modules: dict[str, set[str]] = defaultdict(set)
    builtin_modules.update(
        {i: set() for i in stdlibs()},
    )
    path_list = [Path(p).absolute() for p in args.filenames if p.lower().endswith(".py")]
    if args.dst_dir:
        path_list.append(Path(args.dst_dir).absolute())

    project_dirs = [p for p in path_list if p.is_dir() and not any(p.name.startswith(".") for p in p.parents)]
    if not project_dirs:
        project_dirs.append(Path().cwd())

    project_module_candidates: set[str] = set()
    for project in project_dirs:
        for py_path in project.glob("**/*.py"):
            if py_path.name.startswith("."):
                continue

            relative_path = py_path.relative_to(project)
            if any(part.startswith(".") for part in relative_path.parts):
                continue

            if py_path.name == "__init__.py":
                # Include package directories that only expose an __init__.py
                if not relative_path.parent.parts:
                    continue
                module_name = ".".join(relative_path.parent.parts)
            else:
                module_name = ".".join(relative_path.with_suffix("").parts)

            if module_name:  # pragma: no branch
                project_module_candidates.add(module_name)

    project_modules.update(project_module_candidates)
    project_modules.update(m for p in project_modules.copy() for m in p.split("."))

    if not args.req_paths:
        for project in project_dirs:
            # Look for pyproject.toml files first
            pyproject_files = list(project.glob("**/pyproject.toml"))
            for path in pyproject_files:
                args.req_paths.add(path)

            # Then look for requirements.txt files
            for path in project.glob("**/*requirement*.txt"):
                args.req_paths.add(path)

    if not args.req_paths:
        msg = (
            'No files matched pattern "*requirement*.txt" or "pyproject.toml", '
            "you need to specify the requirement(s) path(s)"
        )
        raise ValueError(msg)

    # Track the types of config files being used
    config_file_types = set()
    all_declared_packages = set()
    package_dependencies: dict[str, set[str]] = {}  # Maps directly declared package to its dependencies

    for path in args.req_paths:
        path_obj = Path(path)
        if path_obj.name.endswith((".toml", "pyproject.toml")):
            config_file_types.add("pyproject.toml")
        else:
            config_file_types.add("requirements.txt")

        # Load modules for checking missing imports
        for module, value in load_req_modules(path).items():
            builtin_modules[module].update(value)

        # Load all package names for unused dependency checking
        if args.unused:
            # Only load directly declared packages (not transitive dependencies)
            all_declared_packages.update(load_all_packages(path, include_transitive=False))
            # Load dependency mappings for each directly declared package
            package_dependencies.update(load_package_dependencies(path))

    # Generate appropriate config file suggestion
    if len(config_file_types) == 1:
        config_suggestion = next(iter(config_file_types))
    else:
        # Multiple config file types, suggest both
        config_suggestion = " or ".join(sorted(config_file_types))

    error_count = 0
    args.ignore.add("pip")
    args.ignore = {v.lower() for v in args.ignore}
    used_modules = get_imports(path_list, use_parallel=args.parallel, max_workers=args.max_workers)
    builtin_modules = {name.replace("-", "_"): items for name, items in builtin_modules.items()}
    used_modules = {name.replace("-", "_"): items for name, items in used_modules.items()}

    # Check for missing imports
    for module, paths in used_modules.items():
        if module in args.ignore:
            continue
        if module not in builtin_modules:
            print(
                red(f'Bad import detected: "{module}", check your {config_suggestion} please.'),
            )
            for _path in paths:
                print(_path)
            error_count += 1
        elif len(builtin_modules[module]) > 1:
            print(f'"{module}" required by: {builtin_modules[module]}')

    # Check for unused dependencies if requested
    if args.unused:
        # Find all used packages by checking which packages provide the modules we actually use
        used_packages = set()
        for module in used_modules:
            if module in builtin_modules:
                used_packages.update(builtin_modules[module])

        # For each directly declared package, check if it or any of its dependencies are used
        actually_used_packages = set()
        for direct_pkg in all_declared_packages:
            # Check if the package itself is used
            if direct_pkg in used_packages:
                actually_used_packages.add(direct_pkg)
            else:
                # Check if any of its dependencies are used
                deps = package_dependencies.get(direct_pkg, set())
                if any(dep in used_packages for dep in deps):
                    actually_used_packages.add(direct_pkg)

        # Find unused packages (only among directly declared packages)
        unused_packages = all_declared_packages - actually_used_packages
        unused_packages = {pkg for pkg in unused_packages if pkg not in args.ignore}

        if unused_packages:
            print(yellow(f"\nUnused dependencies found in {config_suggestion}:"))
            for package in sorted(unused_packages):
                print(f"  - {package}")
            error_count += len(unused_packages)

    return error_count


def main() -> None:
    """Main entry point for the application."""
    sys.exit(run())


if __name__ == "__main__":
    main()
