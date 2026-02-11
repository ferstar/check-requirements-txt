# SPDX-FileCopyrightText: 2023-present ferstar <zhangjianfei3@gmail.com>
#
# SPDX-License-Identifier: MIT
import argparse
import concurrent.futures
import fnmatch
import importlib.metadata
import locale
import os
import re
import sys
import warnings
from collections import defaultdict
from collections.abc import Generator, Iterable, Sequence
from functools import lru_cache
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[import-not-found]

from packaging.requirements import Requirement

# Constants
MAX_WORKERS_LIMIT = 64

MODULE_IMPORT_P = re.compile(r"^\s*?import\s+(?P<module>[a-zA-Z0-9_]+)")
MODULE_FROM_P = re.compile(r"^\s*?from\s+(?P<module>[a-zA-Z0-9_]+).*?\simport")
DROP_LINE_P = re.compile(r"^\w+:/+", re.I)

project_modules = set()


class GitignoreFilter:
    """A simple gitignore-style pattern matcher."""

    def __init__(self, gitignore_path: Path | None = None, *, respect_gitignore: bool = True):
        """Initialize the GitignoreFilter.

        Args:
            gitignore_path: Path to .gitignore file. If None, will search for .gitignore
                          in current working directory and parent directories.
            respect_gitignore: Whether to respect .gitignore patterns.
        """
        self.respect_gitignore = respect_gitignore
        self.patterns: list[tuple[str, bool, bool]] = []  # (pattern, is_negation, is_absolute)

        if not respect_gitignore:
            return

        if gitignore_path is None:
            gitignore_path = self._find_gitignore()

        if gitignore_path and gitignore_path.exists():
            self._load_patterns(gitignore_path)

    def _find_gitignore(self) -> Path | None:
        """Find .gitignore file in current directory or parent directories."""
        current = Path.cwd()
        while current.parent != current:  # Stop at filesystem root
            gitignore = current / ".gitignore"
            if gitignore.exists():
                return gitignore
            current = current.parent
        return None

    def _load_patterns(self, gitignore_path: Path) -> None:
        """Load patterns from .gitignore file."""
        try:
            with open(gitignore_path, encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue

                    is_negation = line.startswith("!")
                    if is_negation:
                        line = line[1:]

                    # Store the pattern with a flag indicating if it's absolute
                    is_absolute = line.startswith("/")
                    if is_absolute:
                        line = line[1:]  # Remove leading slash

                    # Handle directory patterns (ending with /)
                    if line.endswith("/"):
                        line = line[:-1]

                    # Store pattern with absolute flag
                    self.patterns.append((line, is_negation, is_absolute))
        except (OSError, UnicodeDecodeError):
            # Ignore errors reading .gitignore file
            pass

    def should_ignore(self, path: Path, base_path: Path | None = None) -> bool:
        """Check if a path should be ignored based on gitignore patterns.

        Args:
            path: The path to check
            base_path: Base path for relative pattern matching. If None, uses cwd.

        Returns:
            True if the path should be ignored, False otherwise.
        """
        if not self.respect_gitignore:
            return False

        if base_path is None:
            base_path = Path.cwd()

        try:
            # Get relative path for pattern matching
            if path.is_absolute():
                rel_path = path.relative_to(base_path)
            else:
                rel_path = path
        except ValueError:
            # Path is not relative to base_path
            return False

        rel_path_str = str(rel_path).replace("\\", "/")  # Normalize path separators

        ignored = False
        for pattern, is_negation, is_absolute in self.patterns:
            matches = False

            if is_absolute:
                # Absolute pattern - only match at root level
                # For absolute patterns, the path should match exactly at the root
                matches = (
                    rel_path_str == pattern  # Exact match
                    or (rel_path.name == pattern and len(rel_path.parts) == 1)  # Root level file
                )
            # Relative pattern - can match at any level
            elif "/" in pattern:
                # Pattern with path separators
                matches = fnmatch.fnmatch(rel_path_str, f"*/{pattern}")
            else:
                # Simple filename/dirname pattern
                # Check if any part of the path matches the pattern
                matches = (
                    fnmatch.fnmatch(rel_path_str, f"*/{pattern}")  # File/dir match
                    or fnmatch.fnmatch(rel_path.name, pattern)  # Final component match
                    or any(fnmatch.fnmatch(part, pattern) for part in rel_path.parts)  # Any path component
                    or fnmatch.fnmatch(rel_path_str, f"{pattern}/*")  # Inside directory
                    or rel_path_str.startswith(f"{pattern}/")  # Direct subdirectory
                )

            if matches:
                ignored = not is_negation

        return ignored


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


@lru_cache(maxsize=2048)
def _find_depends_cached(package_name: str, extras_key: tuple[str, ...]) -> tuple[str, ...]:
    extras = set(extras_key)

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
            requires.add(current_package)
            continue

        requires.add(dist.metadata["Name"])

        if dist.requires:
            for req_str in dist.requires:
                req = Requirement(req_str)
                if req.marker:
                    if extras:
                        should_include = False
                        for extra in extras:
                            env_with_extra = {"extra": extra}
                            if req.marker.evaluate(env_with_extra):
                                should_include = True
                                break
                        if not should_include:
                            if not req.marker.evaluate({}):
                                continue
                    elif not req.marker.evaluate({}):
                        continue
                to_process.add(req.name)
    return tuple(requires)


def find_depends(package_name: str, extras: set[str] | None = None) -> list[str]:
    """Find all dependencies for a package, optionally including extras.

    Args:
        package_name: The name of the package
        extras: Optional set of extras to include when resolving dependencies

    Returns:
        List of all required package names
    """
    extras_key = tuple(sorted(extra.lower().replace("_", "-") for extra in extras)) if extras else ()
    return list(_find_depends_cached(package_name, extras_key))


@lru_cache(maxsize=2048)
def _find_real_modules_cached(package_name: str) -> tuple[str, ...]:
    modules = set()
    # Normalize package name for lookup
    normalized_package_name = package_name.replace("-", "_").lower()

    try:
        dist = importlib.metadata.distribution(package_name)
    except importlib.metadata.PackageNotFoundError:
        # If package not found, assume the package name is the module name
        modules.add(normalized_package_name)
        return tuple(modules)

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
                        return tuple(modules)

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
    return tuple(modules)


def find_real_modules(package_name: str) -> list[str]:
    return list(_find_real_modules_cached(package_name))


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


def load_all_packages(req_path: Path | str, *, include_transitive: bool = False) -> set[str]:
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


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _module_name_for_file(py_path: Path, base_path: Path) -> str | None:
    try:
        relative_path = py_path.relative_to(base_path)
    except ValueError:
        return None

    if any(part.startswith(".") for part in relative_path.parts):
        return None

    if py_path.name == "__init__.py":
        if not relative_path.parent.parts:
            return None
        return ".".join(relative_path.parent.parts)
    return ".".join(relative_path.with_suffix("").parts)


def _collect_python_files_and_modules(
    paths: list[Path],
    *,
    module_scan_roots: list[Path] | None = None,
    gitignore_filter: GitignoreFilter | None = None,
) -> tuple[list[Path], set[str]]:
    import_dirs = [p for p in paths if p.is_dir()]
    import_files = {p for p in paths if p.is_file() and p.suffix.lower() == ".py"}
    scan_roots = module_scan_roots if module_scan_roots is not None else paths

    python_files_to_scan: list[Path] = []
    project_module_candidates: set[str] = set()
    seen_files: set[Path] = set()

    def should_scan_imports(py_path: Path) -> bool:
        if py_path in import_files:
            return True
        return any(_is_relative_to(py_path, import_dir) for import_dir in import_dirs)

    def process_py_file(py_path: Path, base_path: Path) -> None:
        if py_path in seen_files:
            return
        seen_files.add(py_path)

        module_name = _module_name_for_file(py_path, base_path)
        if module_name:
            project_module_candidates.add(module_name)

        if should_scan_imports(py_path):
            python_files_to_scan.append(py_path)

    for root in scan_roots:
        if not root.exists():
            continue

        base_path = root if root.is_dir() else root.parent
        if root.is_file():
            if root.suffix.lower() == ".py":
                process_py_file(root, base_path)
            continue

        stack = [root]
        while stack:
            current = stack.pop()

            if gitignore_filter and gitignore_filter.should_ignore(current, base_path):
                continue

            if current.is_dir():
                if current.name.startswith("."):
                    continue
                for item in current.iterdir():
                    stack.append(item)
                continue

            if current.is_file() and current.suffix.lower() == ".py":
                process_py_file(current, base_path)

    return python_files_to_scan, project_module_candidates


def get_imports_parallel(
    paths: Generator[Path, None, None] | list[Path],
    max_workers: int = 4,
    gitignore_filter: GitignoreFilter | None = None,
    project_module_names: set[str] | None = None,
    *,
    collect_project_modules: bool = False,
    module_scan_roots: list[Path] | None = None,
) -> dict[str, set[str]]:
    """Get imports using parallel processing for better performance with many files."""
    modules: dict[str, set[str]] = defaultdict(set)
    path_list = list(paths)

    if project_module_names is None:
        project_module_names = project_modules

    python_files_to_scan, project_module_candidates = _collect_python_files_and_modules(
        path_list,
        module_scan_roots=module_scan_roots,
        gitignore_filter=gitignore_filter,
    )

    if collect_project_modules:
        project_module_names.clear()
        project_module_names.update(project_module_candidates)
        project_module_names.update(m for p in project_module_names.copy() for m in p.split("."))

    def process_single_file(p: Path) -> dict[str, set[str]]:
        """Process a single file and return its imports."""
        path_modules: dict[str, set[str]] = defaultdict(set)
        try:
            with open(p) as file_obj:
                for idx, line in enumerate(file_obj, 1):
                    match = MODULE_IMPORT_P.search(line) or MODULE_FROM_P.search(line)
                    if not match:
                        continue
                    module = match.group("module").lower()
                    path_modules[module].add(f"{p}:{idx}")
        except (UnicodeDecodeError, OSError) as e:
            warnings.warn(f"Failed to read {p}: {e}", stacklevel=2)
        return path_modules

    # Use ThreadPoolExecutor for I/O bound tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(process_single_file, path): path for path in python_files_to_scan}

        for future in concurrent.futures.as_completed(future_to_path):
            try:
                path_result = future.result()
                # Merge results
                for module, locations in path_result.items():
                    modules[module].update(locations)
            except Exception as e:
                # Handle any exceptions in individual file processing
                # Individual file errors shouldn't stop the entire process
                path = future_to_path[future]
                warnings.warn(f"Failed to process {path}: {e}", stacklevel=2)

    return {module: locations for module, locations in modules.items() if module not in project_module_names}


def get_imports(
    paths: Generator[Path, None, None] | list[Path],
    *,
    use_parallel: bool = False,
    max_workers: int = 4,
    gitignore_filter: GitignoreFilter | None = None,
    project_module_names: set[str] | None = None,
    collect_project_modules: bool = False,
    module_scan_roots: list[Path] | None = None,
) -> dict[str, set[str]]:
    """Get imports from Python files."""
    path_list = list(paths)

    if project_module_names is None:
        project_module_names = project_modules

    if use_parallel:
        return get_imports_parallel(
            path_list,
            max_workers,
            gitignore_filter,
            project_module_names,
            collect_project_modules=collect_project_modules,
            module_scan_roots=module_scan_roots,
        )
    else:
        python_files_to_scan, project_module_candidates = _collect_python_files_and_modules(
            path_list,
            module_scan_roots=module_scan_roots,
            gitignore_filter=gitignore_filter,
        )

        if collect_project_modules:
            project_module_names.clear()
            project_module_names.update(project_module_candidates)
            project_module_names.update(m for p in project_module_names.copy() for m in p.split("."))

        modules: dict[str, set[str]] = defaultdict(set)
        for py_path in python_files_to_scan:
            try:
                with open(py_path) as file_obj:
                    for idx, line in enumerate(file_obj, 1):
                        match = MODULE_IMPORT_P.search(line) or MODULE_FROM_P.search(line)
                        if not match:
                            continue
                        module = match.group("module").lower()
                        modules[module].add(f"{py_path}:{idx}")
            except (UnicodeDecodeError, OSError) as e:
                warnings.warn(f"Failed to read {py_path}: {e}", stacklevel=2)

        return {module: locations for module, locations in modules.items() if module not in project_module_names}


def param_as_set(value: str) -> set[str]:
    return {v.strip() for v in value.split(",") if v.strip()}


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
    parser.add_argument(
        "--respect-gitignore",
        action="store_true",
        default=True,
        help="respect .gitignore patterns when scanning files (default: True)",
    )
    parser.add_argument(
        "--no-gitignore",
        action="store_true",
        help="ignore .gitignore patterns and scan all files",
    )
    parser.add_argument(
        "--gitignore-path",
        type=Path,
        help="path to custom .gitignore file (default: auto-detect)",
    )
    args = parser.parse_args(argv)

    # Validate max_workers parameter
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    if args.max_workers > MAX_WORKERS_LIMIT:
        parser.error(f"--max-workers should not exceed {MAX_WORKERS_LIMIT}")

    # Create gitignore filter
    respect_gitignore = args.respect_gitignore and not args.no_gitignore
    gitignore_filter = GitignoreFilter(
        gitignore_path=args.gitignore_path,
        respect_gitignore=respect_gitignore,
    )

    if not argv and not any([args.filenames, args.dst_dir, args.req_paths]):
        parser.print_help()
        sys.exit(0)

    builtin_modules: dict[str, set[str]] = defaultdict(set)
    builtin_modules.update(
        {i: set() for i in stdlibs()},
    )
    path_list = []
    for raw_path in args.filenames:
        path = Path(raw_path).absolute()
        if path.is_dir() or path.suffix.lower() == ".py":
            path_list.append(path)

    if args.dst_dir:
        path_list.append(Path(args.dst_dir).absolute())

    if not path_list:
        path_list.append(Path.cwd())

    project_dirs = [p for p in path_list if p.is_dir() and not any(p.name.startswith(".") for p in p.parents)]
    if not project_dirs:
        project_dirs.append(Path().cwd())

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
    used_modules = get_imports(
        path_list,
        use_parallel=args.parallel,
        max_workers=args.max_workers,
        gitignore_filter=gitignore_filter,
        project_module_names=project_modules,
        collect_project_modules=True,
        module_scan_roots=project_dirs,
    )
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
