# SPDX-FileCopyrightText: 2023-present ferstar <zhangjianfei3@gmail.com>
#
# SPDX-License-Identifier: MIT
import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Sequence, Set, Union

import pkg_resources

MODULE_IMPORT_P = re.compile(r"^\s*?import\s+(?P<module>[a-zA-Z0-9_]+)")
MODULE_FROM_P = re.compile(r"^\s*?from\s+(?P<module>[a-zA-Z0-9_]+).*?\simport")
DROP_LINE_P = re.compile(r"^\w+:/+", re.I)
project_modules = set()


def stdlibs() -> List[str]:
    ver = sys.version_info
    if ver < (3, 10):
        from stdlib_list import stdlib_list
        return stdlib_list(f"{ver.major}.{ver.minor}")
    else:
        return list(set(list(sys.stdlib_module_names) + list(sys.builtin_module_names)))


def find_depends(package_name: str) -> List[str]:
    requires = set()
    to_process = [package_name]
    processed = set()

    while to_process:
        current_package = to_process.pop(0)

        if current_package in processed:
            continue

        processed.add(current_package)

        try:
            dist_obj = pkg_resources.get_distribution(current_package)
        except pkg_resources.DistributionNotFound:
            requires.add(current_package)
            continue

        requires.add(current_package)

        for req in dist_obj.requires():
            if req.marker and not req.marker.evaluate():
                continue
            to_process.append(req.name)

    return list(requires)


def find_real_modules(package_name: str) -> List[str]:
    modules = set()
    to_process = [package_name]
    processed = set()

    while to_process:
        current_package = to_process.pop(0)

        if current_package in processed:
            continue

        processed.add(current_package)

        try:
            egg_dir = Path(pkg_resources.get_distribution(current_package).egg_info)
        except pkg_resources.DistributionNotFound:
            modules.add(current_package)
            continue

        top_level_file = egg_dir / "top_level.txt"
        if top_level_file.exists() and top_level_file.is_file():
            with open(top_level_file) as file_obj:
                for line in file_obj:
                    modules.add(line.strip().lower())

        # Some packages do not have "top_level.txt", such as "attrs".
        # We can use "RECORD" file to find the possible modules.
        record = egg_dir / "RECORD"
        if record.exists() and record.is_file():
            with open(record) as file_obj:
                for line in file_obj:
                    path = line.split(",", 1)[0].strip()
                    if egg_dir.name in path:
                        continue
                    if "__init__." in path:
                        modules.add(Path(path).parent.name.lower())

    if not modules:
        modules.add(package_name)
    return list(modules)


def parse_requirements(path: Path) -> Iterable[str]:
    import locale

    system_encoding = locale.getpreferredencoding()
    supported_encodings = ['utf-8', 'ISO-8859-1', 'utf-16']

    if system_encoding not in supported_encodings:
        supported_encodings.insert(1, system_encoding)

    last_error = None
    for encoding in supported_encodings:
        try:
            with open(path, encoding=encoding) as req_file:
                for line in req_file:
                    if line.startswith("-r"):
                        # nested requirements path: "-r another-path.txt"
                        nested_path = Path(line.replace("-r", "", 1).split("#", 1)[0].strip())
                        if not nested_path.exists():
                            nested_path = path.parent.joinpath(nested_path)
                        yield from parse_requirements(nested_path)
                    if line.startswith("-") or DROP_LINE_P.search(line):
                        continue
                    if line.startswith("git+https") and "#egg=" in line:
                        yield line.rsplit("#egg=", maxsplit=1)[-1].strip()
                        continue
                    for req in pkg_resources.parse_requirements(line):
                        yield req.key
                        for ext in req.extras:
                            yield ext.lower()
            return
        except (UnicodeDecodeError, UnicodeError) as e:
            last_error = e
            continue
        except Exception as e:
            raise e

    raise UnicodeDecodeError(
        f"Failed to decode {path} with any supported encoding: {supported_encodings}. "
        f"Last error: {last_error}"
    )


def load_req_modules(req_path: Union[Path, str]) -> Dict[str, Set[str]]:
    modules = defaultdict(set)
    if isinstance(req_path, str):
        req_path = Path(req_path)
    for package in parse_requirements(req_path):
        for module in find_real_modules(package):
            modules[module].add(package)
        for pack in find_depends(package):
            for mod in find_real_modules(pack):
                modules[mod].add(package)
    return modules


def get_imports(
        paths: Union[Generator[Path, None, None], List[Path]]
) -> Dict[str, Set[str]]:
    modules: Dict[str, Set[str]] = defaultdict(set)

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


def param_as_set(value: str) -> Set[str]:
    return {v.strip() for v in value.split(",") if v}


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    parser.add_argument(
        "-i",
        "--ignore",
        type=param_as_set,
        default="pip",
        help="ignore,modules,with,comma,separated",
    )
    parser.add_argument(
        "-d", "--dst_dir", default="", help="destination directory you want to check"
    )
    parser.add_argument(
        "-r",
        "--req-txt-path",
        dest="req_paths",
        type=param_as_set,
        default="",
        help="path of your requirements file(with comma separated)",
    )
    args = parser.parse_args(argv)
    if not argv and len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    builtin_modules: Dict[str, Set[str]] = defaultdict(set)
    builtin_modules.update(
        {
            i: set()
            for i in stdlibs()
        },
    )
    path_list = [
        Path(p).absolute() for p in args.filenames if p.lower().endswith(".py")
    ]
    if args.dst_dir:
        path_list.append(Path(args.dst_dir).absolute())

    project_dirs = [
        p
        for p in path_list
        if p.is_dir() and not any(p.name.startswith(".") for p in p.parents)
    ]
    if not project_dirs:
        project_dirs.append(Path().cwd())

    project_modules.update(
        os.path.splitext(p.as_posix().replace(project.as_posix(), "").lstrip("/"))[
            0
        ].replace("/", ".")
        for project in project_dirs
        for p in project.glob("**/*.py")
        if not p.name.startswith(".")
        and p.name != "__init__.py"
        and (os.path.isdir(p) or p.name.endswith(".py"))
    )
    project_modules.update(m for p in project_modules.copy() for m in p.split("."))

    if not args.req_paths:
        for project in project_dirs:
            for path in args.req_paths or project.glob("**/*requirement*.txt"):
                args.req_paths.add(path)
    if not args.req_paths:
        msg = 'No files matched pattern "*requirement*.txt", you need to specify the requirement(s) path(s)'
        raise ValueError(msg)

    for path in args.req_paths:
        for module, value in load_req_modules(path).items():
            builtin_modules[module].update(value)

    error_count = 0
    args.ignore.add("pip")
    args.ignore = {v.lower() for v in args.ignore}
    used_modules = get_imports(path_list)
    builtin_modules = {name.replace("-", "_"): items for name, items in builtin_modules.items()}
    used_modules = {name.replace("-", "_"): items for name, items in used_modules.items()}
    for module, paths in used_modules.items():
        if module in args.ignore:
            continue
        if module not in builtin_modules:
            print(
                f'Bad import detected: "{module}", check your requirements.txt please.',
            )
            for _path in paths:
                print(_path)
            error_count += 1
        elif len(builtin_modules[module]) > 1:
            print(f'"{module}" required by: {builtin_modules[module]}')
    return error_count


if __name__ == "__main__":
    sys.exit(run())
