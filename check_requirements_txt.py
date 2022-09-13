import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Union

import pkg_resources
from stdlib_list import stdlib_list

MODULE_IMPORT_P = re.compile(r"^\s*?import\s+(?P<module>[a-zA-Z0-9_]+)")
MODULE_FROM_P = re.compile(r"^\s*?from\s+(?P<module>[a-zA-Z0-9_]+).*?\simport")
DROP_LINE_P = re.compile(r"^\w+:/+", re.I)
project_modules = set()


def find_depends(package_name: str) -> List[str]:
    package = pkg_resources.working_set.by_key.get(package_name)
    if not package:
        return [package_name]
    return [r.key for r in package.requires()]


def find_real_modules(package_name: str) -> List[str]:
    try:
        metadata_dir = pkg_resources.get_distribution(package_name).egg_info
    except pkg_resources.DistributionNotFound:
        return [package_name]
    top_level_file = Path(metadata_dir) / "top_level.txt"
    if top_level_file.exists() and top_level_file.is_file():
        real_modules = []
        with open(top_level_file) as file_obj:
            for line in file_obj:
                real_modules.append(line.strip().lower())
        return real_modules
    return [package_name]


def parse_requirements(path: Path) -> Iterable[str]:
    with open(path) as req_file:
        for line in req_file:
            if line.startswith("-r"):
                # nested requirements path: "-r another-path.txt"
                nested_path = Path(line.replace("-r", "", 1).split("#", 1)[0].strip())
                if not nested_path.exists():
                    nested_path = path.parent / nested_path.name
                yield from parse_requirements(nested_path)
            if line.startswith("-") or DROP_LINE_P.search(line):
                continue
            for req in pkg_resources.parse_requirements(line):
                yield req.key
                for ext in req.extras:
                    yield ext.lower()


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
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".py":
            with open(path) as file_obj:
                for idx, line in enumerate(file_obj, 1):
                    match = MODULE_IMPORT_P.search(line) or MODULE_FROM_P.search(line)
                    if not match:
                        continue
                    module = match.group("module").lower()
                    if module not in project_modules:
                        modules[module].add(f"{path}:{idx}")
        elif path.is_dir():
            for module, files in get_imports(path.glob("**/*.py")).items():
                modules[module].update(files)
    return modules


def param_as_set(value: str) -> Set[str]:
    return {v.strip() for v in value.split(",") if v}


def main(argv: Optional[Sequence[str]] = None) -> int:
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
            for i in stdlib_list(f"{sys.version_info.major}.{sys.version_info.minor}")
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
    assert (
        args.req_paths
    ), 'No files matched pattern "*requirement*.txt", you need to specify the requirement(s) path(s)'

    for path in args.req_paths:
        for module, value in load_req_modules(path).items():
            builtin_modules[module].update(value)

    error_count = 0
    args.ignore.add("pip")
    for module, paths in get_imports(path_list).items():
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
    sys.exit(main())
