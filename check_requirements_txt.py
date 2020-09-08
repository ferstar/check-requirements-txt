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

from pip._vendor import pkg_resources
from stdlib_list import stdlib_list

MODULE_IMPORT_P = re.compile(r'^\s*?import\s+(?P<module>[a-zA-Z0-9_]+)')
MODULE_FROM_P = re.compile(r'^\s*?from\s+(?P<module>[a-zA-Z0-9_]+).*?\simport')
project_dir = Path().cwd()
sub_dirs = [d for d in os.listdir(project_dir) if not d.startswith('.')]


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
    top_level_file = Path(metadata_dir) / 'top_level.txt'
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
            if line.startswith('-'):
                continue
            for req in pkg_resources.parse_requirements(line):
                yield req.key
                for ext in req.extras:
                    yield ext.lower()


def load_req_modules(req_path: Path) -> Dict[str, Set[str]]:
    modules = defaultdict(set)
    for package in parse_requirements(req_path):
        for module in find_real_modules(package):
            modules[module].add(package)
        for pack in find_depends(package):
            for mod in find_real_modules(pack):
                modules[mod].add(package)
    return modules


def builtin_packages(path: str) -> Dict[str, Set[str]]:
    packages: Dict[str, Set[str]] = defaultdict(set)
    with open(path) as req_file:
        for line in req_file:
            packages[line.strip()] = set()
    return packages


def get_imports(paths: Union[Generator[Path, None, None], List[str]]) -> Dict[str, Set[str]]:
    modules: Dict[str, Set[str]] = defaultdict(set)
    for path in [Path(p) if isinstance(p, str) else p for p in paths]:
        if path.name.startswith('.'):
            continue
        if path.is_file():
            with open(path) as file_obj:
                for idx, line in enumerate(file_obj, 1):
                    match = MODULE_IMPORT_P.search(line) or MODULE_FROM_P.search(line)
                    if not match:
                        continue
                    module = match.group('module').lower()
                    if module not in sub_dirs:
                        modules[module].add(f'{path}:{idx}')
        elif path.is_dir():
            for module, files in get_imports(path.glob('**/*.py')).items():
                modules[module].update(files)
    return modules


def parse_ignore(value: str) -> Set[str]:
    return {v.strip() for v in value.split(',') if v}.union({'pip'})


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='*')
    parser.add_argument('--ignore', type=parse_ignore, default='pip', help='ignore some modules')
    parser.add_argument('--dst_dir', default='', help='destination directory you want to check(absolute path needed)')
    args = parser.parse_args(argv)

    modules: Dict[str, Set[str]] = defaultdict(set)
    modules.update(
        {
            i: set()
            for i in stdlib_list(f'{sys.version_info.major}.{sys.version_info.minor}')
        },
    )
    for path in project_dir.glob('**/*requirements*.txt'):
        for module, value in load_req_modules(path).items():
            modules[module].update(value)

    file_list = args.filenames
    if args.dst_dir and Path(args.dst_dir).is_dir():
        file_list = [args.dst_dir]

    error_count = 0
    for module, paths in get_imports(file_list).items():
        if module in args.ignore:
            continue
        if module not in modules:
            print(
                f'Bad import detected: "{module}", check your requirements.txt please.',
            )
            for _path in paths:
                print(_path)
            error_count += 1
        elif len(modules[module]) > 1:
            print(f'"{module}" required by: {modules[module]}')
    return error_count


if __name__ == '__main__':
    exit(main())
