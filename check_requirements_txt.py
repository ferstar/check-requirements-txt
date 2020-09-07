import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, List, Iterable
from typing import Sequence
from typing import Set

from pip._vendor import pkg_resources
from stdlib_list import stdlib_list

module_import_p = re.compile(r'^\s*?import\s+(?P<module>[a-zA-Z0-9_]+)')
module_from_p = re.compile(r'^\s*?from\s+(?P<module>[a-zA-Z0-9_]+).*?\simport')
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
        with open(top_level_file, 'r') as file_obj:
            for line in file_obj:
                real_modules.append(line.strip().lower())
        return real_modules
    return [package_name]


def parse_requirements(path: str) -> Iterable[str]:
    with open(path, 'r') as req_file:
        for line in req_file:
            if line.startswith('-'):
                continue
            for req in pkg_resources.parse_requirements(line):
                yield req.key
                for ext in req.extras:
                    yield ext.lower()


def load_req_modules(req_path: str) -> List[str]:
    modules = []
    for package in parse_requirements(req_path):
        for module in find_real_modules(package):
            modules.append(module)
        for pack in find_depends(package):
            modules.extend(find_real_modules(pack))
    return modules


def builtin_packages(path: str) -> Dict[str, set]:
    packages = defaultdict(set)
    with open(path, 'r') as req_file:
        for line in req_file:
            packages[line.strip()] = set()
    return packages


def get_all_imports(path: str) -> Dict[str, set]:
    modules = defaultdict(set)
    with open(path, 'r') as file_obj:
        for idx, line in enumerate(file_obj, 1):
            match = module_import_p.search(line) or module_from_p.search(line)
            if not match:
                continue
            module = match.group('module').lower()
            if module not in sub_dirs:
                modules[module].add(f'{path}:{idx}')
    return modules


def parse_ignore(value: str) -> Set[str]:
    return set(v.strip() for v in value.split(',') if v)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='*')
    parser.add_argument('--ignore', type=parse_ignore, default='pip')
    args = parser.parse_args(argv)

    modules = stdlib_list(f'{sys.version_info.major}.{sys.version_info.minor}')
    for path in project_dir.glob('**/*requirements*.txt'):
        modules.extend(load_req_modules(path))

    error_count = 0
    for filename in args.filenames:
        for module, paths in get_all_imports(filename).items():
            if module not in modules and module not in args.ignore:
                print(f'Bad import detected: "{module}", check your requirements.txt please.')
                for path in paths:
                    print(path)
                error_count += 1

    return error_count


if __name__ == '__main__':
    exit(main())
