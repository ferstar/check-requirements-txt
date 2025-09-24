# SPDX-FileCopyrightText: 2023-present ferstar <zhangjianfei3@gmail.com>
#
# SPDX-License-Identifier: MIT
import os
from unittest.mock import MagicMock, patch

import pytest

from check_requirements_txt import (
    find_depends,
    find_real_modules,
    get_imports,
    load_req_modules,
    param_as_set,
    parse_pyproject_toml,
    parse_requirements,
    run,
    stdlibs,
)


class TestParseRequirements:
    """Test the parse_requirements function."""

    def test_basic_requirements(self, tmp_path):
        """Test parsing basic package names."""
        req_content = """
package_a
package_b==1.0
package_c>=2.0,<3.0
        """
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(req_content)
        packages = list(parse_requirements(req_file))
        # pkg_resources normalizes package names to lowercase with hyphens
        assert sorted(packages) == sorted(["package-a", "package-b", "package-c"])

    def test_requirements_with_comments(self, tmp_path):
        """Test requirements file with comments and empty lines."""
        req_content = """
# This is a comment
package_a

package_b # inline comment
        """
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(req_content)
        packages = list(parse_requirements(req_file))
        # pkg_resources normalizes package names to lowercase with hyphens
        assert sorted(packages) == sorted(["package-a", "package-b"])

    def test_requirements_with_extras(self, tmp_path):
        """Test requirements with extras."""
        req_content = """
package_a[extra1,extra2]
package_b[another_extra]
package_c
        """
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(req_content)
        packages = list(parse_requirements(req_file))
        # The current implementation yields extras as separate items
        # importlib.metadata normalizes package names to lowercase with hyphens
        expected = ["package-a", "extra1", "extra2", "package-b", "another-extra", "package-c"]
        assert sorted(packages) == sorted(expected)

    def test_git_requirements(self, tmp_path):
        """Test git+https requirements with #egg= syntax."""
        req_content = """
git+https://github.com/user/repo.git#egg=my_package
package_b
        """
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(req_content)
        packages = list(parse_requirements(req_file))
        assert "my-package" in packages  # normalized to hyphenated form
        assert "package-b" in packages

    def test_nested_requirements(self, tmp_path):
        """Test -r nested requirements files."""
        # Create nested requirements file
        nested_req = tmp_path / "nested.txt"
        nested_req.write_text("nested_package")

        # Main requirements file
        req_content = f"""
package_a
-r {nested_req}
package_b
        """
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(req_content)
        packages = list(parse_requirements(req_file))
        assert "package-a" in packages
        assert "package-b" in packages
        assert "nested-package" in packages


class TestUtilityFunctions:
    """Test utility functions."""

    def test_param_as_set(self):
        """Test param_as_set function."""
        assert param_as_set("a,b,c") == {"a", "b", "c"}
        assert param_as_set("a, b , c ") == {"a", "b", "c"}
        assert param_as_set("") == set()
        assert param_as_set("single") == {"single"}

    def test_stdlibs(self):
        """Test stdlibs function returns standard library modules."""
        stdlib_modules = stdlibs()
        assert isinstance(stdlib_modules, list)
        assert "sys" in stdlib_modules
        assert "os" in stdlib_modules
        assert "re" in stdlib_modules


class TestGetImports:
    """Test the get_imports function."""

    def test_get_imports_basic(self, tmp_path):
        """Test basic import detection."""
        py_file = tmp_path / "test.py"
        py_file.write_text("""
import requests
from pathlib import Path
import os
        """)

        with patch("check_requirements_txt.project_modules", set()):
            imports = get_imports([tmp_path])

        assert "requests" in imports
        assert "pathlib" in imports
        assert "os" in imports

    def test_get_imports_ignores_project_modules(self, tmp_path):
        """Test that project modules are ignored."""
        py_file = tmp_path / "test.py"
        py_file.write_text("""
import requests
import my_project_module
        """)

        with patch("check_requirements_txt.project_modules", {"my_project_module"}):
            imports = get_imports([tmp_path])

        assert "requests" in imports
        assert "my_project_module" not in imports

    def test_get_imports_nested_directories(self, tmp_path):
        """Test scanning nested directories."""
        level1 = tmp_path / "level1"
        level1.mkdir()
        file_l1 = level1 / "l1_file.py"
        file_l1.write_text("import mod_l1")

        level2 = level1 / "level2"
        level2.mkdir()
        file_l2 = level2 / "l2_file.py"
        file_l2.write_text("import mod_l2")

        with patch("check_requirements_txt.project_modules", set()):
            imports = get_imports([tmp_path])

        assert "mod_l1" in imports
        assert "mod_l2" in imports

    def test_get_imports_skips_dot_directories(self, tmp_path):
        """Test that directories starting with . are skipped."""
        dot_dir = tmp_path / ".hidden"
        dot_dir.mkdir()
        hidden_file = dot_dir / "hidden.py"
        hidden_file.write_text("import hidden_module")

        regular_file = tmp_path / "regular.py"
        regular_file.write_text("import regular_module")

        with patch("check_requirements_txt.project_modules", set()):
            imports = get_imports([tmp_path])

        assert "regular_module" in imports
        assert "hidden_module" not in imports


class TestMockedFunctions:
    """Test functions that require mocking pkg_resources."""

    @patch("check_requirements_txt.importlib.metadata.distribution")
    def test_find_depends_basic(self, mock_distribution):
        """Test find_depends with mocked importlib.metadata."""
        # Mock distribution object
        mock_dist = MagicMock()
        mock_dist.metadata = {"Name": "test_package"}
        mock_dist.requires = []
        mock_distribution.return_value = mock_dist

        result = find_depends("test_package")
        assert "test_package" in result

    @patch("check_requirements_txt.importlib.metadata.distribution")
    def test_find_real_modules_basic(self, mock_distribution):
        """Test find_real_modules with mocked importlib.metadata."""
        # Mock distribution object
        mock_dist = MagicMock()
        mock_dist.files = None
        mock_distribution.return_value = mock_dist

        result = find_real_modules("test_package")
        assert "test_package" in result


class TestRunFunction:
    """Test the main run function."""

    @patch("check_requirements_txt.stdlibs")
    @patch("check_requirements_txt.load_req_modules")
    @patch("check_requirements_txt.get_imports")
    @patch("check_requirements_txt.project_modules", set())
    def test_run_successful_check(self, mock_get_imports, mock_load_req, mock_stdlibs, tmp_path):
        """Test a successful run where all imports are in requirements."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        py_file = project_dir / "main.py"
        py_file.write_text("import requests")

        req_file = project_dir / "requirements.txt"
        req_file.write_text("requests")

        # Mock functions
        mock_stdlibs.return_value = ["sys", "os", "re"]
        mock_load_req.return_value = {"requests": {"requests"}}
        mock_get_imports.return_value = {"requests": {f"{py_file}:1"}}

        return_code = run([str(project_dir), "--req-txt-path", str(req_file)])
        assert return_code == 0

    @patch("check_requirements_txt.stdlibs")
    @patch("check_requirements_txt.load_req_modules")
    @patch("check_requirements_txt.get_imports")
    @patch("check_requirements_txt.project_modules", set())
    def test_run_missing_import(self, mock_get_imports, mock_load_req, mock_stdlibs, tmp_path, capsys):
        """Test run with missing import."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        py_file = project_dir / "main.py"
        py_file.write_text("import missing_module")

        req_file = project_dir / "requirements.txt"
        req_file.write_text("some_other_package")

        # Mock functions
        mock_stdlibs.return_value = ["sys", "os", "re"]
        mock_load_req.return_value = {"some_other_package": {"some_other_package"}}
        mock_get_imports.return_value = {"missing_module": {f"{py_file}:1"}}

        return_code = run([str(project_dir), "--req-txt-path", str(req_file)])
        assert return_code == 1

        captured = capsys.readouterr()
        assert 'Bad import detected: "missing_module"' in captured.out

    def test_run_no_requirements_file(self, tmp_path):
        """Test run when no requirements file is found."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        py_file = project_dir / "main.py"
        py_file.write_text("import some_module")

        # Change to the project directory to avoid finding requirements.txt in the repo root
        original_cwd = os.getcwd()
        try:
            os.chdir(str(project_dir))
            with pytest.raises(ValueError, match="No files matched pattern"):
                run([str(project_dir)])
        finally:
            os.chdir(original_cwd)


class TestPyprojectTomlSupport:
    """Test pyproject.toml parsing and support."""

    def test_parse_pyproject_toml_basic(self, tmp_path):
        """Test parsing basic pyproject.toml dependencies."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_content = """
[project]
name = "test-project"
version = "0.1.0"
dependencies = [
    "packaging>=21.0",
    "requests[security]>=2.25.0",
]
"""
        pyproject_file.write_text(pyproject_content)

        deps = list(parse_pyproject_toml(pyproject_file))
        assert "packaging" in deps
        assert "requests" in deps

    def test_parse_pyproject_toml_optional_dependencies(self, tmp_path):
        """Test parsing optional dependencies from pyproject.toml."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_content = """
[project]
name = "test-project"
version = "0.1.0"
dependencies = ["packaging"]

[project.optional-dependencies]
dev = ["pytest>=6.0", "black"]
docs = ["sphinx", "sphinx-rtd-theme"]
"""
        pyproject_file.write_text(pyproject_content)

        deps = list(parse_pyproject_toml(pyproject_file))
        assert "packaging" in deps
        assert "pytest" in deps
        assert "black" in deps
        assert "sphinx" in deps
        assert "sphinx-rtd-theme" in deps

    def test_parse_pyproject_toml_dependency_groups(self, tmp_path):
        """Test parsing dependency groups from pyproject.toml."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_content = """
[project]
name = "test-project"
version = "0.1.0"
dependencies = ["packaging"]

[dependency-groups]
dev = ["pytest>=6.0", "ruff"]
test = ["coverage", "pytest-cov"]
"""
        pyproject_file.write_text(pyproject_content)

        deps = list(parse_pyproject_toml(pyproject_file))
        assert "packaging" in deps
        assert "pytest" in deps
        assert "ruff" in deps
        assert "coverage" in deps
        assert "pytest-cov" in deps

    def test_parse_pyproject_toml_uv_dev_dependencies(self, tmp_path):
        """Test parsing legacy uv dev-dependencies from pyproject.toml."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_content = """
[project]
name = "test-project"
version = "0.1.0"
dependencies = ["packaging"]

[tool.uv]
dev-dependencies = ["pytest>=6.0", "mypy"]
"""
        pyproject_file.write_text(pyproject_content)

        deps = list(parse_pyproject_toml(pyproject_file))
        assert "packaging" in deps
        assert "pytest" in deps
        assert "mypy" in deps

    def test_load_req_modules_with_pyproject_toml(self, tmp_path):
        """Test load_req_modules with pyproject.toml file."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_content = """
[project]
name = "test-project"
version = "0.1.0"
dependencies = ["packaging"]
"""
        pyproject_file.write_text(pyproject_content)

        modules = load_req_modules(pyproject_file)
        assert "packaging" in modules

    def test_run_with_pyproject_toml(self, tmp_path):
        """Test running check with pyproject.toml file."""
        # Create a pyproject.toml file
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_content = """
[project]
name = "test-project"
version = "0.1.0"
dependencies = ["packaging"]
"""
        pyproject_file.write_text(pyproject_content)

        # Create a simple Python file that imports packaging
        py_file = tmp_path / "test.py"
        py_file.write_text("import packaging\n")

        # Run the check - should pass since packaging is in pyproject.toml
        result = run([str(py_file), "-r", str(pyproject_file)])
        assert result == 0

    def test_parse_pyproject_toml_include_groups(self, tmp_path):
        """Test parsing dependency groups with include-group syntax."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_content = """
[project]
name = "test-project"
version = "0.1.0"
dependencies = ["packaging"]

[dependency-groups]
dev = [
    {include-group = "lint"},
    {include-group = "test"},
    "pytest>=6.0"
]
lint = ["ruff", "black"]
test = ["coverage", "pytest-cov"]
"""
        pyproject_file.write_text(pyproject_content)

        deps = list(parse_pyproject_toml(pyproject_file))
        assert "packaging" in deps
        assert "pytest" in deps
        assert "ruff" in deps
        assert "black" in deps
        assert "coverage" in deps
        assert "pytest-cov" in deps

    def test_parse_pyproject_toml_invalid_file(self, tmp_path, capsys):
        """Test parsing invalid pyproject.toml file."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_content = """
[project
name = "test-project"  # Missing closing bracket
"""
        pyproject_file.write_text(pyproject_content)

        deps = list(parse_pyproject_toml(pyproject_file))
        assert len(deps) == 0

        captured = capsys.readouterr()
        assert "Warning: Failed to parse" in captured.out
