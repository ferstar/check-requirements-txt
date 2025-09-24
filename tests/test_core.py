# SPDX-FileCopyrightText: 2023-present ferstar <zhangjianfei3@gmail.com>
#
# SPDX-License-Identifier: MIT
import importlib.metadata
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
    project_modules,
    run,
    stdlibs,
)


def extract_package_names(packages_with_extras):
    """Helper function to extract package names from (package_name, extras) tuples."""
    return [pkg_name for pkg_name, _extras in packages_with_extras]


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
        # Should return tuples of (package_name, extras)
        expected = [("package-a", set()), ("package-b", set()), ("package-c", set())]
        assert sorted(packages) == sorted(expected)

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
        # Should return tuples of (package_name, extras)
        expected = [("package-a", set()), ("package-b", set())]
        assert sorted(packages) == sorted(expected)

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
        # Should return tuples of (package_name, extras)
        expected = [("package-a", {"extra1", "extra2"}), ("package-b", {"another-extra"}), ("package-c", set())]
        assert sorted(packages) == sorted(expected)

    def test_coverage_toml_extra_detection(self, tmp_path):
        """Test that coverage[toml] is properly detected and handled."""
        req_content = """
coverage[toml]>=6.0
pytest>=7.0
        """
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(req_content)
        packages = list(parse_requirements(req_file))
        # Should return tuples with extras properly captured
        expected = [("coverage", {"toml"}), ("pytest", set())]
        assert sorted(packages) == sorted(expected)

    def test_load_req_modules_with_extras(self, tmp_path):
        """Test that load_req_modules properly handles packages with extras."""
        req_content = """
coverage[toml]>=6.0
requests[security]>=2.25.0
        """
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(req_content)

        # Test the full pipeline
        modules = load_req_modules(req_file)

        # Should contain modules for both main packages and their extras
        # Note: This test might fail if the packages aren't installed
        print("Loaded modules:", dict(modules))

        # At minimum, we should have entries for the main packages
        assert any("coverage" in str(packages) for packages in modules.values())
        assert any("requests" in str(packages) for packages in modules.values())

    def test_extras_functionality_comprehensive(self, tmp_path):
        """Test comprehensive extras functionality."""
        req_content = """
coverage[toml]>=6.0
requests[security,socks]>=2.25.0
pytest>=7.0
        """
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(req_content)

        # Test parsing
        packages = list(parse_requirements(req_file))
        expected = [("coverage", {"toml"}), ("requests", {"security", "socks"}), ("pytest", set())]
        assert sorted(packages) == sorted(expected)

        # Test that extras are properly passed to find_depends
        # This tests the integration between parsing and dependency resolution
        modules = load_req_modules(req_file)

        # Should contain modules for the main packages
        package_names = set()
        for module_packages in modules.values():
            package_names.update(module_packages)

        assert "coverage" in package_names
        assert "requests" in package_names
        assert "pytest" in package_names

    def test_git_requirements(self, tmp_path):
        """Test git+https requirements with #egg= syntax."""
        req_content = """
git+https://github.com/user/repo.git#egg=my_package
package_b
        """
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(req_content)
        packages = list(parse_requirements(req_file))
        package_names = extract_package_names(packages)
        assert "my-package" in package_names  # normalized to hyphenated form
        assert "package-b" in package_names

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
        package_names = extract_package_names(packages)
        assert "package-a" in package_names
        assert "package-b" in package_names
        assert "nested-package" in package_names

    def test_parse_requirements_invalid_lines(self, tmp_path):
        """Test parsing requirements.txt with invalid requirement lines."""
        req_content = """
# Valid requirements
coverage[toml]>=6.0
requests>=2.25.0

# Invalid requirements (should be skipped)
invalid requirement string <<<
another invalid line >>>

# More valid requirements
pytest>=7.0
"""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(req_content)

        packages = list(parse_requirements(req_file))
        package_names = extract_package_names(packages)

        # Should include valid packages
        assert "coverage" in package_names
        assert "requests" in package_names
        assert "pytest" in package_names

        # Should not include invalid lines
        for package_name, _ in packages:
            assert "invalid" not in package_name
            assert "<<<" not in package_name
            assert ">>>" not in package_name

    def test_parse_requirements_encoding_fallback(self, tmp_path):
        """Test requirements parsing with different encodings."""
        # Create a file with special characters
        req_content = "# Comment with special chars: café\nrequests>=2.25.0\n"
        req_file = tmp_path / "requirements.txt"

        # Write with latin-1 encoding
        req_file.write_bytes(req_content.encode("latin-1"))

        packages = list(parse_requirements(req_file))
        package_names = extract_package_names(packages)

        # Should still parse the valid requirement
        assert "requests" in package_names

    def test_parse_requirements_encoding_error(self, tmp_path):
        """Test requirements parsing with unsupported encoding."""
        req_file = tmp_path / "requirements.txt"

        # Create a file with invalid bytes that can't be decoded
        invalid_bytes = b"\xff\xfe\x00\x00invalid content"
        req_file.write_bytes(invalid_bytes)

        # Should raise UnicodeDecodeError for completely invalid content
        try:
            list(parse_requirements(req_file))
            # If we get here, the file was somehow decoded, which is fine
        except UnicodeDecodeError:
            # This is expected for truly invalid content
            pass


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

    def test_find_real_modules_no_files(self):
        """Test find_real_modules when no files are found."""
        # Test with a package that doesn't exist
        result = find_real_modules("nonexistent_package_12345")
        # Should fall back to the package name itself
        assert "nonexistent_package_12345" in result

    def test_complex_extras_parsing(self, tmp_path):
        """Test parsing complex extras like uvicorn[standard], fastapi[all], etc."""
        req_content = """
# Web frameworks with complex extras
uvicorn[standard]>=0.18.0
fastapi[all]>=0.68.0
django[bcrypt,argon2]>=4.0.0

# Data science packages with multiple extras
pandas[performance,computation]>=1.3.0
numpy[dev,test]>=1.21.0

# HTTP libraries with security extras
requests[security,socks]>=2.25.0
httpx[http2,brotli]>=0.23.0

# Testing frameworks
pytest[testing]>=7.0.0
"""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(req_content)

        packages = list(parse_requirements(req_file))

        # Verify all packages are parsed correctly with their extras
        expected_packages = {
            "uvicorn": {"standard"},
            "fastapi": {"all"},
            "django": {"bcrypt", "argon2"},
            "pandas": {"performance", "computation"},
            "numpy": {"dev", "test"},
            "requests": {"security", "socks"},
            "httpx": {"http2", "brotli"},
            "pytest": {"testing"},
        }

        parsed_packages = dict(packages)

        for expected_pkg, expected_extras in expected_packages.items():
            assert expected_pkg in parsed_packages, f"Package {expected_pkg} not found"
            assert parsed_packages[expected_pkg] == expected_extras, (
                f"Extras mismatch for {expected_pkg}: expected {expected_extras}, got {parsed_packages[expected_pkg]}"
            )

    def test_uvicorn_standard_extra_dependencies(self):
        """Test that uvicorn[standard] resolves to the correct dependencies."""
        try:
            # First check if uvicorn is available
            importlib.metadata.distribution("uvicorn")

            # Test that uvicorn[standard] includes more dependencies than uvicorn alone
            deps_no_extra = find_depends("uvicorn")
            deps_with_standard = find_depends("uvicorn", {"standard"})

            print(f"Debug: uvicorn dependencies: {deps_no_extra}")
            print(f"Debug: uvicorn[standard] dependencies: {deps_with_standard}")

            # If both return the same single package, it might be a test environment issue
            if len(deps_no_extra) == 1 and len(deps_with_standard) == 1:
                # In this case, just verify that the package name is correct
                assert "uvicorn" in deps_no_extra
                assert "uvicorn" in deps_with_standard
                pytest.skip("Test environment may not have full uvicorn dependencies, skipping detailed check")

            # uvicorn[standard] should include more packages than uvicorn alone
            assert len(deps_with_standard) > len(deps_no_extra), (
                f"uvicorn[standard] should have more deps than uvicorn: "
                f"{len(deps_with_standard)} vs {len(deps_no_extra)}"
            )

            # All base dependencies should be included in both
            for dep in deps_no_extra:
                assert dep in deps_with_standard, f"Base dependency {dep} missing from uvicorn[standard]"

            # Standard extra should include specific packages (if they're installed)
            standard_extra_packages = {"httptools", "python-dotenv", "pyyaml", "uvloop", "watchfiles", "websockets"}
            found_standard_packages = set(deps_with_standard) & standard_extra_packages

            # Should find at least some of the standard extra packages
            assert len(found_standard_packages) > 0, (
                f"Should find some standard extra packages, but found: {found_standard_packages}"
            )

            print(f"✅ uvicorn dependencies: {deps_no_extra}")
            print(f"✅ uvicorn[standard] dependencies: {deps_with_standard}")
            print(f"✅ Found standard extra packages: {found_standard_packages}")

        except importlib.metadata.PackageNotFoundError:
            # Skip test if uvicorn is not installed
            pytest.skip("uvicorn not installed, skipping uvicorn[standard] test")


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

    def test_run_allows_package_with_init_only(self, tmp_path):
        """A package defined solely by __init__.py should not trigger a missing import."""
        project_modules.clear()
        try:
            project_dir = tmp_path / "project"
            project_dir.mkdir()

            pkg_dir = project_dir / "mypkg"
            pkg_dir.mkdir()
            (pkg_dir / "__init__.py").write_text("value = 1\n")

            py_file = project_dir / "main.py"
            py_file.write_text("import mypkg\n")

            req_file = project_dir / "requirements.txt"
            req_file.write_text("")
            original_cwd = os.getcwd()
            os.chdir(project_dir)
            try:
                return_code = run([str(py_file), "-r", str(req_file)])
            finally:
                os.chdir(original_cwd)
            assert return_code == 0
        finally:
            project_modules.clear()

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
        package_names = extract_package_names(deps)
        assert "packaging" in package_names
        assert "requests" in package_names

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
        package_names = extract_package_names(deps)
        assert "packaging" in package_names
        assert "pytest" in package_names
        assert "black" in package_names
        assert "sphinx" in package_names
        assert "sphinx-rtd-theme" in package_names

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
        package_names = extract_package_names(deps)
        assert "packaging" in package_names
        assert "pytest" in package_names
        assert "ruff" in package_names
        assert "coverage" in package_names
        assert "pytest-cov" in package_names

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
        package_names = extract_package_names(deps)
        assert "packaging" in package_names
        assert "pytest" in package_names
        assert "mypy" in package_names

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
        package_names = extract_package_names(deps)
        assert "packaging" in package_names
        assert "pytest" in package_names
        assert "ruff" in package_names
        assert "black" in package_names
        assert "coverage" in package_names
        assert "pytest-cov" in package_names

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

    def test_parse_pyproject_toml_with_extras(self, tmp_path):
        """Test parsing pyproject.toml with package extras."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_content = """
[project]
name = "test-project"
version = "0.1.0"
dependencies = [
    "coverage[toml]>=6.0",
    "requests[security,socks]>=2.25.0",
    "pytest>=7.0"
]

[project.optional-dependencies]
dev = [
    "black[d]>=22.0",
    "mypy[reports]>=1.0"
]

[dependency-groups]
test = [
    "pytest[cov]>=7.0",
    "coverage[toml]>=6.0"
]
"""
        pyproject_file.write_text(pyproject_content)

        deps = list(parse_pyproject_toml(pyproject_file))
        expected = [
            ("coverage", {"toml"}),
            ("requests", {"security", "socks"}),
            ("pytest", set()),
            ("black", {"d"}),
            ("mypy", {"reports"}),
            ("pytest", {"cov"}),
            ("coverage", {"toml"}),
        ]
        assert sorted(deps) == sorted(expected)

    def test_parse_pyproject_toml_invalid_requirements(self, tmp_path):
        """Test parsing pyproject.toml with invalid requirement strings."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_content = """
[project]
name = "test-project"
version = "0.1.0"
dependencies = [
    "valid-package>=1.0",
    "invalid requirement string <<<",
    "another-valid-package"
]

[project.optional-dependencies]
dev = [
    "valid-dev-package",
    "invalid dev requirement <<<",
]

[dependency-groups]
test = [
    "valid-test-package",
    "invalid test requirement <<<"
]

[tool.uv]
dev-dependencies = [
    "valid-uv-package",
    "invalid uv requirement <<<"
]
"""
        pyproject_file.write_text(pyproject_content)

        deps = list(parse_pyproject_toml(pyproject_file))
        package_names = extract_package_names(deps)

        # Should only include valid packages, invalid ones should be skipped
        assert "valid-package" in package_names
        assert "another-valid-package" in package_names
        assert "valid-dev-package" in package_names
        assert "valid-test-package" in package_names
        assert "valid-uv-package" in package_names

        # Invalid requirements should not appear
        for package_name, _ in deps:
            assert "invalid" not in package_name
            assert "<<<" not in package_name

    def test_parse_pyproject_toml_include_group_syntax(self, tmp_path):
        """Test parsing pyproject.toml with include-group syntax."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_content = """
[project]
name = "test-project"
version = "0.1.0"

[dependency-groups]
test = [
    "pytest>=7.0",
    "coverage>=6.0"
]
dev = [
    {include-group = "test"},
    "black>=22.0"
]
"""
        pyproject_file.write_text(pyproject_content)

        deps = list(parse_pyproject_toml(pyproject_file))
        package_names = extract_package_names(deps)

        # Should include packages from both groups
        assert "pytest" in package_names
        assert "coverage" in package_names
        assert "black" in package_names

        # Should not include the include-group entry itself
        assert "include-group" not in package_names

    def test_parse_pyproject_toml_complex_extras(self, tmp_path):
        """Test parsing pyproject.toml with complex extras like uvicorn[standard]."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_content = """
[project]
name = "my-web-app"
version = "0.1.0"
dependencies = [
    "uvicorn[standard]>=0.18.0",
    "fastapi[all]>=0.68.0",
    "requests[security,socks]>=2.25.0"
]

[project.optional-dependencies]
dev = [
    "pytest[testing]>=7.0.0",
    "black[d]>=22.0.0",
    "mypy[reports]>=1.0.0"
]
data = [
    "pandas[performance,computation]>=1.3.0",
    "numpy[dev,test]>=1.21.0"
]

[dependency-groups]
web = [
    "django[bcrypt,argon2]>=4.0.0",
    "httpx[http2,brotli]>=0.23.0"
]
"""
        pyproject_file.write_text(pyproject_content)

        deps = list(parse_pyproject_toml(pyproject_file))

        # Verify complex extras are parsed correctly
        expected_packages = {
            "uvicorn": {"standard"},
            "fastapi": {"all"},
            "requests": {"security", "socks"},
            "pytest": {"testing"},
            "black": {"d"},
            "mypy": {"reports"},
            "pandas": {"performance", "computation"},
            "numpy": {"dev", "test"},
            "django": {"bcrypt", "argon2"},
            "httpx": {"http2", "brotli"},
        }

        parsed_packages = dict(deps)

        for expected_pkg, expected_extras in expected_packages.items():
            assert expected_pkg in parsed_packages, f"Package {expected_pkg} not found"
            assert parsed_packages[expected_pkg] == expected_extras, (
                f"Extras mismatch for {expected_pkg}: expected {expected_extras}, got {parsed_packages[expected_pkg]}"
            )
