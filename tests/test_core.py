# SPDX-FileCopyrightText: 2023-present ferstar <zhangjianfei3@gmail.com>
#
# SPDX-License-Identifier: MIT
import importlib
import importlib.metadata as importlib_metadata
import locale
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from check_requirements_txt import (
    colorize,
    find_depends,
    find_real_modules,
    get_imports,
    load_req_modules,
    main,
    param_as_set,
    parse_pyproject_toml,
    parse_requirements,
    project_modules,
    red,
    run,
    stdlibs,
    supports_color,
    yellow,
)


def extract_package_names(packages_with_extras):
    """Helper function to extract package names from (package_name, extras) tuples."""
    return [pkg_name for pkg_name, _extras in packages_with_extras]


class TestModuleImportBehavior:
    """Exercise module import fallback paths."""

    def test_tomllib_fallback_uses_tomli(self, monkeypatch):
        """Simulate ImportError for tomllib and ensure tomli fallback is used."""
        dummy_tomli = ModuleType("tomli")
        monkeypatch.setitem(sys.modules, "tomli", dummy_tomli)

        def remove_module(name: str) -> None:
            monkeypatch.setitem(sys.modules, name, None)
            monkeypatch.delitem(sys.modules, name, raising=False)

        remove_module("check_requirements_txt")

        original_import = __import__

        def fake_import(name, globalns=None, localns=None, fromlist=(), level=0):
            if name == "tomllib":
                message = "tomllib not available"
                raise ImportError(message)
            return original_import(name, globalns, localns, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            module = importlib.import_module("check_requirements_txt")

        assert module.tomllib is dummy_tomli

        monkeypatch.delitem(sys.modules, "tomli", raising=False)
        importlib.reload(module)


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

    def test_nested_requirements_relative_path(self, tmp_path):
        """Test that relative nested requirement paths are resolved."""
        nested_req = tmp_path / "nested.txt"
        nested_req.write_text("nested_package")

        req_file = tmp_path / "requirements.txt"
        req_file.write_text("""
-r nested.txt
package_a
        """)

        packages = list(parse_requirements(req_file))
        package_names = extract_package_names(packages)
        assert "nested-package" in package_names
        assert "package-a" in package_names

    def test_parse_requirements_skips_option_lines(self, tmp_path):
        """Ensure option-style requirement lines are ignored."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("""
--find-links https://example.com
package_a
        """)

        packages = list(parse_requirements(req_file))
        package_names = extract_package_names(packages)
        assert "package-a" in package_names
        assert all(not name.startswith("--") for name in package_names)

    def test_parse_requirements_adds_system_encoding(self, tmp_path, monkeypatch):
        """Ensure system encoding is inserted when not already listed."""
        monkeypatch.setattr(locale, "getpreferredencoding", lambda: "cp1252")

        req_file = tmp_path / "requirements.txt"
        req_file.write_text("package_a\n")

        packages = list(parse_requirements(req_file))
        assert extract_package_names(packages) == ["package-a"]

    def test_parse_requirements_existing_system_encoding(self, tmp_path, monkeypatch):
        """System encoding already present should not modify supported encodings."""
        monkeypatch.setattr(locale, "getpreferredencoding", lambda: "utf-8")

        req_file = tmp_path / "requirements.txt"
        req_file.write_text("package_a\n")

        packages = list(parse_requirements(req_file))
        assert extract_package_names(packages) == ["package-a"]

    def test_parse_requirements_all_encodings_fail(self, tmp_path, monkeypatch):
        """Ensure an informative UnicodeDecodeError is raised when decoding fails."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("package_a")

        def broken_open(*_args, **_kwargs):
            encoding = "broken"
            error_message = "boom"
            raise UnicodeDecodeError(encoding, b"", 0, 1, error_message)

        monkeypatch.setattr("builtins.open", broken_open)

        with pytest.raises(UnicodeDecodeError) as exc_info:
            list(parse_requirements(req_file))

        assert "Failed to decode" in str(exc_info.value)

    def test_parse_requirements_propagates_unexpected_error(self, tmp_path, monkeypatch):
        """Unexpected exceptions while reading should be propagated."""
        req_file = tmp_path / "requirements.txt"

        class UnexpectedError(Exception):
            pass

        def broken_open(*_args, **_kwargs):
            raise UnexpectedError

        monkeypatch.setattr("builtins.open", broken_open)

        with pytest.raises(UnexpectedError):
            list(parse_requirements(req_file))

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

    def test_parse_requirements_encoding_handling(self, tmp_path):
        """Test requirements parsing with different encodings and errors."""
        # Test 1: Latin-1 encoding with special characters
        req_content = "# Comment with special chars: café\nrequests>=2.25.0\n"
        req_file = tmp_path / "requirements_latin1.txt"
        req_file.write_bytes(req_content.encode("latin-1"))

        packages = list(parse_requirements(req_file))
        package_names = extract_package_names(packages)
        assert "requests" in package_names

        # Test 2: Invalid bytes that can't be decoded should not crash
        invalid_file = tmp_path / "requirements_invalid.txt"
        invalid_bytes = b"\xff\xfe\x00\x00invalid content"
        invalid_file.write_bytes(invalid_bytes)

        list(parse_requirements(invalid_file))


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


class TestColorFunctions:
    """Test color utility functions."""

    @patch("check_requirements_txt.sys.stdout")
    @patch("check_requirements_txt.os.environ")
    def test_supports_color_with_tty_and_no_env_vars(self, mock_environ, mock_stdout):
        """Test supports_color returns True when stdout is a TTY and no env vars are set."""
        mock_stdout.isatty.return_value = True
        mock_environ.get.side_effect = lambda _key, default="": default

        result = supports_color()
        assert result is True

    @patch("check_requirements_txt.sys.stdout")
    def test_supports_color_no_tty(self, mock_stdout):
        """Test supports_color returns False when stdout is not a TTY."""
        mock_stdout.isatty.return_value = False

        result = supports_color()
        assert result is False

    @patch("check_requirements_txt.sys.stdout")
    def test_supports_color_no_isatty_attribute(self, mock_stdout):
        """Test supports_color returns False when stdout has no isatty attribute."""
        del mock_stdout.isatty

        result = supports_color()
        assert result is False

    @patch("check_requirements_txt.sys.stdout")
    @patch("check_requirements_txt.os.environ")
    def test_supports_color_no_color_env_var(self, mock_environ, mock_stdout):
        """Test supports_color returns False when NO_COLOR is set."""
        mock_stdout.isatty.return_value = True
        mock_environ.get.side_effect = lambda key, default="": "1" if key == "NO_COLOR" else default

        result = supports_color()
        assert result is False

    @patch("check_requirements_txt.sys.stdout")
    @patch("check_requirements_txt.os.environ")
    def test_supports_color_force_color_env_var(self, mock_environ, mock_stdout):
        """Test supports_color returns True when FORCE_COLOR is set."""
        mock_stdout.isatty.return_value = False  # Even if not a TTY
        mock_environ.get.side_effect = lambda key, default="": "1" if key == "FORCE_COLOR" else default

        result = supports_color()
        assert result is True

    @pytest.mark.parametrize(
        "term_value,expected",
        [
            ("dumb", False),
            ("unknown", False),
            ("xterm", True),
            ("", True),
        ],
    )
    @patch("check_requirements_txt.sys.stdout")
    @patch("check_requirements_txt.os.environ")
    def test_supports_color_terminal_types(self, mock_environ, mock_stdout, term_value, expected):
        """Test supports_color with different TERM values."""
        mock_stdout.isatty.return_value = True
        mock_environ.get.side_effect = lambda key, default="": term_value if key == "TERM" else default

        result = supports_color()
        assert result is expected

    @patch("check_requirements_txt.supports_color")
    def test_colorize_with_color_support(self, mock_supports_color):
        """Test colorize adds ANSI codes when color is supported."""
        mock_supports_color.return_value = True

        result = colorize("test text", "91")
        assert result == "\033[91mtest text\033[0m"

    @patch("check_requirements_txt.supports_color")
    def test_colorize_without_color_support(self, mock_supports_color):
        """Test colorize returns plain text when color is not supported."""
        mock_supports_color.return_value = False

        result = colorize("test text", "91")
        assert result == "test text"

    @patch("check_requirements_txt.supports_color", return_value=True)
    def test_red_integration(self, _supports_color):
        """Test red adds ANSI codes when color is supported."""
        result = red("error message")
        assert result.startswith("\033[91m")
        assert result.endswith("\033[0m")
        assert "error message" in result

    @patch("check_requirements_txt.supports_color", return_value=False)
    def test_yellow_integration(self, _supports_color):
        """Test yellow returns plain text when color is not supported."""
        result = yellow("warning message")
        assert result == "warning message"


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
    def test_find_depends_handles_marker_extras_and_cycles(self, mock_distribution):
        """Ensure find_depends handles extras markers and avoids infinite loops."""

        class FakeDist:
            def __init__(self, name, requires):
                self.metadata = {"Name": name}
                self.requires = requires
                self.files = None

        def factory(package_name):
            if package_name == "root-package":
                return FakeDist("root-package", ['dep; extra == "feature"'])
            if package_name == "dep":
                return FakeDist("dep", ['root-package; extra == "feature"'])
            raise importlib_metadata.PackageNotFoundError(package_name)

        with pytest.raises(importlib_metadata.PackageNotFoundError):
            factory("missing")

        mock_distribution.side_effect = factory

        result = find_depends("root-package", {"feature"})
        assert set(result) == {"root-package", "dep"}

    @patch("check_requirements_txt.importlib.metadata.distribution")
    def test_find_depends_marker_branch_coverage(self, mock_distribution):
        """Cover marker evaluation fallbacks with extras and without extras."""

        class FakeDist:
            def __init__(self, name, requires):
                self.metadata = {"Name": name}
                self.requires = requires
                self.files = None

        def factory(name: str):
            if name == "branch-root":
                return FakeDist(
                    "branch-root",
                    [
                        'extra-match; extra == "feature"',
                        'extra-fallback; extra == "other" or python_version >= "3"',
                        'no-extra-keep; python_version < "4"',
                        'no-extra-skip; python_version < "3"',
                    ],
                )
            if name in {"extra-match", "extra-fallback", "no-extra-keep"}:
                return FakeDist(name, [])
            raise importlib_metadata.PackageNotFoundError(name)

        with pytest.raises(importlib_metadata.PackageNotFoundError):
            factory("unknown")

        mock_distribution.side_effect = factory

        result = find_depends("branch-root", {"feature"})

        assert "extra-match" in result
        assert "extra-fallback" in result
        assert "no-extra-keep" in result
        assert "no-extra-skip" not in result

    @patch("check_requirements_txt.importlib.metadata.distribution")
    def test_find_depends_marker_no_extras_branch(self, mock_distribution):
        """Exercise marker evaluation when no extras are provided."""

        class FakeDist:
            def __init__(self, name, requires):
                self.metadata = {"Name": name}
                self.requires = requires
                self.files = None

        def factory(name: str):
            if name == "root-noextras":
                return FakeDist(
                    "root-noextras",
                    [
                        'keep-me; python_version < "4"',
                        'skip-me; python_version < "3"',
                    ],
                )
            if name == "keep-me":
                return FakeDist("keep-me", [])
            raise importlib_metadata.PackageNotFoundError(name)

        with pytest.raises(importlib_metadata.PackageNotFoundError):
            factory("unknown")

        mock_distribution.side_effect = factory

        result = find_depends("root-noextras")

        assert "keep-me" in result
        assert "skip-me" not in result

    @patch("check_requirements_txt.importlib.metadata.distribution")
    def test_find_depends_skips_non_matching_extras(self, mock_distribution):
        """Requirements guarded by unmatched extras markers should be ignored."""

        class FakeDist:
            def __init__(self, name, requires):
                self.metadata = {"Name": name}
                self.requires = requires
                self.files = None

        def requires_extra(package_name: str):
            if package_name == "feature-root":
                return FakeDist("feature-root", ['dep-one; extra == "feature"'])
            raise importlib_metadata.PackageNotFoundError(package_name)

        with pytest.raises(importlib_metadata.PackageNotFoundError):
            requires_extra("other")

        mock_distribution.side_effect = requires_extra

        result = find_depends("feature-root", None)
        assert result == ["feature-root"]

        def requires_other_extra(package_name: str):
            if package_name == "feature-root":
                return FakeDist("feature-root", ['dep-two; extra == "other"'])
            raise importlib_metadata.PackageNotFoundError(package_name)

        with pytest.raises(importlib_metadata.PackageNotFoundError):
            requires_other_extra("other")

        mock_distribution.side_effect = requires_other_extra

        result = find_depends("feature-root", {"feature"})
        assert result == ["feature-root"]

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

    @patch("check_requirements_txt.importlib.metadata.distribution")
    def test_find_real_modules_path_inference_variants(self, mock_distribution):
        """Find modules from standalone files and __init__ sentinels."""

        class FakePath:
            def __init__(self, path):
                self._path = path
                self.name = path.split("/")[-1]

            def __str__(self):
                return self._path

        class FakeDist:
            def __init__(self):
                self.files = [FakePath("module.py"), FakePath("pkg/__init__.pyi")]

        mock_distribution.return_value = FakeDist()

        result = find_real_modules("example-package")
        assert "module" in result
        assert "pkg" in result

    @patch("check_requirements_txt.importlib.metadata.distribution")
    def test_find_real_modules_top_level_file(self, mock_distribution):
        """Handle top_level.txt metadata when present."""

        class FakePath:
            def __init__(self, name):
                self.name = name

        class FakeDist:
            def __init__(self) -> None:
                self.files = [FakePath("top_level.txt")]

            @staticmethod
            def read_text(_name: str) -> str:
                return "module_one\nmodule_two\n"

        mock_distribution.return_value = FakeDist()

        result = find_real_modules("package-with-top-level")
        assert set(result) == {"module_one", "module_two"}

    @patch("check_requirements_txt.importlib.metadata.distribution")
    def test_find_real_modules_top_level_blank_lines(self, mock_distribution):
        """Blank lines in top_level.txt should be ignored gracefully."""

        class FakePath:
            def __init__(self, name):
                self.name = name

        class FakeDist:
            def __init__(self) -> None:
                self.files = [FakePath("top_level.txt")]

            @staticmethod
            def read_text(_name: str) -> str:
                return "module_one\n\nmodule_two\n"

        mock_distribution.return_value = FakeDist()

        result = find_real_modules("package-with-blank-lines")
        assert set(result) == {"module_one", "module_two"}

    @patch("check_requirements_txt.importlib.metadata.distribution")
    def test_find_real_modules_top_level_empty(self, mock_distribution):
        """Empty top_level.txt should fall back to file inference."""

        class FakePath:
            def __init__(self, name, path_str):
                self.name = name
                self._path_str = path_str

            def __str__(self) -> str:
                return self._path_str

        class FakeDist:
            def __init__(self) -> None:
                self.files = [
                    FakePath("top_level.txt", "top_level.txt"),
                    FakePath("pkg/__init__.py", "pkg/__init__.py"),
                    FakePath("standalone.py", "standalone.py"),
                ]

            @staticmethod
            def read_text(_name: str) -> str:
                return ""

        mock_distribution.return_value = FakeDist()

        result = find_real_modules("package-empty-top-level")
        assert set(result) == {"pkg", "standalone"}

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


class TestEntrypoints:
    """Test the command entrypoints exposed by the package."""

    def test_main_exits_with_run_code(self):
        """Main should exit with the return code produced by run()."""
        with (
            patch("check_requirements_txt.run", return_value=0) as mock_run,
            patch("check_requirements_txt.sys.exit") as mock_exit,
        ):
            main()

        mock_run.assert_called_once_with()
        mock_exit.assert_called_once_with(0)

    def test_module_main_importable(self):
        """The __main__ module should be importable without side effects."""
        module_name = "check_requirements_txt.__main__"
        sys.modules[module_name] = ModuleType(module_name)
        del sys.modules[module_name]

        module = importlib.import_module(module_name)
        assert module is not None


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
        assert "requirements.txt" in captured.out

    @pytest.mark.parametrize(
        "color_support,should_have_ansi",
        [
            (True, True),
            (False, False),
        ],
    )
    @patch("check_requirements_txt.stdlibs")
    @patch("check_requirements_txt.load_req_modules")
    @patch("check_requirements_txt.get_imports")
    @patch("check_requirements_txt.supports_color")
    @patch("check_requirements_txt.project_modules", set())
    def test_run_missing_import_color_handling(
        self,
        mock_supports_color,
        mock_get_imports,
        mock_load_req,
        mock_stdlibs,
        tmp_path,
        capsys,
        color_support,
        should_have_ansi,
    ):
        """Test run with missing import handles color output correctly."""
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
        mock_supports_color.return_value = color_support

        return_code = run([str(project_dir), "--req-txt-path", str(req_file)])
        assert return_code == 1

        captured = capsys.readouterr()
        assert 'Bad import detected: "missing_module"' in captured.out
        assert "requirements.txt" in captured.out

        # Check ANSI codes based on color support
        if should_have_ansi:
            assert "\033[91m" in captured.out  # Red color code
            assert "\033[0m" in captured.out  # Reset color code
        else:
            assert "\033[91m" not in captured.out  # No red color code
            assert "\033[0m" not in captured.out  # No reset color code

    @patch("check_requirements_txt.stdlibs")
    @patch("check_requirements_txt.load_req_modules")
    @patch("check_requirements_txt.get_imports")
    @patch("check_requirements_txt.project_modules", set())
    def test_run_missing_import_with_pyproject_toml(
        self, mock_get_imports, mock_load_req, mock_stdlibs, tmp_path, capsys
    ):
        """Test run with missing import shows pyproject.toml in error message."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        py_file = project_dir / "main.py"
        py_file.write_text("import missing_module")

        pyproject_file = project_dir / "pyproject.toml"
        pyproject_file.write_text("""
[project]
name = "test-project"
dependencies = ["some_other_package"]
""")

        # Mock functions
        mock_stdlibs.return_value = ["sys", "os", "re"]
        mock_load_req.return_value = {"some_other_package": {"some_other_package"}}
        mock_get_imports.return_value = {"missing_module": {f"{py_file}:1"}}

        return_code = run([str(project_dir), "--req-txt-path", str(pyproject_file)])
        assert return_code == 1

        captured = capsys.readouterr()
        assert 'Bad import detected: "missing_module"' in captured.out
        assert "pyproject.toml" in captured.out

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

    @patch("check_requirements_txt.stdlibs")
    @patch("check_requirements_txt.load_req_modules")
    @patch("check_requirements_txt.get_imports")
    @patch("check_requirements_txt.project_modules", set())
    def test_run_auto_discovers_configs_and_handles_duplicates(
        self,
        mock_get_imports,
        mock_load_req,
        mock_stdlibs,
        tmp_path,
        capsys,
    ):
        """Auto-discover requirements/pyproject files and report shared modules."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Files to exercise directory scanning edge cases
        (project_dir / "__init__.py").write_text("# root package marker\n")
        (project_dir / ".hidden.py").write_text("import os\n")
        pkg_dir = project_dir / "pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("# pkg init\n")
        module_path = pkg_dir / "module.py"
        module_path.write_text("import shared_mod\nimport ignored_mod\n")

        # Create config files for autodiscovery
        pyproject = project_dir / "pyproject.toml"
        pyproject.write_text("""
[project]
name = "auto"
dependencies = ["shared_mod"]
""")
        requirements = project_dir / "requirements.txt"
        requirements.write_text("shared_mod\n")

        mock_stdlibs.return_value = ["sys"]

        def load_side_effect(path):
            path = Path(path)
            if path.name == "pyproject.toml":
                return {"shared_mod": {"pkg_pyproject"}}
            return {
                "shared_mod": {"pkg_requirements"},
                "ignored_mod": {"pkg_requirements"},
            }

        mock_load_req.side_effect = load_side_effect
        mock_get_imports.return_value = {
            "shared_mod": {f"{module_path}:1"},
            "ignored_mod": {f"{module_path}:2"},
        }

        result = run(["--dst_dir", str(project_dir), "--ignore", "ignored_mod"])
        assert result == 0

        captured = capsys.readouterr()
        assert '"shared_mod" required by:' in captured.out

        called_paths = {Path(call.args[0]).name for call in mock_load_req.call_args_list}
        assert {"pyproject.toml", "requirements.txt"} <= called_paths

    def test_run_without_arguments_displays_help(self, capsys):
        """Calling run with no arguments should print help and exit cleanly."""
        with pytest.raises(SystemExit) as exc_info:
            run([])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "usage" in captured.out.lower()


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

    def test_parse_pyproject_toml_dict_without_include(self, tmp_path):
        """Dict entries without include-group keys should be ignored safely."""
        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_content = """
[project]
name = "test-project"
version = "0.1.0"

[dependency-groups]
extras = [
    {optional = "value"},
    "requests"
]
"""
        pyproject_file.write_text(pyproject_content)

        deps = list(parse_pyproject_toml(pyproject_file))
        package_names = extract_package_names(deps)
        assert "requests" in package_names

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
