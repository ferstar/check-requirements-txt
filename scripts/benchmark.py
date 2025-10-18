import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# Common libraries for generating import statements
LIBRARIES = [
    "requests", "numpy", "pandas", "os", "sys", "json", "datetime", "re",
    "collections", "math", "random", "time", "urllib.request", "subprocess",
    "pathlib", "logging", "threading", "multiprocessing", "argparse", "csv"
]

BENCHMARK_DIR = Path("benchmark_src")


def generate_files(num_files: int):
    """Generate a specified number of Python files for benchmarking."""
    if BENCHMARK_DIR.exists():
        # Clean up previous benchmark files
        for f in BENCHMARK_DIR.glob("**/*"):
            f.unlink()
    else:
        BENCHMARK_DIR.mkdir()

    for i in range(num_files):
        filename = BENCHMARK_DIR / f"file_{i}.py"
        with open(filename, "w") as f:
            num_imports = random.randint(1, 10)
            imports = random.sample(LIBRARIES, num_imports)
            for lib in imports:
                f.write(f"import {lib}\\n")
            f.write("\\n\\ndef main():\\n")
            f.write("    pass\\n")

    # Create a dummy pyproject.toml
    with open(BENCHMARK_DIR / "pyproject.toml", "w") as f:
        f.write("[project]\\n")
        f.write('name = "benchmark-project"\\n')
        f.write('version = "0.1.0"\\n')
        f.write('dependencies = ["requests", "numpy", "pandas"]\\n')

    print(f"Generated {num_files} Python files in {BENCHMARK_DIR}/")


def run_benchmark(num_files: int):
    """Run the benchmark and print the results."""
    generate_files(num_files)

    # Install the tool in the current environment
    print("Installing check-requirements-txt...")
    result = subprocess.run(["uv", "pip", "install", "."], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error installing check-requirements-txt:")
        print(result.stderr)
        sys.exit(1)

    # --- Run in sequential mode ---
    print("\\nRunning benchmark in sequential mode...")
    start_time = time.perf_counter()
    subprocess.run(
        [
            "check-requirements-txt",
            str(BENCHMARK_DIR),
            "-r",
            str(BENCHMARK_DIR / "pyproject.toml"),
        ],
        check=True,
        capture_output=True,
    )
    sequential_time = time.perf_counter() - start_time

    # --- Run in parallel mode ---
    print("Running benchmark in parallel mode...")
    start_time = time.perf_counter()
    subprocess.run(
        [
            "check-requirements-txt",
            str(BENCHMARK_DIR),
            "-r",
            str(BENCHMARK_DIR / "pyproject.toml"),
            "--parallel",
        ],
        check=True,
        capture_output=True,
    )
    parallel_time = time.perf_counter() - start_time

    # --- Print results ---
    print("\\n--- Benchmark Results ---")
    print(f"Number of files: {num_files}")
    print(f"Sequential mode: {sequential_time:.4f} seconds")
    print(f"Parallel mode:   {parallel_time:.4f} seconds")

    if parallel_time < sequential_time:
        improvement = (sequential_time - parallel_time) / sequential_time * 100
        print(f"\\n🚀 Improvement of {improvement:.2f}% with --parallel flag!")
    else:
        slowdown = (parallel_time - sequential_time) / sequential_time * 100
        print(f"\\n⚠️ Parallel mode was {slowdown:.2f}% slower.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark for check-requirements-txt.")
    parser.add_argument(
        "--num-files",
        type=int,
        default=5000,
        help="Number of Python files to generate for the benchmark.",
    )
    args = parser.parse_args()

    run_benchmark(args.num_files)