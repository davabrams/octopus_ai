load("@bazel_tools//tools/python:toolchain.bzl", "py_runtime_pair")

# Pin Bazel's Python to the project venv so `bazel run` / `bazel test` work
# without first activating .venv. Bazel does not manage pip deps here (see
# CLAUDE.md); tensorflow/matplotlib/flask/etc. live only in .venv, and the
# default autodetecting toolchain would otherwise grab the system python3.
#
# interpreter_path must be an absolute path. It is machine-specific (so is the
# venv itself, which is gitignored), pointing at this checkout's .venv.
py_runtime(
    name = "venv_py3_runtime",
    interpreter_path = "/Users/davabrams/Desktop/__Projects/Project - Octo/octopus_ai/.venv/bin/python3",
    python_version = "PY3",
)

py_runtime_pair(
    name = "venv_runtime_pair",
    py3_runtime = ":venv_py3_runtime",
)

toolchain(
    name = "venv_python_toolchain",
    toolchain = ":venv_runtime_pair",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
)
