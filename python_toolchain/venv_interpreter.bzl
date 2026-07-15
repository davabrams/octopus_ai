"""Repository rule that pins Bazel's Python to the project venv.

Bazel does not manage pip deps here (see CLAUDE.md); tensorflow, matplotlib,
flask, etc. live only in .venv. The default autodetecting toolchain would grab
the system python3 and fail with ModuleNotFoundError unless .venv was activated
first.

`py_runtime.interpreter_path` must be an *absolute* path (a platform interpreter
lives outside the build graph, so there is no anchor for a relative one). To
avoid committing a machine-specific path, we compute it at build time from the
workspace root: <workspace>/.venv/bin/python3. Every checkout resolves its own
venv, so the committed files stay generic and portable.
"""

_BUILD_TEMPLATE = """\
load("@bazel_tools//tools/python:toolchain.bzl", "py_runtime_pair")

py_runtime(
    name = "venv_py3_runtime",
    interpreter_path = "{interpreter}",
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
    visibility = ["//visibility:public"],
)
"""

def _venv_python_repo_impl(repository_ctx):
    interpreter = str(repository_ctx.workspace_root) + "/.venv/bin/python3"
    if not repository_ctx.path(interpreter).exists:
        fail(
            "Project venv interpreter not found at {}.\n".format(interpreter) +
            "Create it (see TRAINING.md) before running bazel: python3 -m venv .venv",
        )
    repository_ctx.file(
        "BUILD.bazel",
        _BUILD_TEMPLATE.format(interpreter = interpreter),
        executable = False,
    )

venv_python_repo = repository_rule(
    implementation = _venv_python_repo_impl,
    doc = "Generates a py_runtime toolchain pointing at <workspace>/.venv.",
    local = True,
)

def _venv_python_ext_impl(_module_ctx):
    venv_python_repo(name = "venv_python")

venv_python = module_extension(
    implementation = _venv_python_ext_impl,
    doc = "Module extension that materializes the @venv_python repo.",
)
