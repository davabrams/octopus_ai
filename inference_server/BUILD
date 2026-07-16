# imports = ["."] puts inference_server/ on the runfiles import path, so the
# in-directory bare imports (`from model_inference import ...`, `from server
# import app`) resolve under `bazel run`/`bazel test` the same way they do
# when run with `cd inference_server && python ...`.
py_library(
    name = "model_inference",
    srcs = ["model_inference.py"],
    data = ["//training:saved_models"],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:simutil",
        "//training:losses",
    ],
)

# The Flask app, importable as a library (app defined at module level; app.run
# is behind a __main__ guard). The binary and the test both depend on it, so
# `from server import app` resolves under `bazel test` too.
py_library(
    name = "server_lib",
    srcs = ["server.py"],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        ":model_inference",
    ],
)

py_binary(
    name = "server",
    srcs = ["server.py"],
    imports = ["."],
    main = "server.py",
    visibility = ["//visibility:public"],
    deps = [
        ":server_lib",
    ],
)

# Integration test: boots the real Flask app and talks to it over
# localhost:8080. That needs a bindable port, which the Bazel sandbox forbids,
# so run it unsandboxed on the host (as pytest/run_tests.py already does).
py_test(
    name = "test_server",
    srcs = ["test_server.py"],
    imports = ["."],
    tags = ["local"],
    visibility = ["//visibility:public"],
    deps = [
        ":model_inference",
        ":server_lib",
    ],
)
