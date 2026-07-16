exports_files(["analyzer.html"])

py_binary(
    name = "octo_viz",
    srcs = ["octo_viz.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:generators",
        "//training",
    ],
)

# The analyzer server, importable as a library (the server class is defined at
# module level; asyncio.run is behind a __main__ guard). The binary and the
# protocol tests both depend on it, so imports resolve under `bazel test` too.
py_library(
    name = "websocket_server_lib",
    srcs = ["websocket_server.py"],
    # The server serves this HTML page from next to itself, so it must be in
    # the runfiles.
    data = ["analyzer.html"],
    imports = [".."],
    visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:headless_runner",
        "//simulator:run_store",
        "//simulator:sim_recorder",
        "//training",
    ],
)

py_binary(
    name = "websocket_server",
    srcs = ["websocket_server.py"],
    data = ["analyzer.html"],
    main = "websocket_server.py",
    visibility = ["//visibility:public"],
    deps = [":websocket_server_lib"],
)
