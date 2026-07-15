py_binary(
    name = "octo_viz",
    srcs = ["octo_viz.py"],
	visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//training:training",
        "//simulator:generators"
        ]
)

py_binary(
    name = "websocket_server",
    srcs = ["websocket_server.py"],
    # The server serves this HTML page from next to itself, so it must be in
    # the binary's runfiles.
    data = ["octopus-visualizer.html"],
	visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//training:training",
        "//simulator:generators"
        ]
)
