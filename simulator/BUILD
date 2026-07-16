py_library(
    name = "simutil",
    srcs = ["simutil.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "agent_generator",
    srcs = ["agent_generator.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":simutil",
        "//octopus_ai:octo_util",
    ],
)

py_library(
    name = "spring_chain",
    srcs = ["spring_chain.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "force_logger",
    srcs = ["force_logger.py"],
    visibility = ["//visibility:public"],
    deps = ["//octopus_ai:octo_util"],
)

py_library(
    name = "frame_recorder",
    srcs = ["frame_recorder.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "octopus_generator",
    srcs = ["octopus_generator.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":simutil",
        ":spring_chain",
        ":surface_generator",
        "//octopus_ai:octo_util",
    ],
)

py_library(
    name = "surface_generator",
    srcs = ["surface_generator.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":simutil",
        "//octopus_ai:octo_util",
    ],
)

# Convenience aggregator: pulls in every simulator library so callers can
# depend on the whole package with one label. No srcs of its own — each module
# is owned by its dedicated target above.
py_library(
    name = "generators",
    visibility = ["//visibility:public"],
    deps = [
        ":agent_generator",
        ":force_logger",
        ":frame_recorder",
        ":octopus_generator",
        ":simutil",
        ":spring_chain",
        ":surface_generator",
    ],
)
