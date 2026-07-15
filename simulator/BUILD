
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
    ]
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
        ":surface_generator",
        ":spring_chain",
        "//octopus_ai:octo_util",
    ]
)

py_library(
    name = "surface_generator",
    srcs = ["surface_generator.py"],
    deps = [
        ":simutil",
        "//octopus_ai:octo_util",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "generators",
    srcs = glob(["*.py"]),
    visibility = ["//visibility:public"],
    deps = ["//octopus_ai:octo_util"],
)