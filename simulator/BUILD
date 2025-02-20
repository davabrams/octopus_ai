
py_library(
    name = "simutil",
    srcs = ["simutil.py"],
	visibility = ["//visibility:public"],
)

py_library(
    name = "agent_generator",
    srcs = ["agent_generator.py"],
    deps = [":simutil"]
)

py_library(
    name = "octopus_generator",
    srcs = ["octopus_generator.py"],
    deps = [":simutil"]
)

py_library(
    name = "surface_generator",
    srcs = ["surface_generator.py"],
    deps = [":simutil"]
)

py_library(
    name = "generators",
    srcs = glob(["*.py"]),
    visibility = ["//visibility:public"],
)