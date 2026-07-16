py_library(
    name = "costs",
    srcs = ["costs.py"],
    visibility = ["//visibility:public"],
    deps = ["//simulator:simutil"],
)

py_library(
    name = "solver",
    srcs = ["solver.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "arm",
    srcs = ["arm.py"],
    visibility = ["//visibility:public"],
    deps = [":solver"],
)

py_binary(
    name = "nodemesh",
    srcs = ["nodemesh.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":costs",
        "//simulator:simutil",
    ],
)
