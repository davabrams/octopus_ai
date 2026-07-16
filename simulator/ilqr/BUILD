py_library(
    name = "solver",
    srcs = ["solver.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "residuals",
    srcs = ["residuals.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "arm",
    srcs = ["arm.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":residuals",
        ":solver",
    ],
)
