py_library(
    name = "solver",
    srcs = ["solver.py"],
    visibility = ["//visibility:public"],
    deps = ["//simulator:profiling"],
)

py_library(
    name = "solver_parallel",
    srcs = ["solver_parallel.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":solver",
        "//simulator:profiling",
    ],
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
        ":solver_parallel",
    ],
)
