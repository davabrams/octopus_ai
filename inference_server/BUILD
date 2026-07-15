py_library(
    name = "model_inference",
    srcs = ["model_inference.py"],
    data = ["//training:saved_models"],
    visibility = ["//visibility:public"],
    deps = [
        "//:octo_util",
        "//simulator:simutil",
        "//training:losses",
    ],
)

py_binary(
    name = "server",
    srcs = ["server.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":model_inference",
    ],
)

py_test(
    name = "test_server",
    srcs = ["test_server.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":model_inference",
    ],
)
