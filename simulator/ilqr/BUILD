py_library(
    name = "costs",
    srcs = ["costs.py"],
	visibility = ["//visibility:public"],
    deps = ["//simulator:simutil"]
)

py_binary(
    name = "nodemesh",
    srcs = ["nodemesh.py"],
	visibility = ["//visibility:public"],
    deps = [
        ":costs",
        "//simulator:simutil"
        ]
)
