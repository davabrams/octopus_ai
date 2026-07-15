py_library(
    name = "octo_util",
    srcs = ["__init__.py", "util.py", "config.py", "config_schema.py"],
	visibility = ["//visibility:public"],
    deps = [
        "//simulator:simutil",
        "//training:data_utils",
        ]
)

py_binary(
    name = "datagen",
    srcs = ["datagen.py"],
	visibility = ["//visibility:public"],
    deps = [
        ":octo_util",
        "//training:model_loader",
        "//training:data_loader",
        "//simulator:generators",
        ],
)

py_binary(
    name = "model",
    srcs = ["model.py"],
	visibility = ["//visibility:public"],
    deps = [
        ":octo_util",
        "//simulator:generators",
        "//training:training",
        ]
)
