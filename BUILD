py_library(
    name = "octo_util",
    srcs = ["util.py", "OctoConfig.py"],
	visibility = ["//visibility:public"],
    deps = ["//simulator:simutil"]
)

py_binary(
    name = "octo_datagen",
    srcs = ["octo_datagen.py"],
	visibility = ["//visibility:public"],
    deps = [
        "//training:model_loader",
        "//simulator:generators",
        ],
)

py_binary(
    name = "octo_model",
    srcs = ["octo_model.py"],
	visibility = ["//visibility:public"],
    deps = [
        "//simulator:generators",
        "//training:training",
        ":octo_util"
        ]
)

py_binary(
    name = "octo_viz",
    srcs = ["octo_viz.py"],
	visibility = ["//visibility:public"],
    deps = [
        ":octo_util",
        "//training:training",
        "//simulator:generators"
        ]
)