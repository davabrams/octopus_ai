py_library(
    name="losses",
    srcs=["losses.py"],
	visibility = [
        "//visibility:public"
        ],
)

py_library(
    name="trainutil",
    srcs=["trainutil.py"],
	visibility = [
        "//visibility:public"
        ],
)

filegroup(
    name="saved_models",
    srcs=glob(["models/*.keras"]),
	visibility = [
        "//visibility:public"
        ],
)

py_library(
    name="data_utils",
    srcs=["data_utils.py"],
	visibility = [
        "//visibility:public"
        ],
)

py_library(
    name="base_loader",
    srcs = ["models/base_loader.py"],
	visibility = [
        "//visibility:public"
        ],
    deps = ["//simulator:simutil"],
)

py_library(
    name="model_loader",
    srcs = ["models/model_loader.py"],
    data = glob(["models/*"]),
	visibility = [
        "//visibility:public"
        ],
    deps = [
        ":base_loader",
        ":losses",
        ],
)

py_library(
    name="data_loader",
    srcs = ["datagen/data_loader.py"],
    data = glob(["datagen/*"]),
	visibility = [
        "//visibility:public"
        ],
    deps = [":base_loader"],
)

py_library(
    name="training",
    srcs=[
        "sucker.py",
        "limb.py",
    ],
	visibility = [
        "//visibility:public"
        ],
    deps=[
        "//octopus_ai:datagen",
        ":losses",
        ":trainutil",
        ":data_utils",
        ":model_loader",
        ":data_loader",
        ]
)
