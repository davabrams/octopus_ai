py_library(
    name="losses",
    srcs=["losses.py"]
)

py_library(
    name="trainutil",
    srcs=["trainutil.py"]
)

py_library(
    name="model_loader",
    srcs = ["models/model_loader.py"],
    data = glob(["models/*"]),
	visibility = [
        "//visibility:public"
        ],
)

py_library(
    name="data_loader",
    srcs = ["datagen/data_loader.py"],
    data = glob(["datagen/*"]),
	visibility = [
        "//visibility:public"
        ],
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
        "//:octo_datagen",
        ":losses",
        ":trainutil",
        ":model_loader",
        ":data_loader",
        ]
)