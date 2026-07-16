py_library(
    name = "octo_util",
    srcs = [
        "__init__.py",
        "config.py",
        "config_schema.py",
        "util.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//simulator:simutil",
        "//training:data_utils",
    ],
)

# The OctoDatagen class, importable as a library. Depended on by //training
# and the integration test; the datagen binary below wraps it.
py_library(
    name = "datagen_lib",
    srcs = ["datagen.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":octo_util",
        "//simulator:generators",
        "//training:data_loader",
        "//training:model_loader",
    ],
)

py_binary(
    name = "datagen",
    srcs = ["datagen.py"],
    main = "datagen.py",
    visibility = ["//visibility:public"],
    deps = [":datagen_lib"],
)

py_binary(
    name = "model",
    srcs = ["model.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":octo_util",
        "//simulator:generators",
        "//training",
    ],
)
