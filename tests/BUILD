py_test(
    name = "test_simulator",
    srcs = ["test_simulator.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//simulator:simutil",
        "//simulator:surface_generator",
        "//simulator:agent_generator",
    ]
)

py_test(
    name = "test_octopus_generator",
    srcs = ["test_octopus_generator.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//simulator:octopus_generator",
        "//simulator:simutil",
        "//simulator:surface_generator",
        "//:octo_util",
    ]
)

py_test(
    name = "test_utilities",
    srcs = ["test_utilities.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//:octo_util",
        "//simulator:simutil",
    ]
)

py_test(
    name = "test_training_losses",
    srcs = ["test_training_losses.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//training:losses",
    ]
)

py_test(
    name = "test_inference_server",
    srcs = ["test_inference_server.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//inference_server:model_inference",
    ]
)

py_test(
    name = "test_kinematics",
    srcs = ["test_kinematics.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//simulator:simutil",
        "//simulator/ilqr:costs",
    ]
)

py_test(
    name = "test_trainers",
    srcs = ["test_trainers.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//training:training",
        "//simulator:simutil",
    ]
)

py_test(
    name = "test_integration",
    srcs = ["test_integration.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//simulator:generators",
        "//simulator:octopus_generator",
        "//training:training",
        "//inference_server:model_inference",
        "//:octo_util",
        "//:octo_datagen",
    ]
)
