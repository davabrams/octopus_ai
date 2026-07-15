py_test(
    name = "test_simulator",
    srcs = ["test_simulator.py"],
    visibility = ["//visibility:public"],
    imports = ["."],
    deps = [
        "//simulator:simutil",
        "//simulator:surface_generator",
        "//simulator:agent_generator",
    ]
)

py_test(
    name = "test_octopus_generator",
    srcs = ["test_octopus_generator.py", "helpers.py"],
    visibility = ["//visibility:public"],
    imports = ["."],
    deps = [
        "//simulator:octopus_generator",
        "//simulator:simutil",
        "//simulator:surface_generator",
        "//octopus_ai:octo_util",
    ]
)

py_test(
    name = "test_utilities",
    srcs = ["test_utilities.py"],
    visibility = ["//visibility:public"],
    imports = ["."],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:simutil",
    ]
)

py_test(
    name = "test_training_losses",
    srcs = ["test_training_losses.py"],
    visibility = ["//visibility:public"],
    imports = ["."],
    deps = [
        "//training:losses",
    ]
)

py_test(
    name = "test_inference_server",
    srcs = ["test_inference_server.py"],
    visibility = ["//visibility:public"],
    imports = ["."],
    deps = [
        "//inference_server:model_inference",
    ]
)

py_test(
    name = "test_kinematics",
    srcs = ["test_kinematics.py"],
    visibility = ["//visibility:public"],
    imports = ["."],
    deps = [
        "//simulator:simutil",
        "//simulator/ilqr:costs",
    ]
)

py_test(
    name = "test_trainers",
    srcs = ["test_trainers.py", "helpers.py"],
    visibility = ["//visibility:public"],
    imports = ["."],
    deps = [
        "//training:training",
        "//simulator:simutil",
        "//octopus_ai:octo_util",
    ]
)

py_test(
    name = "test_integration",
    srcs = ["test_integration.py", "helpers.py"],
    visibility = ["//visibility:public"],
    imports = ["."],
    deps = [
        "//simulator:generators",
        "//simulator:octopus_generator",
        "//training:training",
        "//inference_server:model_inference",
        "//octopus_ai:octo_util",
        "//octopus_ai:datagen",
    ]
)

py_test(
    name = "test_agent_movement",
    srcs = ["test_agent_movement.py", "helpers.py"],
    visibility = ["//visibility:public"],
    imports = ["."],
    deps = [
        "//simulator:octopus_generator",
        "//simulator:agent_generator",
        "//simulator:simutil",
        "//octopus_ai:octo_util",
    ]
)

py_test(
    name = "test_config_provenance",
    srcs = ["test_config_provenance.py"],
    visibility = ["//visibility:public"],
    imports = ["."],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:agent_generator",
        "//simulator:force_logger",
        "//simulator:octopus_generator",
        "//simulator:simutil",
    ]
)

py_test(
    name = "test_config_schema",
    srcs = ["test_config_schema.py"],
    visibility = ["//visibility:public"],
    imports = ["."],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:simutil",
    ]
)

py_test(
    name = "test_force_logger",
    srcs = ["test_force_logger.py", "helpers.py"],
    visibility = ["//visibility:public"],
    imports = ["."],
    deps = [
        "//simulator:octopus_generator",
        "//simulator:agent_generator",
        "//simulator:simutil",
        "//simulator:force_logger",
        "//octopus_ai:octo_util",
    ]
)

py_test(
    name = "test_frame_recorder",
    srcs = ["test_frame_recorder.py"],
    visibility = ["//visibility:public"],
    imports = ["."],
    deps = [
        "//simulator:frame_recorder",
    ]
)

py_test(
    name = "test_helpers",
    srcs = ["test_helpers.py", "helpers.py"],
    visibility = ["//visibility:public"],
    imports = ["."],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:simutil",
    ]
)

py_test(
    name = "test_limb_adjacency",
    srcs = ["test_limb_adjacency.py", "helpers.py"],
    visibility = ["//visibility:public"],
    imports = ["."],
    deps = [
        "//simulator:octopus_generator",
        "//simulator:simutil",
        "//octopus_ai:octo_util",
    ]
)

py_test(
    name = "test_prey_capture",
    srcs = ["test_prey_capture.py", "helpers.py"],
    visibility = ["//visibility:public"],
    imports = ["."],
    deps = [
        "//simulator:octopus_generator",
        "//simulator:agent_generator",
        "//simulator:simutil",
        "//octopus_ai:octo_util",
    ]
)

py_test(
    name = "test_spring_chain",
    srcs = ["test_spring_chain.py", "helpers.py"],
    visibility = ["//visibility:public"],
    imports = ["."],
    deps = [
        "//simulator:octopus_generator",
        "//simulator:simutil",
        "//simulator:spring_chain",
        "//octopus_ai:octo_util",
    ]
)
