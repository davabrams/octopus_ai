py_test(
    name = "test_simulator",
    srcs = ["test_simulator.py"],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//simulator:agent_generator",
        "//simulator:simutil",
        "//simulator:surface_generator",
    ],
)

py_test(
    name = "test_octopus_generator",
    srcs = [
        "helpers.py",
        "test_octopus_generator.py",
    ],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:octopus_generator",
        "//simulator:simutil",
        "//simulator:surface_generator",
    ],
)

py_test(
    name = "test_utilities",
    srcs = ["test_utilities.py"],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:simutil",
    ],
)

py_test(
    name = "test_training_losses",
    srcs = ["test_training_losses.py"],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//training:losses",
    ],
)

py_test(
    name = "test_inference_server",
    srcs = ["test_inference_server.py"],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//inference_server:model_inference",
    ],
)

py_test(
    name = "test_kinematics",
    srcs = ["test_kinematics.py"],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//simulator:simutil",
        "//simulator/ilqr:costs",
    ],
)

py_test(
    name = "test_ilqr",
    srcs = [
        "helpers.py",
        "test_ilqr.py",
    ],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:octopus_generator",
        "//simulator:simutil",
        "//simulator/ilqr:arm",
        "//simulator/ilqr:solver",
    ],
)

py_test(
    name = "test_trainers",
    srcs = [
        "helpers.py",
        "test_trainers.py",
    ],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:simutil",
        "//training",
    ],
)

py_test(
    name = "test_integration",
    srcs = [
        "helpers.py",
        "test_integration.py",
    ],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//inference_server:model_inference",
        "//octopus_ai:datagen_lib",
        "//octopus_ai:octo_util",
        "//simulator:generators",
        "//simulator:octopus_generator",
        "//training",
    ],
)

py_test(
    name = "test_agent_movement",
    srcs = [
        "helpers.py",
        "test_agent_movement.py",
    ],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:agent_generator",
        "//simulator:octopus_generator",
        "//simulator:simutil",
    ],
)

py_test(
    name = "test_config_provenance",
    srcs = ["test_config_provenance.py"],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:agent_generator",
        "//simulator:force_logger",
        "//simulator:octopus_generator",
        "//simulator:simutil",
    ],
)

py_test(
    name = "test_config_schema",
    srcs = [
        "helpers.py",
        "test_config_schema.py",
    ],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:simutil",
    ],
)

py_test(
    name = "test_force_logger",
    srcs = [
        "helpers.py",
        "test_force_logger.py",
    ],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:agent_generator",
        "//simulator:force_logger",
        "//simulator:octopus_generator",
        "//simulator:simutil",
    ],
)

py_test(
    name = "test_frame_recorder",
    srcs = ["test_frame_recorder.py"],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//simulator:frame_recorder",
    ],
)

py_test(
    name = "test_sim_recorder",
    srcs = [
        "helpers.py",
        "test_sim_recorder.py",
    ],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:agent_generator",
        "//simulator:octopus_generator",
        "//simulator:sim_recorder",
        "//simulator:simutil",
        "//simulator:surface_generator",
    ],
)

py_test(
    name = "test_headless_runner",
    srcs = [
        "helpers.py",
        "test_headless_runner.py",
    ],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:headless_runner",
        "//simulator:octopus_generator",
        "//simulator:simutil",
    ],
)

py_test(
    name = "test_websocket_protocol",
    srcs = ["test_websocket_protocol.py"],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:headless_runner",
        "//simulator:simutil",
        "//visualizer:websocket_server_lib",
    ],
)

py_test(
    name = "test_record_playback",
    srcs = [
        "helpers.py",
        "test_record_playback.py",
    ],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:headless_runner",
        "//simulator:simutil",
        "//visualizer:websocket_server_lib",
    ],
)

# Node-based analyzer core-logic tests (node run skipped if node is missing in
# the sandbox; the static HTML checks always run). analyzer.html is a data dep.
py_test(
    name = "test_analyzer_core",
    srcs = ["test_analyzer_core.py"],
    data = ["//visualizer:analyzer.html"],
    imports = ["."],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_helpers",
    srcs = [
        "helpers.py",
        "test_helpers.py",
    ],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:simutil",
    ],
)

py_test(
    name = "test_limb_adjacency",
    srcs = [
        "helpers.py",
        "test_limb_adjacency.py",
    ],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:octopus_generator",
        "//simulator:simutil",
    ],
)

py_test(
    name = "test_prey_capture",
    srcs = [
        "helpers.py",
        "test_prey_capture.py",
    ],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:agent_generator",
        "//simulator:octopus_generator",
        "//simulator:simutil",
    ],
)

py_test(
    name = "test_spring_chain",
    srcs = [
        "helpers.py",
        "test_spring_chain.py",
    ],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "//octopus_ai:octo_util",
        "//simulator:octopus_generator",
        "//simulator:simutil",
        "//simulator:spring_chain",
    ],
)
