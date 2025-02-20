py_library(
    name="training",
    srcs=[
        "sucker.py",
        "limb.py",
        "losses.py",
        "trainutil.py",
    ],
	visibility = ["//visibility:public"],
    data=[
        ":models/limb.keras",
        ":models/sucker.keras"
        ]
)