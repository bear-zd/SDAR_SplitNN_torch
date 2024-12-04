def load_config(dataset="cifar10", model="resnet20"):
    config = {
        "e_lr": 0.001,
        "d_lr": 0.0005,
        "dropout": 0.0,
        "fg_lr": 0.001,
        "alpha": 0.0,
        "width": "standard",
    }
    config_table = {
        "resnet20": {
            "cifar10": {"lambda1": 0.02, "lambda2": 1e-5, "flip_rate": 0.0},
            "cifar100": {"lambda1": 0.04, "lambda2": 1e-5, "flip_rate": 0.0},
            "tinyimagenet": {"lambda1": 0.04, "lambda2": 1e-5, "flip_rate": 0.0},
            "stl10": {
                "lambda1": 0.04,
                "lambda2": 1e-5,
                "flip_rate": 0.0,
            },
        },
        "plainnet": {
            "cifar10": {"lambda1": 0.04, "lambda2": 1e-5, "flip_rate": 0.0},
            "cifar100": {"lambda1": 0.04, "lambda2": 1e-5, "flip_rate": 0.0},
            "tinyimagenet": {"lambda1": 0.04, "lambda2": 1e-5, "flip_rate": 0.0},
            "stl10": {
                "lambda1": 0.04,
                "lambda2": 1e-5,
                "flip_rate": 0.0,
            }
        }
    }

    config = {**config, **config_table[model][dataset]}

    config["e_dis_lr"] = config["e_lr"] * config["lambda1"]
    config["d_dis_lr"] = config["d_lr"] * config["lambda2"]

    return config
