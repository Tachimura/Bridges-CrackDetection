import random

import numpy as np
import torch


def init_random(deterministic_behaviour=False, rseed=6789):
    # Python random module.
    random.seed(rseed)
    # Numpy module.
    np.random.seed(rseed)
    # torch random
    torch.manual_seed(rseed)
    torch.cuda.manual_seed(rseed)
    # if you are using multi-GPU.
    torch.cuda.manual_seed_all(rseed)
    torch.manual_seed(rseed)
    # For reproducible training, use deterministic_behaviour=True
    # For fast inference set it to false
    if deterministic_behaviour:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
