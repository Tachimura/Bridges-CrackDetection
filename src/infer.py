import torch

from utils.trainer import inference
from utils.utils import init_random, get_device
from utils.model import get_model


def main(config):
    # Init
    init_random(deterministic_behaviour=False, rseed=config["seed"])
    device = get_device()

    # Load model and trained weights
    model = get_model(device=device, n_classes=config["n_classes"], verbose=False)
    model.load_state_dict(torch.load(config["model_path"]))
    model.eval()

    # Perform inference on some data
    result = inference(model, config["data_path"], device)

    # Print the result
    if result == 0:
        print("This image has no problems!")
    elif result == 1:
        print("A crack was detected in the image!")
    elif result == 2:
        print("A spalling was detected in the image!")
    else:
        print("Something wrong happened! The model should have only 2/3 classes!")


if __name__ == "__main__":
    simulation_config = {
        "seed": 0,
        "model_path": "..\\SavedWeights\\trained_model.pt",
        "data_path": "..\\Dataset\\test.jpeg",
        "n_classes": 2,
    }
    main(simulation_config)
