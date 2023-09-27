import PIL.Image
import torch

from utils.utils import init_random, get_device
from utils.model import get_model
from utils.dataloader import valid_transformations


def main(config):
    # Init
    init_random(deterministic_behaviour=False, rseed=config["seed"])
    device = get_device()

    # Load model and trained weights
    model = get_model(device=device, n_classes=2, verbose=False)
    model.load_state_dict(torch.load(config["model_path"]))
    model.eval()

    # Load data and apply basic transformations (e.g. ToTensor)
    data = PIL.Image.open(config["data_path"])
    print(f"Loaded image: {config['data_path']}")
    data_transformed = valid_transformations(data)
    # Move to the same device as the model and create the batch dimension
    data_transformed = data_transformed.to(device, non_blocking=True)[None, :]

    # Perform inference
    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        output = model(data_transformed)
        output = torch.argmax(output, dim=-1).item()

    # Print the result
    if output == 0:
        print("No crack detected in the image!")
    elif output == 1:
        print("A crack was detected in the image!")
    else:
        print("Something wrong happened! The model should have only 2 classes!")


if __name__ == "__main__":
    simulation_seed = 0
    config = {
        "seed": 0,
        "model_path": "..\\SavedWeights\\trained_model.pt",
        "data_path": "..\\Dataset\\test.jpeg",
    }
    main(config)
