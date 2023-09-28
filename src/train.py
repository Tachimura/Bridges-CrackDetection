import torch.cuda.amp

from utils.dataloader import load_data
from utils.model import get_model
from utils.trainer import train_model
from utils.utils import init_random, get_device


def main(config: dict):
    # Init
    init_random(deterministic_behaviour=True, rseed=config["seed"])
    device = get_device()
    print(f"Device in use: {device}")

    # Get the model, only train the classifier
    model = get_model(device=device, fine_tune=True, n_classes=config["n_classes"])

    # Setup optimizer, scheduler and scaler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=0, last_epoch=-1)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    # Load data
    data = load_data(config["data_path"], validation_size=config["validation_size"], seed=config["seed"],
                     batch_size=config["batch_size"])

    # Train the model
    best_model_data, train_results = train_model(model, optimizer, scheduler, scaler, data,
                                                 device, epochs=config["epochs"])
    best_model, best_model_accuracy = best_model_data
    # Save the trained model and print its validation accuracy
    print(f"The best model has a validation accuracy of: {best_model_accuracy}")
    if best_model:
        torch.save(best_model.state_dict(), config["model_path"])


if __name__ == "__main__":
    simulation_config = {
        "seed": 0,
        # Data folder containing 2 subfolders: Positive and Negative
        "data_path": "..\\Dataset\\Train\\",
        "validation_size": 0.3,
        "epochs": 5,
        "lr": 0.001,
        "wd": 5e-3,
        "batch_size": 512,
        # Output where we save the best model
        "model_path": "..\\SavedWeights\\trained_model.pt",
        "n_classes": 2,
    }
    main(simulation_config)
