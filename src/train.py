import copy
import datetime
import time

import torch.cuda.amp
from torch.nn.functional import cross_entropy

from utils.dataloader import load_data
from utils.measures import Accuracy, AverageMeter
from utils.model import get_model
from utils.utils import init_random, get_device


def run(model, dataloader, optimizer, scaler, device, epoch, run_type="Train"):
    acc = Accuracy((1,))

    accuracy_meter_1 = AverageMeter()
    loss_meter = AverageMeter()
    batch_time = AverageMeter()

    train = optimizer is not None
    model.train(train)

    if train:
        optimizer.zero_grad()
    t1 = time.time()
    # Define the loss function
    loss_fn = cross_entropy

    for batch, (images, target) in enumerate(dataloader):
        target = target.type(torch.LongTensor)
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                output = model(images)
                loss = loss_fn(output, target)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

        accuracy = acc(output, target)
        accuracy_meter_1.update(accuracy[0].item(), target.shape[0])
        loss_meter.update(loss.item(), target.shape[0])

        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(dataloader) - batch)

        print(f"{run_type}: [{epoch}][{batch + 1}/{len(dataloader)}]:\t"
              f"BT {batch_time.avg:.3f}\t"
              f"ETA {datetime.timedelta(seconds=eta)}\t"
              f"loss {loss_meter.avg:.3f}\t")
    return {
        'loss': loss_meter.avg,
        'accuracy': {
            "top1": accuracy_meter_1.avg / 100,
        }
    }


def train_cycle(model, optimizer, scheduler, scaler, data, device, epochs=100):
    best_model = None
    best_accuracy = 0.0
    train_loader, valid_loader = data
    for epoch in range(epochs):
        train_perf = run(model, train_loader, optimizer, scaler, device, epoch, "Train")
        valid_perf = run(model, valid_loader, None, None, device, epoch, "Validation")
        if valid_perf["accuracy"]["top1"] > best_accuracy:
            best_model = copy.deepcopy(model)
            best_accuracy = valid_perf["accuracy"]["top1"]
        scheduler.step()
        print("Train:", train_perf)
        print("Validation:", valid_perf)
    return best_model, best_accuracy


def main(config: dict):
    # Init
    init_random(deterministic_behaviour=True, rseed=config["seed"])
    device = get_device()

    # Get the model, only train the classifier
    model = get_model(device=device, fine_tune=True, n_classes=2)

    # Setup optimizer, scheduler and scaler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=0, last_epoch=-1)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    # Load data
    data = load_data(config["data_path"], validation_size=config["validation_size"], seed=config["seed"],
                     batch_size=config["batch_size"])

    # Train the model
    best_model, best_model_accuracy = train_cycle(model, optimizer, scheduler, scaler, data,
                                                  device, epochs=config["epochs"])

    # Save the trained model and print its validation accuracy
    print(f"The best model has a validation accuracy of: {best_model_accuracy}")
    if best_model:
        torch.save(best_model.state_dict(), config["model_path"])


if __name__ == "__main__":
    simulation_config = {
        "seed": 0,
        # Data folder containing 2 subfolders: Positive and Negative
        "data_path": "C:\\Users\\Gianluca\\Downloads\\5y9wdsg2zt-2\\",
        "validation_size": 0.3,
        "epochs": 5,
        "lr": 0.001,
        "wd": 5e-3,
        "batch_size": 512,
        # Output where we save the best model
        "model_path": "..\\SavedWeights\\trained_model.pt"
    }
    main(simulation_config)
