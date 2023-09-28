import copy
import datetime
import time

from PIL import Image
import torch
from torch.nn import Module
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from utils.dataloader import valid_transformations
from utils.measures import Accuracy, AverageMeter


def inference(model: Module, data_path: str, device) -> int:
    data = Image.open(data_path)
    print(f"Loaded image: {data_path}")
    data_transformed = valid_transformations(data)
    # Move to the same device as the model and create the batch dimension
    data_transformed = data_transformed.to(device, non_blocking=True)[None, :]
    # Perform inference
    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        output = model(data_transformed)
        output = torch.argmax(output, dim=-1).item()
    return output


def run(model: Module, dataloader: DataLoader, optimizer: Optimizer | None, scaler: GradScaler | None,
        device: str, epoch: int, run_type: str = "Train") -> dict:
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


def train_model(model: Module, optimizer: Optimizer, scheduler, scaler: GradScaler,
                data: tuple[DataLoader, DataLoader], device: str, epochs: int = 100) \
        -> tuple[tuple[Module, float], tuple[list[dict], list[dict]]]:
    best_model = None
    best_accuracy = 0.0
    train_loader, valid_loader = data
    train_accuracies, valid_accuracies = [], []
    print(f"Starting training for {epochs} epochs.")
    for epoch in range(epochs):
        train_perf = run(model, train_loader, optimizer, scaler, device, epoch, "Train")
        valid_perf = run(model, valid_loader, None, None, device, epoch, "Validation")
        if valid_perf["accuracy"]["top1"] > best_accuracy:
            best_model = copy.deepcopy(model)
            best_accuracy = valid_perf["accuracy"]["top1"]
        scheduler.step()
        print("Train:", train_perf)
        print("Validation:", valid_perf)
        train_accuracies.append(train_perf)
        valid_accuracies.append(valid_perf)
    return (best_model, best_accuracy), (train_accuracies, valid_accuracies)
