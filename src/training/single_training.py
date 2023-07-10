import logging
from pathlib import Path

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config.settings import CONFIG_PATH, DATA_PATH
from src.models.evaluation import evaluation, evaluation_batch, evaluation_tile
from src.models.model import Paper_network
from src.models.training import train_loop, train_loop_batch
from src.utils.timing import Timer


def single_training(
    train_loader, validation_loader, validation_loader_tile, config, save_model_path: str, device, use_scheduler: bool
):
    config_tensorboard = config.data.training.tensorboard

    if config_tensorboard.use_tensorboard:
        save_dir = Path(config_tensorboard.tensorboard_writer)
        log_dir = save_dir / save_model_path.name.split(".")[0]
        save_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(log_dir)

    epochs = config.data.training.epochs
    lr = config.data.training.lr

    model = Paper_network(input_size=2048, dropout=0.5)
    model = model.to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # if config.data.training.previous_model is not None:
    #    print("LOADING CHECKPOINTS")
    #    state_dict = torch.load("/data/data/saved_models/experiment_10/model3461.pt")
    #    model.load_state_dict(state_dict["model_state_dict"])
    #    optimizer.load_state_dict(state_dict["optimizer_state_dict"])
    #    optimizer.param_groups[0]["lr"] = 0.0001

    n = config.data.training.n_monte_carlo

    best_accuracy, best_auc, best_auc_tile = 0, 0, 0

    for epoch in tqdm(range(epochs)):
        running_loss = train_loop(train_loader, model, criterion, optimizer, epoch, device, n=n)

        # auc, acc, recall, precision, fig_bag = evaluation(model, validation_loader, device)
        auc, acc, recall, precision, f1, cf_matrix, fig_bag = evaluation(model, validation_loader, device)

        if validation_loader_tile is not None:
            print("test", validation_loader_tile)
            auc_tile, acc_tile, recall_tile, precision_tile, fig_tile = evaluation_tile(
                model=model, loader=validation_loader_tile, n=n, device=device
            )

        if config_tensorboard.use_tensorboard:
            writer.add_scalar("training loss", running_loss, epoch)
            writer.add_scalar("auc", auc, epoch)
            writer.add_scalar("accuracy", acc, epoch)
            writer.add_scalar("recall", recall, epoch)
            writer.add_scalar("precision", precision, epoch)
            writer.add_figure("Confusion matrix", fig_bag, epoch)
            if validation_loader_tile is not None:
                writer.add_scalar("auc_tile", auc_tile, epoch)
                writer.add_scalar("accuracy_tile", acc_tile, epoch)
                writer.add_scalar("recall_tile", recall_tile, epoch)
                writer.add_scalar("precision_tile", precision_tile, epoch)
                writer.add_figure("Confusion matrix", fig_tile, epoch)

        if auc >= best_auc and acc >= best_accuracy:
            best_auc = auc
            best_accuracy = acc

            print("SAVING MODEL AUC")
            if validation_loader_tile is not None:
                print(f"BEST ACCURACY = {acc}, BEST AUC TILE = {auc_tile} , BEST AUC : {best_auc}")
            else:
                print(f"BEST ACCURACY = {acc}, BEST AUC : {best_auc}")

            torch.save(
                obj={
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "auc": auc,
                    "acc": acc,
                    "confusion_matrix": fig_bag
                    # "auc_tile": auc_tile,
                    # "acc_tile": acc_tile,
                },
                f=save_model_path,
            )

    if config_tensorboard.use_tensorboard:
        writer.close()


def single_training_batch(
    train_loader, validation_loader, validation_loader_tile, config, criterion, save_model_path, device, use_scheduler
):
    config_tensorboard = config.data.training.tensorboard

    if config_tensorboard.use_tensorboard:
        save_dir = Path(config_tensorboard.tensorboard_writer)
        log_dir = save_dir / save_model_path.name.split(".")[0]
        save_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(log_dir)

    epochs = config.data.training.epochs
    lr = config.data.training.lr

    model = Paper_network(input_size=2048, dropout=0.5)
    model = model.to(device)

    # criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if use_scheduler:
        scheduler = StepLR(optimizer, step_size=150, gamma=0.5)

    # if config.data.training.previous_model is not None:
    #    print("LOADING CHECKPOINTS")
    #    state_dict = torch.load("/data/data/saved_models/experiment_10/model3461.pt")
    #    model.load_state_dict(state_dict["model_state_dict"])
    #    optimizer.load_state_dict(state_dict["optimizer_state_dict"])
    #    optimizer.param_groups[0]["lr"] = 0.0001

    n = config.data.training.n_monte_carlo

    best_accuracy, best_auc, best_auc_tile, best_f1 = 0, 0, 0, 0

    for epoch in tqdm(range(epochs)):
        logging.info(f"EPOCH {epoch} - TRAINING")

        with Timer() as timer:
            running_loss = train_loop_batch(train_loader, model, criterion, optimizer, epoch, device, n=n)

        # auc, acc, recall, precision, f1, cf_matrix, fig_bag = evaluation_batch(model, validation_loader, device)
        logging.info(f"EPOCH {epoch} - VALIDATION")
        with Timer() as timer:
            auc, acc, recall, precision, f1, cf_matrix, fig_bag = evaluation(model, validation_loader, device)

        if validation_loader_tile is not None:
            print("test", validation_loader_tile)
            auc_tile, acc_tile, recall_tile, precision_tile, fig_tile = evaluation_tile(
                model=model, loader=validation_loader_tile, n=n, device=device
            )

        if config_tensorboard.use_tensorboard:
            writer.add_scalar("training loss", running_loss, epoch)
            writer.add_scalar("auc", auc, epoch)
            writer.add_scalar("accuracy", acc, epoch)
            writer.add_scalar("recall", recall, epoch)
            writer.add_scalar("precision", precision, epoch)
            writer.add_figure("Confusion matrix", fig_bag, epoch)
            writer.add_scalar("F1", f1, epoch)

            if validation_loader_tile is not None:
                writer.add_scalar("auc_tile", auc_tile, epoch)
                writer.add_scalar("accuracy_tile", acc_tile, epoch)
                writer.add_scalar("recall_tile", recall_tile, epoch)
                writer.add_scalar("precision_tile", precision_tile, epoch)
                writer.add_figure("Confusion matrix", fig_tile, epoch)

        if use_scheduler:
            scheduler.step()

        if f1 >= best_f1:  # auc >= best_auc and
            best_auc = auc
            best_f1 = f1

            print("SAVING MODEL AUC")
            if validation_loader_tile is not None:
                print(f"BEST ACCURACY = {acc}, BEST AUC TILE = {auc_tile} , BEST AUC : {best_auc}")
            else:
                print(f"BEST ACCURACY = {acc}, BEST AUC : {best_auc}, BEST F1 : {f1}")

            torch.save(
                obj={
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "auc": auc,
                    "acc": acc,
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                    "confusion_matrix": cf_matrix,
                    "epoch": epoch,
                },
                f=save_model_path,
            )

    if config_tensorboard.use_tensorboard:
        writer.close()
