from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config.settings import CONFIG_PATH, DATA_PATH
from src.data.dataset import bag_dataset
from src.models.evaluation import evaluation
from src.models.model import Paper_network
from src.models.training import train_loop
from src.utils.utils import collate_fn, load_yaml

if __name__ == "__main__":
    config = load_yaml(path=CONFIG_PATH)

    metadata_file = DATA_PATH / config.data.training.metadata_file
    data_root = DATA_PATH / config.data.training.data_dir

    save_path = Path(config.data.training.save_dir)
    save_path.mkdir(exist_ok=True)

    df_annotated_bags = pd.read_csv(metadata_file)

    id_train_bag, id_val_bag, target_train_bag, target_val_bag = train_test_split(
        df_annotated_bags.ID.to_list(), df_annotated_bags.Target.to_list(), test_size=0.30, random_state=42
    )

    config_tensorboard = config.data.training.tensorboard
    if config_tensorboard.use_tensorboard:
        writer = SummaryWriter(config_tensorboard.tensorboard_writer)

    epochs = config.data.training.epochs
    lr = config.data.training.lr
    batch_size = config.data.training.batch_size

    train_ds = bag_dataset(id_train_bag, target_train_bag, data_root, random_selection=True)
    val_ds = bag_dataset(id_val_bag, target_val_bag, data_root, random_selection=False)

    train_pooling_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    validation_pooling_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Paper_network(input_size=2048, dropout=0.5)
    model = model.to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = config.data.training.n_monte_carlo

    best_accuracy = 0
    best_auc = 0
    best_model = None

    for epoch in tqdm(range(epochs)):
        running_loss = train_loop(train_pooling_loader, model, criterion, optimizer, epoch, device, n=100)

        auc, acc, recall, precision = evaluation(model, validation_pooling_loader, device)

        if config_tensorboard.use_tensorboard:
            writer.add_scalar("training loss", running_loss, epoch)
            writer.add_scalar("auc", auc, epoch)
            writer.add_scalar("accuracy", acc, epoch)
            writer.add_scalar("recall", recall, epoch)
            writer.add_scalar("precision", precision, epoch)

        if auc > best_auc:
            best_auc = auc
            best_accuracy = acc

            print(f"BEST ACCURACY = {best_accuracy} , BEST AUC : {best_auc}")

            torch.save(
                obj={
                    "model_state_dict": model.state_dict(),
                },
                f=save_path / "best_model_11_04.pt",
            )
