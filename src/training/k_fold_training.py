from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.config.settings import CONFIG_PATH, DATA_PATH
from src.data.dataset import bag_dataset
from src.training.single_training import single_training
from src.utils.utils import collate_fn, load_yaml

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_yaml(path=CONFIG_PATH)

    metadata_file = DATA_PATH / config.data.training.metadata_file
    data_root = DATA_PATH / config.data.training.data_dir

    save_path = Path(config.data.training.save_dir)
    save_path.mkdir(exist_ok=True)

    save_model_path = save_path / config.data.training.saved_model
    checkpoint_path = config.data.training.previous_model

    df_annotated_bags = pd.read_csv(metadata_file)

    annotated_bags_id = np.array(df_annotated_bags.ID.to_list())
    annotated_bags_target = np.array(df_annotated_bags.Target.to_list())

    # kf = KFold(n_splits=5)
    skf = StratifiedKFold(n_splits=5)
    for i, (train_index, test_index) in enumerate(skf.split(annotated_bags_id, annotated_bags_target), 1):
        save_model_path = save_path / f"model_fold{i}.pt"

        id_train_bag, target_train_bag = annotated_bags_id[train_index], annotated_bags_target[train_index]
        id_val_bag, target_val_bag = annotated_bags_id[test_index], annotated_bags_target[test_index]

        batch_size = config.data.training.batch_size

        train_ds = bag_dataset(id_train_bag, target_train_bag, data_root, random_selection=True)
        val_ds = bag_dataset(id_val_bag, target_val_bag, data_root, random_selection=False)

        train_pooling_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn
        )
        validation_pooling_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn
        )

        single_training(
            train_loader=train_pooling_loader,
            validation_loader=validation_pooling_loader,
            config=config,
            save_model_path=save_model_path,
            device=device,
        )
