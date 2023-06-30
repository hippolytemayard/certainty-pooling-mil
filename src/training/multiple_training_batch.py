from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.config.settings import CONFIG_PATH, DATA_PATH
from src.data.dataset import bag_dataset, tiles_dataset
from src.training.single_training import single_training, single_training_batch
from src.models.losses import W_BCEWithLogitsLoss
from src.utils.utils import collate_fn, load_yaml, compute_class_freqs
import random


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_yaml(path=CONFIG_PATH)

    metadata_file = DATA_PATH / config.data.training.metadata_file
    meta_data_file_tile = DATA_PATH / config.data.training.metadata_file_tile

    tile_df = pd.read_csv(meta_data_file_tile)
    tile_df["ID"] = tile_df.iloc[:, 0].str[:6]
    tiles_ID = tile_df.ID.unique()

    data_root = DATA_PATH / config.data.training.data_dir

    save_path = Path(config.data.training.save_dir)
    save_path.mkdir(exist_ok=True)

    ok = False
    for i in range(config.data.training.number_training):
        save_model_path = save_path / f"model{i}.pt"

        df_annotated_bags = pd.read_csv(metadata_file)

        annotated_bags_id = np.array(df_annotated_bags.ID.to_list())
        annotated_bags_target = np.array(df_annotated_bags.Target.to_list())

        id_train_bag, id_val_bag, target_train_bag, target_val_bag = train_test_split(
            annotated_bags_id,
            annotated_bags_target,
            test_size=config.data.split,
            random_state=None,
            stratify=annotated_bags_target,
        )

        w_p, w_n = compute_class_freqs(target_train_bag)

        print(f"Positive samples : {sum(annotated_bags_target)}")
        print(f"Negative samples : {len(annotated_bags_target) - sum(annotated_bags_target)}")

        id_val_bag_ = [i[:6] for i in id_val_bag if i.split("_")[-1] == "annotated"]
        list_id_tiles = [id_ for id_ in tiles_ID if id_ in id_val_bag_]

        print(len(list_id_tiles))

        batch_size = config.data.training.batch_size
        use_scheduler = config.data.training.scheduler
        num_workers = config.data.training.num_workers

        train_ds = bag_dataset(id_train_bag, target_train_bag, data_root, random_selection=True)
        val_ds = bag_dataset(id_val_bag, target_val_bag, data_root, random_selection=False)
        val_ds_tile = tiles_dataset(list_id=list_id_tiles, tile_df=tile_df, data_root=data_root, random_selection=False)

        train_pooling_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn
        )
        validation_pooling_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
        )
        # validation_loader = DataLoader(val_ds_tile, batch_size=1, shuffle=False, num_workers=4)
        validation_loader = None
        criterion = W_BCEWithLogitsLoss(w_p=w_p, w_n=w_n)
        # criterion = torch.nn.BCELoss()

        single_training_batch(
            train_loader=train_pooling_loader,
            validation_loader=validation_pooling_loader,
            validation_loader_tile=validation_loader,
            config=config,
            save_model_path=save_model_path,
            device=device,
            use_scheduler=use_scheduler,
            criterion=criterion,
        )
