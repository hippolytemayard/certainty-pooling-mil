import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config.settings import CONFIG_PATH, DATA_PATH
from src.data.dataset import bag_dataset_test
from src.models.certainty_pooling import certainty_pooling
from src.models.model import Paper_network
from src.utils.utils import collate_fn, collate_fn_submission, load_yaml

if __name__ == "__main__":
    config = load_yaml(path=CONFIG_PATH)  # args.config_path)

    data_root = DATA_PATH / config.data.training.test_data_dir
    save_path = Path(config.data.training.save_dir)
    # save_model_path = save_path / config.data.training.saved_model
    # print(config.submission)

    path_models = Path(config.data.training.save_dir).glob("*.pt")

    for save_model_path in path_models:
        print(save_model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        list_id_test = [i.name.split(".")[0] for i in list(data_root.iterdir())]

        test_ds = bag_dataset_test(list_id=list_id_test, data_root=data_root)
        # print(f"len dataset : {len(test_ds)}")
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=1, collate_fn=collate_fn_submission)

        best_model = Paper_network(input_size=2048, dropout=0.5)
        best_model = best_model.to(device)

        best_model_state_dict = torch.load(save_model_path)
        best_model.load_state_dict(best_model_state_dict["model_state_dict"])

        predictions = []
        predictions_list = []
        # print(f"len {len(test_loader)}")
        # print(list_id_test)

        n = config.data.training.n_monte_carlo

        for sample in test_loader:
            inputs = sample  # [0]

            inputs_pooling = []
            with torch.no_grad():
                for input in inputs:
                    certainty_pooling_ = certainty_pooling(
                        model=best_model, x=input, n=n, device=device, epsilon=1.0e-3
                    )

                    inputs_pooling.append(certainty_pooling_)

                inputs_ = torch.tensor(inputs_pooling).float().cuda()

            best_model.eval()
            outputs = best_model(inputs_)
            outputs = outputs.squeeze(-1)
            predictions_list.append(outputs.cpu().detach().numpy())
            del inputs, inputs_pooling, certainty_pooling_, inputs_

        predictions = np.concatenate(predictions_list).ravel()

        df_predictions = pd.DataFrame(data={"ID": [i[-3:] for i in list_id_test], "Target": predictions.tolist()})

        path_csv = Path(config.submission.path_csv).parent / str(
            "submission_" + save_model_path.name.split(".")[0] + ".csv"
        )
        df_predictions.to_csv(path_csv, index=False)
