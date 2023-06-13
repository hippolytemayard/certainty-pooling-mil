from omegaconf import OmegaConf


def collate_fn(batch):
    return tuple(zip(*batch))


def load_yaml(path: str):
    yaml_file = OmegaConf.load(path)
    return yaml_file
