from model_training.common.datasets.bones_dataset import BonesDataset

_DATASETS = {
    "bones_dataset": BonesDataset
}


def get_dataset(dataset_config):
    dataset_name = dataset_config["name"]
    model = _DATASETS[dataset_name].from_config(dataset_config)
    return model
