from .cls.meta_dataset import MetaDatasets
from .cls.miniimagenet import MiniImageNetDataset
from .cls.tieredimagenet import TieredImageNetDataset
from .cls.omniglot import Omniglot
from .dataset_utils import DatasetEnum
from .multi_dataset_sampler import MultipleDataset
from .cls.cifar_fs import CifarFS_dataset


def get_dataset(args, ds_name, split):
    """
    Get a traditional dataset
    :param ds_name: Name of the dataset
    :param split: Split of the dataset
    :return: Dataset object
    """
    dataset_mappings = {
        DatasetEnum.miniimagenet.name: MiniImageNetDataset,
        DatasetEnum.tieredimagenet.name: TieredImageNetDataset,
        DatasetEnum.CIFAR_FS.name: CifarFS_dataset,
        DatasetEnum.FUNGI.name: MetaDatasets,
        DatasetEnum.AIRCRAFT.name: MetaDatasets,
        DatasetEnum.BIRD.name: MetaDatasets,
        DatasetEnum.TEXTURE.name: MetaDatasets,
        DatasetEnum.omniglot_py.name: Omniglot
    }

    if ds_name not in dataset_mappings:
        raise ValueError("Unknown dataset: {}, {}".format(ds_name, split))

    if dataset_mappings[ds_name] == MetaDatasets:
        return dataset_mappings[ds_name](split=split, ds_name=ds_name)
    else:
        return dataset_mappings[ds_name](split=split)



def get_multi_dataset(args, ds_name, split="train"):
    """
    Get a meta-dataset
    :param ds_name: Name of the dataset
    :param split: Split of the dataset (default: "train")
    :return: Meta-dataset
    """
    dataset_mappings = {
        DatasetEnum.MetaBTAF.name: [DatasetEnum.BIRD, DatasetEnum.TEXTURE, DatasetEnum.AIRCRAFT, DatasetEnum.FUNGI],
        DatasetEnum.MetaCIO.name: [DatasetEnum.CIFAR_FS, DatasetEnum.miniimagenet, DatasetEnum.omniglot_py],
        DatasetEnum.MetaABF.name: [DatasetEnum.AIRCRAFT, DatasetEnum.BIRD, DatasetEnum.FUNGI],
        DatasetEnum.miniimagenet.name: [DatasetEnum.miniimagenet],
        DatasetEnum.tieredimagenet.name: [DatasetEnum.tieredimagenet],
        DatasetEnum.CIFAR_FS.name: [DatasetEnum.CIFAR_FS],
        DatasetEnum.omniglot_py.name: [DatasetEnum.omniglot_py]
    }

    if ds_name not in dataset_mappings:
        raise ValueError("Unknown meta-dataset: {}".format(ds_name))

    metadatasets = []
    for ds_enum in dataset_mappings[ds_name]:
        metadatasets.append((ds_enum.name, get_dataset(args, ds_name=ds_enum.name, split=split)))

    multiple_ds = MultipleDataset(metadatasets)
    return multiple_ds


