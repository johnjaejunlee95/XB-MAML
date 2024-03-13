from enum import Enum
from torch.utils.data.dataloader import DataLoader
import torch


class DatasetEnum(Enum):
    """
    DATASET NAME: dataset path under data
    """
    #####################################
    # cls
    #####################################
    CIFAR_FS = "CIFAR_FS"
    miniimagenet = "miniimagenet"
    omniglot_py = "omniglot_py"
    tieredimagenet = "tieredimagenet"

    BIRD = "meta-dataset/CUB_Bird"
    TEXTURE = "meta-dataset/DTD_Texture"
    AIRCRAFT = "meta-dataset/FGVC_Aircraft"
    FUNGI = "meta-dataset/FGVCx_Fungi"

    #####################################
    # meta-dataset
    #####################################
    MetaBTAF = "MetaBTAF"
    MetaCIO = "MetaCIO"
    MetaABF = "MetaABF"

    #####################################
    # reg
    #####################################
    Lines = "Lines"

    @classmethod
    def get_value_by_name(cls, ds_name):
        ds_dict = dict([(ds.name, ds.value) for ds in DatasetEnum])
        return ds_dict[ds_name]


def compute_mean_std(dataset, image_size=84):
    """
    compute the mean and std for normalization transformation
    refer to https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/4
    :param dataset:
    :param image_size:
    :return:
    """
    loader = DataLoader(dataset,
                             batch_size=64,
                             num_workers=0,
                             shuffle=False)

    mean = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dataset)

    var = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / (len(loader.dataset) * image_size * image_size))
    return mean, std


import pickle
def load_data(file):
    try:
        with open(file, 'rb') as fo:

            data = pickle.load(fo)

        return data
    except Exception as e:
        print(e)
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data
    

