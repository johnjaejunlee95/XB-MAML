import os

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from ..dataset_utils import DatasetEnum
from utils.args import parse_args

args = parse_args()
PATH = args.datasets_path

default_stat = ([0.5074, 0.4867, 0.4411], [0.2675, 0.2566, 0.2763])

class Omniglot(Dataset):
    """
    refer to https://github.com/google-research/meta-dataset
    """
    def __init__(self, split, ds_name=DatasetEnum.omniglot_py.name):
        self.split = split
        self.is_train = split == "train"

        self.labels = None
        self.samples = None

        self.ds_name = ds_name
        data_root = os.path.join(PATH, DatasetEnum.get_value_by_name(ds_name=ds_name))
        data_path = os.path.join(data_root, "{}".format(self.split))

        # build transformers
        self.transform = None
        resize_image = 84
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize([resize_image, resize_image]),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize([resize_image, resize_image]),
                transforms.ToTensor(),
            ])

        self.m_dataset = ImageFolder(data_path, self.transform)

        self.samples = list(self.m_dataset.samples)
        self.labels = list(self.m_dataset.targets)

        print("loaded dataset: {}".format(self.__str__()))

    def __getitem__(self, index):
        img, label = self.m_dataset[index]
        return img, label

    def __len__(self):
        return len(self.labels)

    def get_label_num(self):
        return len(set(self.labels))

    def __str__(self):
        return "{}\t{}\t#samples: {}\t#classes: {}".format(self.ds_name, self.split, len(self.samples), len(set(self.labels)))
