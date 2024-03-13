import os

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from ..dataset_utils import load_data, DatasetEnum
from utils.args import parse_args

args = parse_args()
PATH = args.datasets_path

class CifarFS_dataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.is_train = split == "train"

        # load data
        self.ds_name = DatasetEnum.CIFAR_FS.name
        data_path = os.path.join(PATH + self.ds_name, "{}.pickle".format(self.split))
        data_all = load_data(data_path)
        data_label = data_all['labels']
        label_set = set(data_label)
        label_dict = dict(zip(label_set, range(len(label_set))))
        self.samples = data_all['data']
        self.labels = [label_dict[x] for x in data_label]

        # build transformers
        self.transform = None
        mean = [0.5074, 0.4867, 0.4411]
        std = [0.2675, 0.2566, 0.2763]
        normalize_transform = transforms.Normalize(mean=mean, std=std)
        resize_image = 84
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((resize_image, resize_image)),
                transforms.ToTensor(),
                normalize_transform,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((resize_image, resize_image)),
                transforms.ToTensor(),
                normalize_transform,
            ])
        print("loaded dataset: {}".format(self.__str__()))
        
    def __getitem__(self, index):
        img, label = self.samples[index], self.labels[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, 0

    def __len__(self):
        return len(self.labels)

    def get_label_num(self):
        return len(set(self.labels))

    def __str__(self):
        return "{}\t{}\t#samples: {}\t#classes: {}".format(self.ds_name, self.split, len(self.samples), len(set(self.labels)))