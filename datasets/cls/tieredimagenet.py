from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from ..dataset_utils import load_data, DatasetEnum
import os
import numpy as np
import pickle

from utils.args import parse_args

args = parse_args()
PATH = args.datasets_path

class TieredImageNetDataset(Dataset):
    def __init__(self, split):
        
        self.name = DatasetEnum.tieredimagenet.name
        self.ds_name = PATH + DatasetEnum.tieredimagenet.name
        self.split = split

        # csv_path = path.join(PathUtils.get_dataset_path(self.ds_name), "split", "{}.csv".format(split))
        # images_path = path.join(PathUtils.get_dataset_path(self.ds_name), "images")

        pkl_path = os.path.join(self.ds_name, "{}_labels.pkl".format(split))
        images_path = os.path.join(self.ds_name, "{}_images.pkl".format(split))
        
        labels = load_data(pkl_path)
        data_label = labels['labels']        
        label_set = set(data_label)
        label_dict = dict(zip(label_set, range(len(label_set))))
        self.labels = [label_dict[x] for x in data_label]
        
        with open(images_path, 'rb') as f:
            self.samples = pickle.load(f)
        
        is_train = split == 'train'
        mean, std = [0.4721, 0.4533, 0.4099], [0.2771, 0.2677, 0.2844]
        resize_image = 84
        normalize_transform = transforms.Normalize(mean=mean,  std=std)

        # transforms follows: https://github.com/kjunelee/MetaOptNet/blob/master/data/mini_imagenet.py
        self.transform = None
        if is_train:
           self.transform = transforms.Compose([
                transforms.Resize((resize_image, resize_image)),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize_transform,
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((resize_image, resize_image)),
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize_transform,
            ])
        print("loaded dataset: {}".format(self.__str__()))

    def __getitem__(self, index):
        image, label = self.samples[index], self.labels[index]
        image = Image.fromarray(image)
        
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)

    def get_label_num(self):
        return len(set(self.labels))

    def __str__(self):
        return "{}\t{}\t#samples: {}\t#classes: {}".format(self.name, self.split, len(self.samples), len(set(self.labels)))
