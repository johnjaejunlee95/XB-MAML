import torch
import numpy as np
from torch.utils.data import Dataset

DS_INFO_INDEX = 1
SAMPLE_START_INDEX = 2
SAMPLE_END_INDEX = 3
LABEL_START_INDEX = 4
LABEL_END_INDEX = 5


class MultipleDataset(Dataset):
    def __init__(self, dataset_dict):
        self.cluster_info = [] # index -> (ds_name, ds_info, sample_start, sample_end, label_start, label_end)
        self.labels = None
        self.total_label_num = 0
        self.total_sample_num = 0
        for ds_name, ds_info in dataset_dict:
            label_num = ds_info.get_label_num()
            sample_num = len(ds_info)
            if self.labels is None:
                self.labels = self.total_label_num + np.array(ds_info.labels)
            else:
                self.labels = np.append(self.labels, self.total_label_num + np.array(ds_info.labels))
            info = (ds_name, ds_info,
                    self.total_sample_num, self.total_sample_num+sample_num,
                    self.total_label_num, self.total_label_num+label_num)
            self.cluster_info.append(info)

            self.total_sample_num += sample_num
            self.total_label_num += label_num

    def __getitem__(self, index):
        # find cluster
        cluster_id = 0
        for i, info in enumerate(self.cluster_info):
            if info[SAMPLE_START_INDEX] <= index < info[SAMPLE_END_INDEX]:
                cluster_id = i
        ds_info = self.cluster_info[cluster_id]
        x, _ = ds_info[DS_INFO_INDEX][index - ds_info[SAMPLE_START_INDEX]]
        return x, torch.LongTensor([self.labels[index]])

    def __len__(self):
        return self.total_sample_num

    def get_label_num(self):
        return self.total_label_num


class MultipleDatasetSampler(object):
    def __init__(self, args, multiple_datasets, total_steps=20, n_way=5, k_shot=1, start_fraction=0, end_fraction=1, is_train=True, is_random_classes=False):
        """
        n way k shot sampler for few-shot learning problem
        :param labels: list, index means sample index, value means label
        :param total_steps: train step number
        :param n_way: number of class per batch
        :param k_shot: number of samples per class in one batch
        """
        self.total_steps = total_steps
        self.n_way = n_way
        self.k_shot = k_shot
        self.multiple_datasets = multiple_datasets
        self.args = args
        self.cluster_datasets = []
        self.is_train = is_train
        for i in range(len(multiple_datasets.cluster_info)):
            self.cluster_datasets.append(multiple_datasets.cluster_info[i][0])
        self.cluster_num = len(multiple_datasets.cluster_info)
        
        self.cluster_samples = np.array([], dtype=np.int8)
        for i in range(self.cluster_num):
            if is_train:
                self.cluster_samples = []
                for num in np.random.choice(self.cluster_num, self.total_steps):
                    self.cluster_samples.extend([num]*args.batch_size)
                self.cluster_samples = np.array(self.cluster_samples, dtype=np.int8)
            else:
                sort_list = np.repeat(int(i), args.max_test_task)
                self.cluster_samples = np.hstack((self.cluster_samples, sort_list))
        # np.random.shuffle(self.cluster_samples)
        # print(self.cluster_samples)
        print("{} num_task:{}".format(args.datasets, self.cluster_samples.shape[0]))
            # self.cluster_samples.sort()

        self.label_2_instance_ids = []

        labels = multiple_datasets.labels

        for i in range(max(labels) + 1):
            ids = np.argwhere(labels == i).reshape(-1)  # output shape of "np.argwhere" is column array
            ids = torch.from_numpy(ids)
            num_instance = len(ids)
            start_id = max(0, int(np.floor(start_fraction * num_instance)))
            end_id = min(num_instance, int(np.floor(end_fraction * num_instance)))
            self.label_2_instance_ids.append(ids[start_id: end_id])

        self.labels_num = len(self.label_2_instance_ids)
        self.labels = labels
        self.is_random_classes = is_random_classes

    def __len__(self):
        if self.is_train:
            return self.total_steps*self.args.batch_size
        else:
            return self.total_steps#*self.args.batch_size

    def __iter__(self):
        if self.is_train:
            for i_batch in range(self.total_steps*self.args.batch_size):
                batch = []
                if self.is_random_classes:
                    class_ids = np.random.choice(self.labels_num, self.n_way, replace=False)
                else:
                    cluster_id = self.cluster_samples[i_batch]
                    
                    class_ids = np.random.choice(np.arange(self.multiple_datasets.cluster_info[cluster_id][4], self.multiple_datasets.cluster_info[cluster_id][5]), self.n_way, replace=False)
                for class_id in class_ids:
                    instances_ids = self.label_2_instance_ids[class_id]
                    instances_ids_selected = torch.randperm(len(instances_ids))[0:self.k_shot]
                    batch.append(instances_ids[instances_ids_selected])
                batch = torch.stack(batch).reshape(-1)
                yield batch
        else:
            for i_batch in range(self.total_steps):
                batch = []
                if self.is_random_classes:
                    class_ids = np.random.choice(self.labels_num, self.n_way, replace=False)
                else:
                    cluster_id = self.cluster_samples[i_batch]
                    
                    class_ids = np.random.choice(np.arange(self.multiple_datasets.cluster_info[cluster_id][4], self.multiple_datasets.cluster_info[cluster_id][5]), self.n_way, replace=False)
                for class_id in class_ids:
                    instances_ids = self.label_2_instance_ids[class_id]
                    instances_ids_selected = torch.randperm(len(instances_ids))[0:self.k_shot]
                    batch.append(instances_ids[instances_ids_selected])
                batch = torch.stack(batch).reshape(-1)
                yield batch