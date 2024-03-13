from torch.utils.data.dataloader import DataLoader
from datasets.get_dataset import get_multi_dataset
from datasets.multi_dataset_sampler import MultipleDatasetSampler


MetaBTAF = ['BIRD', 'TEXTURE', 'AIRCRAFT', 'FUNGI']
MetaCIO = ['CIFAR_FS', 'miniimagenet', 'omniglot_py']
MetaABF = ['AIRCRAFT', 'BIRD', 'FUNGI']
miniimagenet = ['miniimagenet']
tiereimagenet = ['tieredimagenet']
omniglot = ['omniglot_py']
CIFAR_FS = ['CIFAR_FS']
metadatasets={'MetaBTAF':MetaBTAF, 'MetaCIO':MetaCIO, 'MetaABF':MetaABF, 'miniimagenet': miniimagenet, 'tieredimagenet': tiereimagenet, 'CIFAR_FS': CIFAR_FS, "omniglot_py": omniglot}

class MetaDatasetsGenerator():
    def __init__(self, args):

        self.stages = {'train':'train', 'valid':'valid', 'test':'test'}

        self.ds_name = args.datasets
        self.cluster_name = metadatasets[self.ds_name]
        
        self.m_dataset = {stage: get_multi_dataset(args, ds_name=self.ds_name, split=stage) for stage in self.stages}
        self.n_cluster = len(self.m_dataset[self.stages['train']].cluster_info)
        max_steps = {'train' : args.epochs, 'valid': args.max_test_task*self.n_cluster, 'test': args.max_test_task*self.n_cluster} #*self.n_cluster*args.batch_size
        shuffling = {'train': True, 'valid': False, 'test': False}
        
        self.m_sampler = {
            stage: MultipleDatasetSampler(args,
                                          self.m_dataset[stage],
                                          total_steps=max_steps[stage],
                                          n_way= args.num_ways, k_shot=args.num_shots + args.num_shots_test,
                                          is_train = shuffling[stage],
                                          is_random_classes=False) for stage in self.stages
        }
        
        self.m_dataloader = {
            stage: DataLoader(self.m_dataset[stage], 
                              batch_sampler=self.m_sampler[stage], 
                              num_workers=8, 
                              pin_memory=True) for stage in self.stages
        }
        

        stat_keys_temp = ["loss", "acc", 'error_norm']
        self.stat_keys = []
        for i in range(self.n_cluster + 1):
            if i == self.n_cluster:
                for x in stat_keys_temp:
                    self.stat_keys.append("{}_{}".format(i, x))
            else:
                for x in stat_keys_temp:
                    self.stat_keys.append("{}_{}".format(self.cluster_name[i], x))
            
        for x in stat_keys_temp:
            self.stat_keys.append(x)