import random
import os
import torch
import numpy as np

from method.xb_maml_test import XB_MAML_Test
from utils.args import parse_args
from utils.meta_generator import MetaDatasetsGenerator


def testing(args, meta_generator, dataloader, multi_maml, num_test=1):

    acc_list = []
    loss_list = []
    cluster_acc_list = {}
    cluster_loss_list = {}
    
    for i in range(num_test):
        RANDOM_SEED = random.randint(0, 1000)
    
        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        ensemble_test = XB_MAML_Test(meta_generator)
        test_acc, loss, cluster_acc, cluster_loss, power_set, features = ensemble_test(args, dataloader, multi_maml)
        acc_list.append(test_acc)
        loss_list.append(loss)
        for cluster_name in (cluster_acc.keys()):
            if cluster_name not in cluster_acc_list.keys():
                cluster_acc_list[cluster_name] = []
                cluster_loss_list[cluster_name] = []
            cluster_acc_list[cluster_name].append(cluster_acc[cluster_name])
            cluster_loss_list[cluster_name].append(cluster_loss[cluster_name])
    
    avg_acc_list = {}
    avg_loss_list = {}
    avg_acc = np.array(acc_list).mean(axis=0)
    avg_loss = np.array(loss_list).mean(axis=0)
    for key in cluster_acc_list.keys():
        avg_acc_list[key] = np.array(cluster_acc_list[key]).mean(axis=0)
        avg_loss_list[key] = np.array(cluster_loss_list[key]).mean(axis=0)

    acc_ci95_list = {}
    acc_stds = np.std(np.array(acc_list), 0)
    acc_ci95 = 1.96*acc_stds/np.sqrt(len(acc_list))
    for key in cluster_acc_list.keys():
        cluster_acc_stds = np.std(np.array(cluster_acc_list[key]), 0)
        cluster_acc_ci95 = 1.96*cluster_acc_stds/np.sqrt(len(cluster_acc_list[key]))
        acc_ci95_list[key] = (cluster_acc_ci95)
    
    loss_ci95_list = {}
    loss_stds = np.std(np.array(loss_list), 0)
    loss_ci95 = 1.96*loss_stds/np.sqrt(len(loss_list))
    for key in cluster_loss_list.keys():
        cluster_loss_stds = np.std(np.array(cluster_loss_list[key]), 0)
        cluster_loss_ci95 = 1.96*cluster_loss_stds/np.sqrt(len(cluster_loss_list[key]))
        loss_ci95_list[key] = (cluster_loss_ci95)
    
    print("version:{}, num MAML:{} ".format(args.version, len(multi_maml)))
    for key in avg_acc_list.keys():
        print("{} with ci95: Average Accuracy -> {:.2f} +- {:.2f}\t Average Loss -> {:.2f} +- {:.2f}".format(key, avg_acc_list[key], acc_ci95_list[key], avg_loss_list[key], loss_ci95_list[key]))
    
    print("{}-ways {}-shots {}".format(args.num_ways, args.num_shots, args.datasets))
    print("Average Accuracy with ci95 -> {:.2f} +- {:.2f}".format(avg_acc, acc_ci95))
    print("Average Loss with ci95 -> {:.2f} +- {:.2f}".format(avg_loss, loss_ci95))
    
    return avg_acc, avg_loss,  power_set, features

if __name__ == "__main__":
    
    args = parse_args(is_test=True)
    torch.cuda.set_device(args.gpu_id)
    
    SAVE_PATH = args.checkpoint_path #./save/ckpt/~~
    CKPT_PATH = "XB_MAML_5-{}_{}_{}_version_{}_best.pt".format(str(args.num_shots), args.datasets, args.model, args.version)
    checkpoint = torch.load(os.path.join(SAVE_PATH, CKPT_PATH))
    
    meta_generators = MetaDatasetsGenerator(args)
    dataloader = meta_generators.m_dataloader
    
    if len(checkpoint['model']) == 1:
        print("init:", 1)
    else:
        print("init:", len(checkpoint['model']))
        
    testing(args, meta_generators, dataloader, checkpoint['model'], num_test=args.num_test)

        
    
