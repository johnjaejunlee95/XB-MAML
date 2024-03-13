import os
import logging
import torch
import numpy as np

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).float().mean().item()

def log_acc(args, avg_dict, meta_datasets, is_train, logger):
    
    logger.info("Version:{}\t Epochs:{}\t num_maml:{} ".format(args.version, args.current_epoch, args.num_maml+1))
    
    for cluster_name in (meta_datasets.cluster_name):
        acc = 100*avg_dict["{}_acc".format(cluster_name)].item()
        loss = avg_dict["{}_loss".format(cluster_name)].item()
        logger.info("cluster {}:\t acc = {:.4f},\t loss = {:.4f}".format(cluster_name, acc, loss))
        
    acc_ = 100 * avg_dict["{}_acc".format(meta_datasets.n_cluster)].item()
    loss_ = avg_dict["{}_loss".format(meta_datasets.n_cluster)].item()
    
    
    best_acc = 0
    best_loss = 100
    if args.best_acc <= acc:
        best_acc = acc_
        best_loss = loss_
    else:
        best_acc = args.best_acc
        best_loss = args.best_loss
    
    
    logger.info("======================Average======================")
    logger.info("Current Epoch:{}".format(args.current_epoch))
    logger.info("Current Acc = {:.4f}\t Best Acc:{:.4f}".format(acc_, best_acc))
    logger.info("Current Loss = {:.4f}\t Best Loss:{:.4f}".format(loss_, best_loss))
    
    if is_train == 'training':
        pass
    else:
        return acc_ , loss_

def cycle(dataloader):
    while True:
        for x in dataloader:
            yield x


class Average(object):

    def __init__(self, name=""):
        self.n = 0
        self.name = name
        self.history = []

    def add(self, value):
        self.history.append(value)

    def add_tensor_list(self, tensor_list):
        for item in tensor_list:
            self.add(item.item())

    def last(self):
        return self.history[-1]

    def item(self):
        return float(np.array(self.history).mean())

    def std(self):
        return np.array(self.history).std()

    def get_history(self):
        return self.history

    def length(self):
        return len(self.history)

    def simple_repr(self):
        if len(self.history) == 0:
            return ""
        if self.name in {"acc"}:
            return "{}: {:.2f}/{:.2f}".format(self.name, self.item() * 100, np.array([x * 100 for x in self.history]).std())
        elif "acc" in self.name:
            return "{}: {:.2f}".format(self.name, self.item() * 100)
        else:
            return "{}: {:.2f}".format(self.name, self.item())

    def __repr__(self):
        if len(self.history) == 0:
            return ""

        if self.name in {"acc"} or "acc" in self.name:
            return "{}: {:.2f}/{:.2f}/{:.2f}".format(self.name, self.last() * 100, self.item() * 100, np.array(self.history).std())
        elif self.name in {"loss"} or "loss" in self.name:
            return "{}: {:.4f}".format(self.name, self.item())
        else:
            return "{}: {:.4f}/{:.4f}".format(self.name, self.item(), np.array(self.history).std())

class MovingAverage:
    def __init__(self, window_size=250, min_steps=500, threshold_count=1000):
        
        self.window_size = window_size
        self.min_steps = min_steps
        self.threshold_count = threshold_count
        self.count_step = int(threshold_count/20)
        self.values = []

    def add_value(self, value):
        self.values.append(value)

    def moving_average(self):
        
        data = np.array(self.values)
        
        cumsum = np.cumsum(np.insert(data, 0, 0))  
        moving_avg = (cumsum[self.window_size:] - cumsum[:-self.window_size]) / self.window_size
        
        return moving_avg
    
    
    def check_improvement(self):
        if len(self.values) < self.min_steps:
            return False  

        moving_avg = self.moving_average()
        count = 0
        
        for i in range(len(moving_avg)-self.threshold_count, len(moving_avg), self.count_step):
            
            if i <= (self.min_steps - self.window_size):
                continue
            
            if moving_avg[i] >= 1:
                return False
            
            if moving_avg[i]> moving_avg[i-self.count_step]:
                count += 1
            else:
                count -=1
                
            if count >= (int(self.threshold_count/self.count_step)-3):
                count = 0
                return True

        return False


def get_basic_expt_info(args):
    
    n_way = args.num_ways
    
    n_support = args.num_shots
    n_query = args.num_shots_test
    y_support = torch.from_numpy(np.repeat(range(n_way), n_support)).cuda()
    y_query = torch.from_numpy(np.repeat(range(n_way), n_query)).cuda()
    return n_way, n_support, n_query, y_support, y_query


def split_support_query(x, n_support, n_query, n_way):
    """
    x: n_sample * shape
    :param x:
    :param n_support:
    :return:
    """
    x_reshaped = x.contiguous().view(n_way, n_support + n_query, *x.shape[1:])
    x_support = x_reshaped[:, :n_support].contiguous().view(n_way * n_support, *x.shape[1:])
    x_query = x_reshaped[:, n_support:].contiguous().view(n_way * n_query, *x.shape[1:])
    return x_support.cuda(), x_query.cuda()

    
def log_(name):
    
    if os.path.exists("./save/log/{}.log".format(name)) == True:
        os.remove("./save/log/{}.log".format(name))
    mylogger = logging.getLogger("my")
    mylogger.setLevel(logging.INFO)

     
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler("./save/log/{}.log".format(name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    
    mylogger.addHandler(stream_handler)
    mylogger.addHandler(file_handler)
    
    return mylogger