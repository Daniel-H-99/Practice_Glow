import random
import matplotlib.pyplot as plt
import torch
import os

# data manager for recording, saving, and plotting
class AverageMeter(object):
    def __init__(self, name='noname', save_all=False, save_dir='.', x_label=None):
        self.name = name
        self.save_all = save_all
        self.save_dir = save_dir
        self.x_label = x_label
        self.reset()
    def reset(self):
        self.max = - 100000000
        self.min = 100000000
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        if self.save_all:
            self.data = []
    def update(self, val, weight=1):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count
        if self.save_all:
            self.data.append(val)
        is_max, is_min = False, False
        if val > self.max:
            self.max = val
            is_max = True
        if val < self.min:
            self.min = val
            is_min = True
        return (is_max, is_min)
    def save(self):
        with open(os.path.join(self.save_dir, "{}.txt".format(self.name)), "w") as file:
            file.write("max: {0:.4f}\nmin: {1:.4f}".format(self.max, self.min))
        if self.save_all:
            plot = plt.figure()
            plt.plot(range(1, len(self.data) + 1), self.data)
            plt.ylabel(self.name)
            if self.x_label is not None:
                plt.xlabel(self.x_label)
            plt.savefig("{}/{}.png".format(self.save_dir, self.name))
            plt.close(plot)
            
def seq2sen(batch, vocab):
    sen_list = []

    for seq in batch:
        seq_strip = seq[:seq.index(1)+1]
        sen = ' '.join([vocab.itow(token) for token in seq_strip[1:-1]])
        sen_list.append(sen)

    return sen_list

def shuffle_list(src, tgt):
    index = list(range(len(src)))
    random.shuffle(index)

    shuffle_src = []
    shuffle_tgt = []

    for i in index:
        shuffle_src.append(src[i])
        shuffle_tgt.append(tgt[i])

    return shuffle_src, shuffle_tgt

# simple metric whether each predicted words match to original ones
def val_check(pred, ans):
    # pred, ans: (batch x length)
    batch, length = pred.shape
    num_correct = (pred == ans).sum()
    total = batch * length
    
    return num_correct, total

# save data, such as model, optimizer
def save(args, surfix, data):
    torch.save(data, os.path.join(args.ckpt_dir, args.name, "{}.pt".format(surfix)))

# load data, such as model, optimizer
def load(args, surfix, map_location='cpu'):
    return torch.load(os.path.join(args.ckpt_dir, "{}.pt".format(surfix)), map_location=map_location)

# draw 2D plot
def tensor_2Dplot(x, save_dir='.', name='noname', x_label=None, y_label=None):
    # x: (-1 x 2)
    assert x.shape[1] == 2
    plot = plt.scatter(x[:, 0], x[:, 1])
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    plt.savefig("{}/{}.png".format(save_dir, name))
    plt.close(plt.gcf())
    