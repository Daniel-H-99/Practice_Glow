import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader 
from dataloader import get_loader
from utils import *
from model import Glow, RealNVP
from tqdm import tqdm
import logging
import time
from PIL import Image

def preprocess(args, data):
    loss = torch.Tensor([0.0]).mean()
    if args.realnvp:
        data = torch.cat(data)
        data = data.unsqueeze(2).unsqueeze(3)
    else:
        data = data[0]
        data, loss = dequantize(args, data)
    return data, loss

def dequantize(args, data):
    data = (data * (args.dequantized_levels - 1) + torch.rand_like(data)) / args.dequantized_levels
    data = (2 * data - 1) * args.dequantized_bound 
    data = (data + 1) / 2
    data = torch.log(data) - torch.log(1 - data)

    loss = (F.softplus(data) + F.softplus(-data) - F.softplus(torch.Tensor([(1 - args.dequantized_bound) / args.dequantized_bound]))).sum() / data.shape[0]
    return data, -loss

def postprocess(args, data):
    if not args.realnvp:
        data = (F.sigmoid(data) * (2 ** args.bits)).floor() / (2 ** args.bits)
    else:
        data = data.squeeze()

    return data

def bits_per_dimensions(args, loss):
	return loss / (np.log(2) * args.pixels)

class PriorDistribution():
    def __init__(self, args, prior):
        self.args = args
        self.prior = prior
    def mean_log_prob(self, input):
        output = self.prior.log_prob(input)
        if self.args.dequantized:
            output -= np.log(args.dequantized_levels) * self.args.pixels
        return -output.mean(dim=0)
    def sample(self, b):
        return self.prior.sample([b])
    
def main(args):
    
    # 0. initial setting
    
    # set environmet
    cudnn.benchmark = True

    if not os.path.isdir(os.path.join(args.path, './ckpt')):
        os.mkdir(os.path.join(args.path,'./ckpt'))
    if not os.path.isdir(os.path.join(args.path,'./results')):
        os.mkdir(os.path.join(args.path,'./results'))    
    if not os.path.isdir(os.path.join(args.path, './ckpt', args.name)):
        os.mkdir(os.path.join(args.path, './ckpt', args.name))
    if not os.path.isdir(os.path.join(args.path, './results', args.name)):
        os.mkdir(os.path.join(args.path, './results', args.name))
    if not os.path.isdir(os.path.join(args.path, './results', args.name, "log")):
        os.mkdir(os.path.join(args.path, './results', args.name, "log"))

    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler = logging.FileHandler(os.path.join(args.path, "results/{}/log/{}.log".format(args.name, time.strftime('%c', time.localtime(time.time())))))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    args.logger = logger
    
    # set cuda
    if torch.cuda.is_available():
        args.logger.info("running on cuda")
        args.device = torch.device("cuda")
        args.use_cuda = True
    else:
        args.logger.info("running on cpu")
        args.device = torch.device("cpu")
        args.use_cuda = False
        
    args.logger.info("[{}] starts".format(args.name))
    
    # 1. load data
    
    if args.realnvp:
        args.logger.info("loading data...")
        loader = get_loader(args, name='realnvp_toydata.csv')
    else:
        transform_cifar = transforms.Compose([
            transforms.RandomHorizontalFlip(),
			# transforms.CenterCrop(160),
			# transforms.Resize(128),
            transforms.ToTensor()
        ])
        dataset = torchvision.datasets.CIFAR10(root=os.path.join(args.path, args.data_dir), train=True, download=True, transform=transform_cifar)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


    # 2. setup
    
    args.logger.info("setting up...")
    args.pixels = args.channels * args.width * args.height
    model = RealNVP(args) if args.realnvp else Glow(args)
    model.to(args.device)
    prior = PriorDistribution(args, torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(args.pixels).to(args.device), torch.eye(args.pixels).to(args.device)))
    optimizer = optim.Adamax(model.parameters(), lr=args.learning_rate, weight_decay=5e-5)
    lr_lambda = lambda epoch: min(1.0, (epoch + 1) / args.warmup) 
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if args.load:
    	loaded_data = load(args, args.ckpt)
    	model.load_state_dict(loaded_data['model'])
    	optimizer.load_state_dict(loaded_data['optimizer'])

    # 3. train / test
    
    if not args.test:
        # train
        args.logger.info("starting training")
        train_loss_meter = AverageMeter(args, name="Loss", save_all=True, x_label="epoch")
        steps = 0
        for epoch in range(1, 1 + args.epochs):
            spent_time = time.time()
            model.train()
            train_loss_tmp_meter = AverageMeter(args)
            for data in tqdm(loader):
                if args.start_from_step is not None:
                    if steps < args.start_from_step:
                        steps += 1
                        continue
                optimizer.zero_grad()
                data, loss = preprocess(args, data)
                batch = data.shape[0]
                z, loss = model(data.to(args.device), loss=loss.to(args.device))
                assert z.shape == (batch, args.pixels)
                loss += prior.mean_log_prob(z)
                loss.backward()
                optimizer.step()
                
                train_loss_tmp_meter.update(loss, weight=batch)
                steps += 1
                
                # validate and save
                if steps % args.save_period == 0:
                    model.eval()
                    spent_time = time.time()
                    save(args, "last", {'model': model.state_dict(),
                    										'optimizer': optimizer.state_dict()})
                    if args.realnvp:
                        original = torch.zeros(0)
                        zs = torch.zeros(0)
                        xs = torch.zeros(0)

                        for data in tqdm(loader):
                            data, _ = preprocess(args, data)
                            batch = data.shape[0]

                            with torch.no_grad():
                                z, _ = model(data.to(args.device))
                                x = model.decode(z)

                            data = data.squeeze()
                            z = z.squeeze()
                            x = x.squeeze()
                            assert z.shape == (batch, args.pixels)
                            assert x.shape == (batch, args.channels)
                            original = torch.cat((original, data))
                            zs = torch.cat((zs, z.cpu()))
                            xs = torch.cat((xs, x.cpu()))

                        plot_2D(args, original[:, 0], original[:, 1], name='[{}]toyplot_data'.format(epoch))                    
                        plot_2D(args, zs[:, 0], zs[:, 1], name='[{}]toyplot_z'.format(epoch))
                        plot_2D(args, xs[:, 0], xs[:, 1], name='[{}]toyplot_x'.format(epoch))

                    else:
                        z = prior.sample(args.display_size)
                        x = postprocess(args, model.decode(z))
                        assert x.shape == (args.display_size, args.channels, args.width, args.height)
                        grids = torchvision.utils.make_grid(x, nrow=int(np.sqrt(args.display_size)))
                        display = transforms.ToPILImage()(grids)
                        display.save(os.path.join(args.path, args.result_dir, args.name, 'step{}.jpg'.format(steps)))
                    
                    spent_time = time.time() - spent_time
                    args.logger.info("[{}] plot recorded, took {} seconds".format(steps, spent_time))
                    train_loss_meter.save()
                
            if args.realnvp:
                train_loss_meter.update(train_loss_tmp_meter.avg)
            else:
                train_loss_meter.update(bits_per_dimensions(args, train_loss_tmp_meter.avg))
            spent_time = time.time() - spent_time
            if args.realnvp:
                args.logger.info("[{}] train loss: {:.3f} took {:.1f} seconds".format(epoch, train_loss_tmp_meter.avg, spent_time))
            else:
                args.logger.info("[{}] train loss: {:.3f} bpd: {:.3f} took {:.1f} seconds".format(epoch, train_loss_tmp_meter.avg, bits_per_dimensions(args, train_loss_tmp_meter.avg), spent_time))
            scheduler.step()
    else:
        # test
        args.logger.info("starting test")
        model.eval()
        
        for epoch in range(1, 1 + args.epochs):
            spent_time = time.time()
            z = prior.sample(args.display_size)
            with torch.no_grad():
                x = postprocess(args, model.decode(z))
            assert x.shape == (args.display_size, args.channels)
            
            if args.realnvp:
                plot_2D(args, x[:, 0], x[:, 1], name='[Test] toyplot_x')
            else:
                grids = torchvision.utils.make_grid(x, nrow=int(np.sqrt(args.display_size)))
                display = transforms.ToPILImage()(grids)
                display.save(os.path.join(args.path, args.result_dir, args.name, 'test.jpg'))
    
            spent_time = time.time() - spent_time
            args.logger.info("Test[{}] took {:.1f} seconds".format(epoch, spent_time))
        

if __name__ == '__main__':
    # set args
    parser = argparse.ArgumentParser(description='Glow')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='dataset')
    parser.add_argument(
        '--result_dir',
        type=str,
        default='results')
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default="ckpt")
    parser.add_argument(
        '--path',
        type=str,
        default='.')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100)
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-5)
    parser.add_argument(
    	'--warmup',
    	type=int,
    	default=5),
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64)
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4)
    parser.add_argument(
        '--test',
        action='store_true')
    parser.add_argument(
        '--save_period',
        type=int,
        default=5)
    parser.add_argument(
        '--name',
        type=str,
        default="train")
    parser.add_argument(
        '--ckpt',
        type=str,
        default='_')
    parser.add_argument(
        '--load',
        action='store_true')
    parser.add_argument(
        '--K',
        type=int,
        default=32)
    parser.add_argument(
        '--L',
        type=int,
        default=3)
    parser.add_argument(
        '--nn_channels',
        type=int,
        default=512)
    parser.add_argument(
        '--nn_kernel',
        type=int,
        default=3)
    parser.add_argument(
        '--channels',
        type=int,
        default=3)
    parser.add_argument(
        '--width',
        type=int,
        default=256)
    parser.add_argument(
        '--height',
        type=int,
        default=256)
    parser.add_argument(
        '--realnvp',
        action='store_true')
    parser.add_argument(
        '--dequantized',
        action='store_true')
    parser.add_argument(
    	'--bits',
    	type=int,
    	default=8)
    parser.add_argument(
        '--dequantized_levels',
        type=int,
        default=256)
    parser.add_argument(
		'--dequantized_bound',
		type=float,
		default=0.9)
    parser.add_argument(
        '--start_from_step',
        type=int,
        default=None)
    parser.add_argument(
    	'--display_size',
    	type=int,
    	default=64)
    
    args = parser.parse_args()

        
    main(args)