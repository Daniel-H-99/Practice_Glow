import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader 
from dataloader import get_loader
from utils import AverageMeter, save, load, tensor_2Dplot
from model import Glow, RealNVP
from tqdm import tqdm
import logging
import time
from PIL import Image

def preprocess(args, data):
    if args.realnvp:
        data = torch.cat(data)
        data = data.unsqueeze(2).unsqueeze(3)
    else:
        data = data[0]
    return data

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
    
    if not os.path.isdir('./ckpt'):
        os.mkdir('./ckpt')
    if not os.path.isdir('./results'):
        os.mkdir('./results')    
    if not os.path.isdir(os.path.join('./ckpt', args.name)):
        os.mkdir(os.path.join('./ckpt', args.name))
    if not os.path.isdir(os.path.join('./results', args.name)):
        os.mkdir(os.path.join('./results', args.name))
    if not os.path.isdir(os.path.join('./results', args.name, "log")):
        os.mkdir(os.path.join('./results', args.name, "log"))

    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler = logging.FileHandler("results/{}/log/{}.log".format(args.name, time.strftime('%c', time.localtime(time.time()))))
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
        transform_train = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        dataset = torchvision.datasets.CIFAR10(root=args.datadir, train=True, download=True, transform=transform_train)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 2. setup
    
    args.logger.info("setting up...")
    args.pixels = args.channels * args.width * args.height
    model = RealNVP(args) if args.realnvp else Glow(args)
    model.to(args.device)
    prior = PriorDistribution(args, torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(args.pixels).to(args.device), torch.eye(args.pixels).to(args.device)))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    if args.load:
        model.load_state_dict(load(args, args.ckpt))
        
    # 3. train / test
    
    if not args.test:
        # train
        args.logger.info("starting training")
        train_loss_meter = AverageMeter(name="Loss", save_all=True, save_dir=os.path.join('results', args.name), x_label="epoch")
        steps = 0
        for epoch in range(1, 1 + args.epochs):
            spent_time = time.time()
            model.train()
            train_loss_tmp_meter = AverageMeter()
            for data in tqdm(loader):
                if args.start_from_step is not None:
                    if steps < args.start_from_step:
                        steps += 1
                        continue
                optimizer.zero_grad()
                data = preprocess(args, data)
                batch = data.shape[0]
                z, loss = model(data.to(args.device))
                assert z.shape == (batch, args.pixels)
                loss += prior.mean_log_prob(z)
                loss.backward()
                optimizer.step()
                
                train_loss_tmp_meter.update(loss, weight=batch)
                steps += 1
                
                if steps % args.save_period == 0:
                    model.eval()
                    spent_time = time.time()
                    save(args, "epoch_{}".format(epoch), model.state_dict())
                    if args.realnvp:
                        datas = torch.zeros(0)
                        zs = torch.zeros(0)
                        xs = torch.zeros(0)

                        for data in tqdm(loader):
                            data = torch.cat(data)
                            data = preprocess(args, data)
                            batch = data.shape[0]

                            with torch.no_grad():
                                z, _ = model(data.to(args.device))
                                x = model.decode(z)

                            data = data.squeeze()
                            z = z.squeeze()
                            x = x.squeeze()
                            assert z.shape == (batch, args.pixels)
                            assert x.shape == (batch, args.channels)
                            datas = torch.cat((datas, data))
                            zs = torch.cat((zs, z.cpu()))
                            xs = torch.cat((xs, x.cpu()))
                    else:
                        z = prior.sample(args.batch_size)
                        x = model.decode(z)
                        assert x.shape == (args.batch_size, args.channels, args.width, args.height)
                        
                        for i in range(x.shape[0]):
                            img = transforms.ToPILImage()(x[i])
                            img.save(os.path.join('./results', args.name, 'epoch[{}]-{}.jpg'.format(epoch, i + 1)))
                            
                    if args.realnvp:
                        tensor_2Dplot(datas, save_dir=os.path.join('results', args.name), name='[{}]toyplot_data'.format(epoch))                    
                        tensor_2Dplot(zs, save_dir=os.path.join('results', args.name), name='[{}]toyplot_z'.format(epoch))
                        tensor_2Dplot(xs, save_dir=os.path.join('results', args.name), name='[{}]toyplot_x'.format(epoch))
                    
                    spent_time = time.time() - spent_time
                    args.logger.info("[{}] plot recorded, took {} seconds".format(steps, spent_time))
                    args.logger.info(str(model.parameters()))
                    train_loss_meter.save()
                
            train_loss_meter.update(train_loss_tmp_meter.avg)
            spent_time = time.time() - spent_time
            args.logger.info("[{}] train loss: {:.3f} took {:.1f} seconds".format(epoch, train_loss_tmp_meter.avg, spent_time))
            
            # validation
            if epoch % args.save_period == 0:
                pass
#                 model.eval()
#                 spent_time = time.time()
#                 save(args, "epoch_{}".format(epoch), model.state_dict())
#                 if args.realnvp:
#                     datas = torch.zeros(0)
#                     zs = torch.zeros(0)
#                     xs = torch.zeros(0)
                    
#                     for data in tqdm(loader):
#                         data = torch.cat(data)
#                         data = preprocess(args, data)
#                         batch = data.shape[0]

#                         with torch.no_grad():
#                             z, _ = model(data.to(args.device))
#                             x = model.decode(z)

#                         data = data.squeeze()
#                         z = z.squeeze()
#                         x = x.squeeze()
#                         assert z.shape == (batch, args.pixels)
#                         assert x.shape == (batch, args.channels)
#                         datas = torch.cat((datas, data))
#                         zs = torch.cat((zs, z.cpu()))
#                         xs = torch.cat((xs, x.cpu()))
#                 else:
#                     z = prior.sample(args.batch_size)
#                     x = model.decode(z)
#                     assert x.shape == (args.batch_size, args.channels, args.width, args.height)
#                     img_array = x.transpose(1, 2).transpose(2, 3).squeeze().numpy()
#                     for i in range(x.shape[0]):
#                         img = Image.fromarray(img_array[i])
#                         im.save(os.path.join(args.result, 'epoch[{}]-{}'.format(epoch, i + 1)))
                        
                    
#                 if args.realnvp:
#                     tensor_2Dplot(datas, save_dir=os.path.join('results', args.name), name='[{}]toyplot_data'.format(epoch))                    
#                     tensor_2Dplot(zs, save_dir=os.path.join('results', args.name), name='[{}]toyplot_z'.format(epoch))
#                     tensor_2Dplot(xs, save_dir=os.path.join('results', args.name), name='[{}]toyplot_x'.format(epoch))
                    
#                 spent_time = time.time() - spent_time
#                 args.logger.info("[{}] plot recorded, took {} seconds".format(epoch, spent_time))
#                 args.logger.info(str(model.parameters()))
#                 train_loss_meter.save()
    else:
        pass
        # test
#         args.logger.info("starting test")
#         test_loader = get_loader(src['test'], tgt['test'], src_vocab, tgt_vocab, batch_size=args.batch_size)
#         pred_list = []
#         model.eval()
        
#         for src_batch, tgt_batch in test_loader:
#             #src_batch: (batch x source_length)
#             src_batch = torch.Tensor(src_batch).long().to(args.device)
#             batch = src_batch.shape[0]           
#             pred_batch = torch.zeros(batch, 1).long().to(args.device)
#             pred_mask = torch.zeros(batch, 1).bool().to(args.device)    # mask whether each sentece ended up
            
#             with torch.no_grad():
#                 for _ in range(args.max_length):
#                     pred = model(src_batch, pred_batch)   # (batch x length x tgt_vocab_size)
#                     pred[:, :, pad_idx] = -1              # ignore <pad>
#                     pred = pred.max(dim=-1)[1][:, -1].unsqueeze(-1)   # next word prediction: (batch x 1)
#                     pred = pred.masked_fill(pred_mask, 2).long()      # fill out <pad> for ended sentences
#                     pred_mask = torch.gt(pred.eq(1) + pred.eq(2), 0)
#                     pred_batch = torch.cat([pred_batch, pred], dim=1)
#                     if torch.prod(pred_mask) == 1:
#                         break
                        
#             pred_batch = torch.cat([pred_batch, torch.ones(batch, 1).long().to(args.device) + pred_mask.long()], dim=1)   # close all sentences
#             pred_list += seq2sen(pred_batch.cpu().numpy().tolist(), tgt_vocab)
            
#         with open('results/pred.txt', 'w', encoding='utf-8') as f:
#             for line in pred_list:
#                 f.write('{}\n'.format(line))

#         os.system('bash scripts/bleu.sh results/pred.txt multi30k/test.de.atok')


if __name__ == '__main__':
    # set args
    parser = argparse.ArgumentParser(description='Glow')
    parser.add_argument(
        '--datadir',
        type=str,
        default='dataset')
    parser.add_argument(
        '--epochs',
        type=int,
        default=100)
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-5)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=75)
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
        '--ckpt_dir',
        type=str,
        default="ckpt")
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
        '--dequantized_levels',
        type=int,
        default=256)
    parser.add_argument(
        '--start_from_step',
        type=int,
        default=None)
    
    args = parser.parse_args()

        
    main(args)