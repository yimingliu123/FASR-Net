from option import args
from utils import mkExpDir
from dataset import dataloader
from model import FASR
from loss.loss import get_loss_dict
from trainer import Trainer
import math
import numpy as np
import os
import torch
import torch.nn as nn
import warnings
from tqdm import tqdm
import datetime
from torch.utils.tensorboard import SummaryWriter

today = datetime.date.today().strftime('%m%d')
tb = SummaryWriter(log_dir = "./runs/exp%s" % today)
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    ### make save_dir
    _logger = mkExpDir(args)

    ### dataloader of training set and testing set
    _dataloader = dataloader.get_dataloader(args) if (not args.test) else None

    ### device and model
    device = torch.device('cpu' if args.cpu else 'cuda')
    _model = FASR.FASR(args).to(device)
    if ((not args.cpu) and (args.num_gpu > 1)):
        _model = nn.DataParallel(_model, list(range(args.num_gpu)))

    ### loss
    _loss_all = get_loss_dict(args, _logger)

    ### trainer
    t = Trainer(args, _logger, _dataloader, _model, _loss_all)

    ### test / eval / train
    if (args.test):
        t.load(model_path=args.model_path)
        t.test()
