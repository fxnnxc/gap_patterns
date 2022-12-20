
"""

Finetuning the encoder

ResNet 18 
ResNet 34

"""

import torch.nn as nn
from omegaconf import OmegaConf
import argparse
from data import get_data
from encoders import get_encoder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np 
import os 
import time 
import torch 
import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser()
parser.add_argument("--data" ,type=str)
parser.add_argument("--encoder-name" ,type=str)
parser.add_argument("--run" ,type=str)
parser.add_argument("--img-size", type=int, default=84)
parser.add_argument("--epochs", type=int,)
parser.add_argument("--eval-freq", type=int,)
parser.add_argument("--save-freq", type=int,)
parser.add_argument("--seed", type=int)
parser.add_argument("--renew-last-layer", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
parser.add_argument("--freeze-pattern", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)

args = parser.parse_args()

flags = OmegaConf.load('configs/general.yaml')
# dump
for key in vars(args):
    setattr(flags, key, getattr(args, key))
torch.cuda.cudnn_enabled = False
np.random.seed(flags.seed)
torch.manual_seed(flags.seed)
torch.cuda.manual_seed(flags.seed)

flags.encoder = getattr(flags, flags.encoder_name)

# -- get class --
train_ds, valid_ds, num_classes, in_channels = get_data(flags)
flags.num_classes = num_classes
flags.in_channels = in_channels

# --- define dataset
train_dl, valid_dl = DataLoader(train_ds, shuffle=True, batch_size=flags.batch_size), DataLoader(valid_ds, shuffle=False, batch_size=flags.batch_size)

encoder =get_encoder(flags)


if flags.renew_last_layer:
    del encoder.fc
    print("new layer is added")
    encoder.add_module('fc', nn.Linear(flags.encoder.cnn_dim, num_classes))
encoder.to(flags.device)

# -- build objects --
model_optimizer = torch.optim.Adam(encoder.parameters(), lr=flags.encoder_lr)
model_lr_scheduler = None 
run_name = f"runs/encoder_train/{flags.data}/{flags.encoder_name}"

writer = SummaryWriter(run_name)
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)
print("--------\n flags")
for k,v  in flags.items():
    print(k, ":", v)
print("--------")


from tqdm import tqdm 

flags.start_time = time.time()
flags.vars = {}
flags.vars.best_accuracy = 0 

for epoch in range(flags.epochs):
    flags.vars.epoch = epoch
    # writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], epoch)
    
    pbar = tqdm(train_dl)
    flags.vars.sample_count = 0 
    flags.vars.running_loss = 0 
    encoder.train()
    if flags.freeze_pattern:
        pass 
    for x,y in pbar:
        x,y = x.to(flags.device), y.to(flags.device)
        y_hat = encoder(x)        
        loss = torch.nn.CrossEntropyLoss()(y_hat, y)
        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()
        
        flags.vars.sample_count += 1 
        flags.vars.running_loss += loss.item()
            
        duration = time.strftime("%H:%M:%S", time.gmtime(time.time()-flags.start_time))    
        writer.add_scalar("train/running_loss", flags.vars.running_loss/flags.vars.sample_count, epoch*len(train_dl)+flags.vars.sample_count)
        pbar.set_description(f"🚀 [INFO {run_name} Train : E:({flags.vars.epoch/flags.epochs:.2f}) D:({duration})]"+ \
                                f"| Loss {flags.vars.running_loss/flags.vars.sample_count:.6E}")    
    
    if epoch % flags.eval_freq == (flags.eval_freq -1) or epoch==flags.vars.epoch-1:
        encoder.eval()
        pbar = tqdm(valid_dl)
        eq = 0
        for k, (x, y) in (enumerate(pbar)):
            x = x.to(flags.device)
            y = y.to(flags.device)
            y_hat = encoder.forward(x).argmax(dim=-1)
            eq += (y == y_hat).sum()
        
        current_performance = (eq/len(valid_dl)).item()
        writer.add_scalar(f"eval/acc", current_performance, epoch)
        if current_performance > flags.vars.best_accuracy:
            flags.vars.best_accuracy = current_performance
            torch.save(encoder, os.path.join(run_name, f"model_best.pt"))
        OmegaConf.save(flags, os.path.join(run_name, "config.yaml"))
                
    if epoch % flags.save_freq == (flags.eval_freq -1):
        torch.save(encoder, os.path.join(run_name, f"model_{epoch}.pt"))
        