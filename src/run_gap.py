
from omegaconf import OmegaConf
import argparse
from data import get_data
from encoders import get_encoder
from gap_modules import get_model
from wrappers import get_wrapper
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np 
import os 
import time 
import torch 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data" ,type=str)
parser.add_argument("--encoder-name" ,type=str)
parser.add_argument("--model-name" ,type=str)
parser.add_argument("--run" ,type=str)
parser.add_argument("--img-size", type=int, default=84)
parser.add_argument("--cnn-dim", type=int,)
parser.add_argument("--epochs", type=int,)
parser.add_argument("--eval-freq", type=int,)
parser.add_argument("--seed", type=int)
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
flags.model = getattr(flags, flags.model_name)

# -- get class --
train_ds, valid_ds, num_classes, in_channels = get_data(flags)
flags.num_classes = num_classes
flags.in_channels = in_channels

train_dl, valid_dl = DataLoader(train_ds, shuffle=True, batch_size=flags.batch_size), DataLoader(valid_ds, shuffle=False, batch_size=flags.batch_size)
gap_module = get_model(flags).to(flags.device)
encoder =get_encoder(flags, gap_module).to(flags.device)
wrapper = get_wrapper('resnet', encoder, gap_module)

# -- build objects --
model_optimizer = torch.optim.Adam(encoder.parameters(), lr=flags.encoder_lr)
gap_optimizer = torch.optim.Adam(gap_module.parameters(), lr=flags.gap_lr)
model_lr_scheduler = None 
gap_lr_scheduler = None 

run_name = f"runs/gap_finetune/{flags.data}/{flags.model_name}/{flags.encoder_name}"
    
# ----
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
    for x,y in pbar:
        x,y = x.to(flags.device), y.to(flags.device)
        y_hat = wrapper.forward(x)        
        loss = torch.nn.CrossEntropyLoss()(y_hat, y)
        gap_optimizer.zero_grad()
        loss.backward()
        gap_optimizer.step()
        
        flags.vars.sample_count += 1 
        flags.vars.running_loss += loss.item()
            
        duration = time.strftime("%H:%M:%S", time.gmtime(time.time()-flags.start_time))    
        writer.add_scalar("train/running_loss", flags.vars.running_loss/flags.vars.sample_count, epoch*len(train_dl)+flags.vars.sample_count)
        pbar.set_description(f"🚀 [INFO {run_name} Train : E:({flags.vars.epoch/flags.epochs:.2f}) D:({duration})]"+ \
                                f"| Loss {flags.vars.running_loss/flags.vars.sample_count:.6E}")    
        break 
    
    if epoch % flags.eval_freq == (flags.eval_freq -1) or epoch==flags.vars.epoch-1:
        encoder.eval()
        pbar = tqdm(valid_dl)
        eq = 0
        for k, (x, y) in (enumerate(pbar)):
            x = x.to(flags.device)
            y = y.to(flags.device)
            y_hat = wrapper.forward(x).argmax(dim=-1)
            eq += (y == y_hat).sum()
        
        current_performance = (eq/len(valid_dl)).item()
        writer.add_scalar(f"eval/acc", current_performance, epoch)
        if current_performance > flags.vars.best_accuracy:
            flags.vars.best_accuracy = current_performance
            torch.save(encoder, os.path.join(run_name, f"model_best.pt"))
        torch.save(encoder, os.path.join(run_name, f"model_{epoch}.pt"))
        OmegaConf.save(flags, os.path.join(run_name, "config.yaml"))
        