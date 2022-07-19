import os, random, argparse, tqdm, json, sys
from datetime import datetime
import torch
import numpy as np
from ConfigSpace.read_and_write import json as config_space_json_r_w

from Generator import Generator_Decoder
import datasets.NASBench101 as NASBench101 
import datasets.NASBench201 as NASBench201
import datasets.NASBenchNLP as NASBenchNLP
from Measurements import Measurements
from Optimizer import Optimizer
from Checkpoint import Checkpoint
import Settings


##############################################################################
#
#                              Arguments
#
##############################################################################
DEBUGGING = False
if DEBUGGING:
    print("!"*28)
    print("!!!! WARNING: DEBUGGING !!!!")
    print("!"*28)
    print()

parser = argparse.ArgumentParser()
parser.add_argument("--device",                 type=str, default="cpu")
parser.add_argument("--dataset",                type=str, default='NB101', help='Choice between NB101, NB201 and NBNLP')
parser.add_argument("--deterministic",          type=int, default=0)
parser.add_argument("--name",                   type=str, default="Train_Generator")
parser.add_argument("--ticks",                  type=int, default=10)
parser.add_argument("--tick_size",              type=int, default=10_0)
parser.add_argument("--criterion",              type=str, default="decoding", help= "Choices between decoding, MLP_decoding")

args = parser.parse_args()

if DEBUGGING:
    args.device = "cpu"

print(f"Experiment: {args.name}")
print(args)

if args.deterministic:
    print("Training deterministically.")
    # Set random seed for reproducibility
    seed = 999
    print(f"Random Seed: {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
##############################################################################
#
#                              Runfolder
#
##############################################################################
    
now = datetime.now()
runfolder = now.strftime("%Y_%m_%d_%H_%M_%S")
runfolder = f"{runfolder}_{args.name}_{args.dataset}"
runfolder = os.path.join(Settings.FOLDER_EXPERIMENTS, runfolder)
os.makedirs(runfolder)

# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(runfolder, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')

print(f"Experiment folder: {runfolder}")

##############################################################################
#
#                           Configs
#
##############################################################################

if args.dataset=='NB101':
    data_config_path='configs/data_configs/NB101_configspace.json'
elif args.dataset=='NB201':
    data_config_path='configs/data_configs/NB201_configspace.json'
elif args.dataset=='NBNLP':
    data_config_path='configs/data_configs/NBNLP_configspace.json'
else:
    raise TypeError("Unknow Seach Space : {:}".format(args.dataset))

data_config = json.load(open(data_config_path, 'r'))

#Get Model configs
model_config_path='configs/model_configs/ag_configspace.json'
model_configspace = config_space_json_r_w.read(open(model_config_path, 'r').read())
model_config = model_configspace.get_default_configuration().get_dictionary()

##############################################################################
#
#                              Generator
#
##############################################################################
G = Generator_Decoder(model_config=model_config, data_config=data_config).to(args.device)

print("Generator parameters:")
print(G.pars)
print(G)
print()

##############################################################################
#
#                              Optimizer
#
##############################################################################


print("Initialize optimizers.")

# Setup Adam optimizers for G 
optimizerG = Optimizer(G, args.criterion).optimizer 

##############################################################################
#
#                              Dataset
#
##############################################################################

print("Creating Dataset.")
if args.dataset=='NB101':
    NASBench = NASBench101
    dataset = NASBench101.Dataset(batch_size=model_config['batch_size'])
elif args.dataset=='NB201':
    NASBench = NASBench201
    dataset = NASBench201.Dataset(batch_size=model_config['batch_size'])
elif args.dataset=='NBNLP':
    NASBench = NASBenchNLP
    dataset = NASBenchNLP.Dataset(batch_size=model_config['batch_size'])
else:
    raise TypeError("Unknow Seach Space : {:}".format(args.dataset))


# Create the dataset
print(f"Dataset size: {dataset.length}")

##############################################################################
#
#                              Measurements
#
##############################################################################

print("Initialize measurements.")

measurements = Measurements(
    G = G, 
    batch_size = model_config['batch_size'],
    NASBench = NASBench
)
measurements.set_fid_real_stats(
    np.array([g.y.cpu().numpy() for g in dataset.data])
)

##############################################################################
#
#                              Checkpoint
#
##############################################################################
chkpt_nets = {
    "G": G,
}

chkpt_optimizers = {
    "G": optimizerG,
}

checkpoint = Checkpoint(
    folder = runfolder,
    nets = chkpt_nets,
    optimizers = chkpt_optimizers,
    measurements = measurements
)


##############################################################################
#
#                              Training Loop
#
##############################################################################

print(f"G on device: {next(G.parameters()).device}")

##############################################################################
def train(
    real,
    b_size,
    G,
    optimizer,
    ):

    optimizer.zero_grad()

    generated, err, mse = G.loss(real, b_size)


    err.backward()
    # optimize
    optimizer.step()
    # return stats

    return err.item(), 

##############################################################################


##############################################################################

def training_loop():
    print("Starting Training Loop...", flush=True)

    instances = 0
    instances_next_eval = args.tick_size
    instances_total = args.ticks * args.tick_size

    with tqdm.tqdm(total=instances_total, desc="Instances", unit="") as pbar:
        while True:
            for batch in dataset.dataloader:
                ############################
                
                G.train()

                batch.to(args.device)
                real = batch
                b_size = real.batch.max().item() + 1

                
                instances += b_size

                ### Training step for G ###
                err =  train(
                    real = real,
                    b_size = b_size,
                    G = G,
                    optimizer = optimizerG,
                )

                
                # measurements for saving
                measurements.add_measure("train_loss",      err,      instances)

                pbar.update(b_size)

                
                if instances >= instances_next_eval or instances >= instances_total:

                    ############################
                    #  Evaluation step
                    ############################
                    prediction = False
                    
                    measurements.measure(
                        instances,
                        args.dataset,
                        args.device,
                        prediction
                    )                      
                    checkpoint.save(instances)

                    instances_next_eval += args.tick_size
                

            if instances >= instances_total:
                break
            
##############################################################################

# torch.autograd.set_detect_anomaly(True)
if __name__ == "__main__":
    training_loop()