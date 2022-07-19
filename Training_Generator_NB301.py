import os, random, argparse, tqdm, json, sys
from datetime import datetime
import torch
import numpy as np
from ConfigSpace.read_and_write import json as config_space_json_r_w

from Generator import Generator_Decoder
import datasets.NASBench301 as NASBench301
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
parser.add_argument("--dataset",                type=str, default='NB301')
parser.add_argument("--deterministic",          type=int, default=0)
parser.add_argument("--name",                   type=str, default="Train_Generator")
parser.add_argument("--ticks",                  type=int, default=10)
parser.add_argument("--tick_size",              type=int, default=10_0)
parser.add_argument("--b_size",                 type=int, default=256)
parser.add_argument("--node_embedding_dim",     type=int, default=32)
parser.add_argument("--graph_embedding_dim",    type=int, default=32)
parser.add_argument("--gnn_iteration_layers",   type=int, default=4)
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

if args.dataset=='NB301':
    data_config_path='configs/data_configs/NB301_configspace.json'
else:
    raise TypeError("Unknow Seach Space : {:}".format(args.dataset))

data_config = json.load(open(data_config_path, 'r'))

#Get Model configs
model_config_path='configs/model_configs/ag_configspace.json'
model_configspace = config_space_json_r_w.read(open(model_config_path, 'r').read())
model_config = model_configspace.get_default_configuration().get_dictionary()

model_config['batch_size']  = args.b_size
model_config['node_embedding_dim']  = args.node_embedding_dim
model_config['graph_embedding_dim']  = args.graph_embedding_dim
model_config['gnn_iteration_layers']  = args.gnn_iteration_layers
##############################################################################
#
#                              Generator
#
##############################################################################
G = Generator_Decoder(model_config=model_config, data_config=data_config, nb301=True).to(args.device)

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
if args.dataset=='NB301':
    NASBench = NASBench301
    dataset = NASBench301.Dataset(batch_size=model_config['batch_size'], generation=True)
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
    np.array([g.y_normal.cpu().numpy() for g in dataset.data])
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

    noise = torch.randn(
        b_size, model_config['graph_embedding_dim'], device=real.x_normal.device
    )
    nodes, edges = G.Decoder(noise)
    ln = G.node_criterion(nodes.view(-1, G.num_node_atts), torch.argmax(real.x_binary_normal, dim=1).view(b_size,-1).flatten())
    le = G.edge_criterion(edges, real.scores_normal.view(b_size,-1))

    recon_loss_normal = 2*(0.5*ln + 0.5*le)

    noise = torch.randn(
        b_size, model_config['graph_embedding_dim'], device=real.x_normal.device
    )
    nodes, edges = G.Decoder(noise)
    ln = G.node_criterion(nodes.view(-1, G.num_node_atts), torch.argmax(real.x_binary_reduce, dim=1).view(b_size,-1).flatten())
    le = G.edge_criterion(edges, real.scores_reduce.view(b_size,-1))

    recon_loss_reduce = 2*(0.5*ln + 0.5*le)
    err = recon_loss_normal + recon_loss_reduce

    err.backward()
    # optimize
    optimizer.step()
    # return stats

    return (err.item(),
            recon_loss_normal.item(), 
            recon_loss_reduce.item()
    ) 

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
                err, recon_loss_normal, recon_loss_reduce =  train(
                    real = real,
                    b_size = b_size,
                    G = G,
                    optimizer = optimizerG,
                )

                
                # measurements for saving
                measurements.add_measure("train_loss",         err,               instances)
                measurements.add_measure("recon_loss_normal",  recon_loss_normal, instances)
                measurements.add_measure("recon_loss_reduce",  recon_loss_reduce, instances)

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
                        prediction, 
                    )                      
                    checkpoint.save(instances)

                    instances_next_eval += args.tick_size
                

            if instances >= instances_total:
                # return
                break
            
##############################################################################

# torch.autograd.set_detect_anomaly(True)
if __name__ == "__main__":
    training_loop()