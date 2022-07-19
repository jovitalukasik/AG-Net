import torch
import numpy as np
import sys, os, random, argparse, pickle, tqdm
from datetime import datetime


from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import DataLoader

sys.path.insert(1, os.path.join(os.getcwd()))
from Generator import Generator_Decoder
from Measurements import Measurements
from Optimizer import Optimizer
from Checkpoint import Checkpoint
import Settings
import datasets.NASBench101 as NASBench101 
import datasets.NASBench201 as NASBench201

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

parser = argparse.ArgumentParser(description='Args for NAS latent space search experiments')
parser.add_argument("--device",                 type=str, default="cpu")
parser.add_argument('--trials',                 type=int, default=1, help='Number of trials')
parser.add_argument("--dataset",                type=str, default='NB201', help='Choice between NB101 and NB201')
parser.add_argument('--image_data',             type=str, default='cifar10_valid_converged', help='Only for NB201 relevant, choices between [cifar10_valid_converged, cifar100, ImageNet16-120]')
parser.add_argument("--name",                   type=str, default="Tabular_Search_wo_bp")
parser.add_argument("--weight_factor",          type=float, default=10e-3)
parser.add_argument("--num_init",               type=int, default=16)
parser.add_argument("--k",                      type=int, default=16)
parser.add_argument("--num_test",               type=int, default=1_00)
parser.add_argument("--ticks",                  type=int, default=30)
parser.add_argument("--tick_size",              type=int, default=16)  
parser.add_argument("--batch_size",             type=int, default=16)
parser.add_argument("--search_data",            type=int, default=310)
parser.add_argument("--saved_path",             type=str, help="Load pretrained Generator", default="state_dicts/NASBench201")
parser.add_argument("--saved_iteration",        type=str,  default='best' , help="Which iteration to load of pretrained Generator")
parser.add_argument("--seed",                   type=int, default=1)
parser.add_argument("--alpha",                  type=float, default=0.9)
parser.add_argument("--verbose",                type=str, default=True)

args = parser.parse_args()

##############################################################################
#
#                              Runfolder
#
##############################################################################
now = datetime.now()
runfolder = now.strftime("%Y_%m_%d_%H_%M_%S")
runfolder = f"NAS_Search_wo_bp_{args.dataset}/{args.image_data}/{runfolder}_{args.name}_{args.dataset}_{args.seed}"
runfolder = os.path.join(Settings.FOLDER_EXPERIMENTS, runfolder)
if not os.path.exists(runfolder):
    os.makedirs(runfolder)

# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(runfolder, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')



##############################################################################
#
#                            Definitions
#
##############################################################################
def sample_data(G,
    visited,
    measurements, 
    device,
    search_space,
    num = 100,
    latent_dim = 32, ):
    v = 0
    validity = 0
    possible_candidates = []
    with torch.no_grad():
        G.eval()
        # generate up to  4500 possible graphs
        for j in range(1_00,1_000,1_00):
            noise = (-6) * torch.rand(j, latent_dim) + 3
            graphs = G(noise.to(device))
            valid_sampled_data = measurements._compute_validity_score(graphs, search_space,  return_valid_spec=True)
            sampled_y = torch.stack([g.y for g in valid_sampled_data])
            sampled_hash_idx = dataset.query(sampled_y)  
            validity += len(valid_sampled_data)
            for i, idx in enumerate(sampled_hash_idx):
                if idx not in visited:
                    visited.append(idx)
                    possible_candidates.append(valid_sampled_data[i])
            v += j
            if len(possible_candidates) > num:
                random_shuffle = np.random.permutation(range(len(possible_candidates)))
                possible_candidates = [possible_candidates[i] for i in random_shuffle[:num]]
                break

    validity = validity/v
    return possible_candidates, visited, validity

def get_rank_weights(outputs, weight_factor):
    outputs_argsort = np.argsort(-np.asarray(outputs))
    ranks = np.argsort(outputs_argsort)
    return 1 / (weight_factor * len(outputs) + ranks)

def w_dataloader(train_data, weight_factor, batch_size, weighted_retraining=True):
    b_size = batch_size
    if weighted_retraining:
        weight_factor = weight_factor 
        outputs = np.array([graph.val_acc.item() for graph in train_data])
        weights = torch.tensor(get_rank_weights(outputs, weight_factor))
    
    else:
        weights = torch.ones(b_size)
    
    sampler = WeightedRandomSampler(
            weights, len(train_data))
    weighted_train_data = [(train_data[i],weights[i]) for i,w in enumerate(weights)]
    weighted_dataloader = DataLoader(weighted_train_data, sampler = sampler, batch_size = b_size, num_workers = 0, pin_memory = True)
    
    return weighted_dataloader

##############################################################################
#
#                              Training Loop
#
##############################################################################


##############################################################################
def train(
    real,
    b_size, 
    G, 
    weights,
    optimizer,
    alpha,
):  
    optimizer.zero_grad()

    noise = torch.randn(
        b_size, 32,
        device = real.x.device
        )
    nodes, edges = G.Decoder(noise)
    
    ln = G.node_criterion(nodes.view(-1,G.num_node_atts),  torch.argmax(real.x_binary, dim=1).view(b_size,-1).flatten())
    le = G.edge_criterion(edges, real.scores.view(b_size, -1))

    ln = torch.mean(ln.view(b_size, -1),1)
    le = torch.mean(le,1) 
    recon_loss = 2*(ln+ 0.5*le)

    acc = G.Predictor(real.y.reshape(b_size,-1))
    mse = G.acc_criterion(acc.view(-1), real.val_acc)

    err = (1-alpha)*recon_loss + alpha*mse
    err = torch.mean(err*weights.to(nodes.device))
    err.backward()
    # optimize
    optimizer.step()
    # return stats
    
    return (err.item(), 
            recon_loss.mean().item(), 
            mse.mean().item()
            )

def save_data(Dataset, train_data, path_measures, verbose=False):
    train_data = [Dataset.get_info_generated_graph(d, args.image_data) for d in train_data]
    torch.save(
        train_data,
        path_measures.format("all")
        )
    
    if verbose:
        top_5_acc = sorted([np.round(d.acc.item(),4) for d in train_data])[-5:]
        print('Top 5 acc after gradient method {}'.format(top_5_acc))

    return train_data

##############################################################################
def training_loop():
    search_data = args.search_data
    print('Amount of to be searched data: {}'.format(search_data))

    print(f"G on device: {next(G.parameters()).device}")

    print("Creating Dataset.")

    path_measures = os.path.join(
                trial_runfolder, "{}.data"
                )

    instances = 0
    tick_size = args.tick_size
    instances_total = args.ticks * tick_size
    with tqdm.tqdm(total=search_data, desc="Instances", unit="") as pbar:
        while True:
            if args.verbose:
                pbar.write("Starting Training Loop...")
            weighted_dataloader = w_dataloader(conditional_train_data, args.weight_factor, args.batch_size)
            upd = len(conditional_train_data)-pbar.n
            if upd > 0:
                pbar.update(upd)
            for batch, w in weighted_dataloader:

                G.train()

                batch = batch.to(args.device)

                b_size = batch.batch.max().item() + 1

                err, recon_loss, pred_loss =  train(
                    real = batch,
                    b_size = b_size,
                    G = G,
                    weights = w,
                    optimizer = optimizerG,
                    alpha = args.alpha, 
                )
                # measurements for saving
                measurements.add_measure("train_loss",      err,      instances)
                measurements.add_measure("recon_loss",      recon_loss,      instances)
                measurements.add_measure("pred_loss",      pred_loss,      instances)

                instances += b_size
                

            if instances >= instances_total:
                if args.verbose:
                    pbar.write("Starting evaluation of conditional generator for data size {}...".format(len(conditional_train_data)))

                visited = dataset.query(torch.stack([g.y for g in conditional_train_data]))  


                # Finished Training, now evaluate trained surrogate model for next samples
                test_data,_,_ = sample_data(G, visited, measurements, args.device, args.dataset, num=args.num_test)
                n = len(conditional_train_data)

                torch.save(test_data, 
                    path_measures.format("sampled_all_test_"+str(len(conditional_train_data)))
                    )

                # Sort given the surrogate model
                sort = sorted(test_data, key=lambda i:i.val_acc)

                for arch in sort[-args.k:]:

                    arch = Dataset.get_info_generated_graph(arch, args.image_data)

                    arch.to('cpu')
                    conditional_train_data.append(arch)

                checkpoint.save(len(conditional_train_data), only_model=True) 

                instances = 0
                tick_size += len(conditional_train_data)-n
                instances_total = args.ticks * tick_size

                save_data(Dataset, conditional_train_data, path_measures, verbose=False)
                
                
                if args.verbose:
                    top_5_acc = sorted([np.round(d.val_acc.item(),4) for d in conditional_train_data])[-5:]
                    print('grid search, top 5 val acc {}'.format(top_5_acc))

                
            if len(conditional_train_data) > search_data:
                output_name = 'round'
                trial = i
                print("Finished Grid Search with conditional prediction...", flush=True) 

                print('\n* Trial summary: results')

                results = []
                for query in range(args.num_init, len(conditional_train_data), args.k):
                    best_arch = sorted(conditional_train_data[:query], key=lambda i:i.val_acc)[-1]
                    test_acc = best_arch.acc.item()
                    val_acc = best_arch.val_acc.item()
                    results.append((query, val_acc, test_acc))

                path = os.path.join(runfolder, '{}_{}.pkl'.format(output_name, trial))
                print('\n* Trial summary: results')
                print(results)
                print('\n* Saving to file {}'.format(path))
                with open(path, 'wb') as f:
                    pickle.dump([results, conditional_train_data], f )
                    f.close()
                
                break


##############################################################################

# torch.autograd.set_detect_anomaly(True)
if __name__ == "__main__":
    for i in range(args.trials):
        if args.trials > 1:
            args.seed = i
        # Set random seed for reproducibility
        print("Search deterministically.")
        seed = args.seed
        print(f"Random Seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        trial_runfolder = os.path.join(runfolder, 'round_{}'.format(i))
        print(f"Experiment folder: {trial_runfolder}")
        if not os.path.exists(trial_runfolder):
            os.makedirs(trial_runfolder)
                    
        ##############################################################################
        #
        #                              Generator
        #
        ##############################################################################

        # load Checkpoint for pretrained Generator + MLP Predictor
        m = torch.load(os.path.join(args.saved_path, f"{args.saved_iteration}.model"), map_location=args.device) #pretrained_dict
        if args.dataset == 'NB101':
            m["nets"]["G"]["pars"]["data_config"]["regression_input"] = 56
        elif args.dataset == 'NB201':
            m["nets"]["G"]["pars"]["data_config"]["regression_input"] = 84
        m["nets"]["G"]["pars"]["data_config"]["regression_output"] = 1
        m["nets"]["G"]["pars"]["data_config"]["regression_hidden"] = m["nets"]["G"]["pars"]["data_config"]["regression_input"]
        m["nets"]["G"]["pars"]["acc_prediction"] = True
        m["nets"]["G"]["pars"]["list_all_lost"] = True
        G = Generator_Decoder(**m["nets"]["G"]["pars"]).to(args.device)
        state_dict = m["nets"]["G"]["state"]
        G_dict = G.state_dict()
        new_state_dict = {}
        for k,v in zip(G_dict.keys(), state_dict.values()):
            if k in G_dict :
                if v.size() == G_dict[k].size():
                    new_state_dict[k] = v
                else:
                    new_state_dict[k] = G_dict[k]
                    if "bias" in k:
                        new_state_dict[k][: v.size(0)] = v
                    else:
                        new_state_dict[k][: v.size(0),:v.size(1)] = v

        G_dict.update(new_state_dict) 
        G.load_state_dict(G_dict)
        
        ##############################################################################
        #
        #                              Dataset
        #
        ##############################################################################
        print("Creating Dataset.")
        if args.dataset=='NB101':
            NASBench = NASBench101
            Dataset = NASBench101.Dataset
            dataset =  NASBench101.Dataset(batch_size=args.batch_size, sample_size=args.num_init, only_prediction=True)
        elif args.dataset=='NB201':
            NASBench = NASBench201
            Dataset = NASBench.Dataset
            dataset =  NASBench.Dataset(batch_size=args.batch_size, sample_size=args.num_init, only_prediction=True, dataset=args.image_data)
        else:
            raise TypeError("Unknow Seach Space : {:}".format(args.dataset))

        conditional_train_data = dataset.train_data
        dataset.load_hashes()

        ##############################################################################
        #
        #                              Losses
        #
        ##############################################################################

        print("Initialize optimizers.")
        optimizerG = Optimizer(G, 'prediction').optimizer 

        ##############################################################################
        #
        #                              Checkpoint
        #
        ##############################################################################
        # Load Measurements
        measurements = Measurements(
                        G = G, 
                        batch_size = args.batch_size,
                        NASBench=NASBench
                    )

        chkpt_nets = {
            "G": G,
        }
        chkpt_optimizers = {
            "G": optimizerG,
        }
        checkpoint = Checkpoint(
            folder = trial_runfolder,
            nets = chkpt_nets,
            optimizers = chkpt_optimizers,
            measurements = measurements
        )
    



        training_loop()