import torch
import numpy as np
import sys, os, random, argparse, pickle, tqdm, time
from datetime import datetime
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F

from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import DataLoader

sys.path.insert(1, os.path.join(os.getcwd()))
from Generator import Generator_Decoder
from Measurements import Measurements
from Optimizer import Optimizer, Loss
from Checkpoint import Checkpoint
import Settings
import datasets.NASBench301 as NASBench301
from datasets.NASBench301 import TENAS

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
parser.add_argument("--device",                 type=str, default="cuda:0")
parser.add_argument('--trials',                 type=int, default=1, help='Number of trials')
parser.add_argument("--dataset",                type=str, default='NB301')
parser.add_argument('--image_data',             type=str, default='cifar10', help='Choice between cifar10 and imagenet-1k')
parser.add_argument("--name",                   type=str, default="TENAS_Search")
parser.add_argument("--weight_factor",          type=float, default=10e-3)
parser.add_argument("--num_init",               type=int, default=16)
parser.add_argument("--k",                      type=int, default=16)
parser.add_argument("--num_test",               type=int, default=50)
parser.add_argument("--ticks",                  type=int, default=1)
parser.add_argument("--tick_size",              type=int, default=16)  
parser.add_argument("--batch_size",             type=int, default=16)
parser.add_argument("--search_data",            type=int, default=208)
parser.add_argument("--saved_path",             type=str, help="Load pretrained Generator", default="state_dicts/NASBench301")
parser.add_argument("--saved_iteration",        type=str,  default='best' , help="Which iteration to load of pretrained Generator")
parser.add_argument("--seed",                   type=int, default=1)
parser.add_argument("--alpha",                  type=float, default=0.9)
parser.add_argument("--verbose",                type=str, default=True)
parser.add_argument("--cifar_data_path",        type=str, help="path for Cifar 10 data", default='home/data/')
parser.add_argument("--config_data_path",        type=str, help="path for confid data", default='datasets/configs/')
args = parser.parse_args()

##############################################################################
#
#                              Runfolder
#
##############################################################################
now = datetime.now()
runfolder = now.strftime("%Y_%m_%d_%H_%M_%S")
runfolder = f"NAS_TENAS_Search_{args.dataset}/{args.image_data}/{runfolder}_{args.name}_{args.seed}"
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
    random_cell,
    visited,
    measurements, 
    device,
    search_space,
    num = 100,
    latent_dim = 32,
    normal=True ):
    i = 0
    validity = 0
    possible_candidates = []
    with torch.no_grad():
        G.eval()
        # generate up to  4500 possible graphs
        for j in range(1_00,1_000,1_00):
            noise = (-6) * torch.rand(j, latent_dim) + 3
            graphs = G(noise.to(device))
            cells = measurements._compute_validity_score(graphs, search_space,  return_valid_spec=True)
            validity += len(cells)
            sampled_data = []
            if normal:
                normal_cells = [random_cell[0] for _ in range(len(cells))]
                reduction_cells = cells
            else:
                normal_cells = cells
                reduction_cells = [random_cell[0] for _ in range(len(cells))]

            for g_n,g_r in zip(normal_cells, reduction_cells):
                if normal:
                    d =g_r
                    d.edge_index_normal = g_n.edge_index_normal
                    d.x_normal = g_n.x_normal
                    d.x_binary_normal = g_n.x_binary_normal
                    d.y_normal = g_n.y_normal
                    d.scores_normal = g_n.scores_normal
                    d.edge_index_reduce = g_r.edge_index
                    d.x_reduce = g_r.x
                    d.x_binary_reduce = g_r.x_binary
                    d.y_reduce = g_r.y
                    d.scores_reduce = g_r.scores
                else:
                    d =g_n
                    d.edge_index_normal = g_n.edge_index
                    d.x_normal = g_n.x
                    d.x_binary_normal = g_n.x_binary
                    d.y_normal = g_n.y
                    d.scores_normal = g_n.scores
                    d.edge_index_reduce = g_r.edge_index_reduce
                    d.x_reduce = g_r.x_reduce
                    d.x_binary_reduce = g_r.x_binary_reduce
                    d.y_reduce = g_r.y_reduce
                    d.scores_reduce = g_r.scores_reduce
                if hasattr(d, "x"):
                    del d.x
                if hasattr(d, "edge_index"):
                    del d.edge_index
                if hasattr(d, "y"):
                    del d.y
                if hasattr(d, "x_binary"):
                    del d.x_binary
                if hasattr(d, "g"):
                    del d.g
                if hasattr(d, "z"):
                    del d.z
                if hasattr(d, "scores"):
                    del d.scores
                
                sampled_data.append(d)
                
                for sample in sampled_data:
                    if str(sample.y_normal.detach().tolist()+sample.y_reduce.detach().tolist()) not in visited:
                        possible_candidates.append(sample)
                        visited[str(d.y_normal.detach().tolist()+d.y_reduce.detach().tolist())] = 1


            i += j

            if len(possible_candidates) == num:
                break
            elif len(possible_candidates) > num :
                random_shuffle = np.random.permutation(range(len(possible_candidates)))
                possible_candidates = [possible_candidates[i] for i in random_shuffle[:num]]
                break

    validity = validity/i
    return possible_candidates, visited, validity

def sample_one(G,
    measurements,
    device,
    search_space,
    num = 1,
    latent_dim = 32,
    normal=True
    ):
    i = 0
    validity = 0
    visited = {}
    possible_candidates = []
    with torch.no_grad():
        G.eval()
        # generate up to  4500 possible graphs
        for j in range(1_000,5_000,1_000):
            noise = torch.randn(j, latent_dim, device=device) 
            graphs = G(noise.to(device))
            cells = measurements._compute_validity_score(graphs, search_space,  return_valid_spec=True)
            validity += len(cells)
            for sample in cells:
                if str(sample.y.detach().tolist()) not in visited:
                    possible_candidates.append(sample)
                    visited[str(sample.y.detach().tolist())] = 1
            i += j

            if len(possible_candidates) >= num:
                random_shuffle = np.random.permutation(range(len(possible_candidates)))
                possible_candidates = [possible_candidates[i] for i in random_shuffle[:num]]
                break

    validity = validity/i
    for arch in possible_candidates:
        if normal:
            arch.edge_index_normal = arch.edge_index
            arch.x_normal = arch.x
            arch.x_binary_normal = arch.x_binary
            arch.y_normal = arch.y
            arch.scores_normal = arch.scores

        else:
            arch.edge_index_reduce = arch.edge_index
            arch.x_reduce = arch.x
            arch.x_binary_reduce = arch.x_binary
            arch.y_reduce = arch.y
            arch.scores_reduce = arch.scores

        if hasattr(arch, "x"):
                del arch.x
        if hasattr(arch, "edge_index"):
            del arch.edge_index
        if hasattr(arch, "y"):
                del arch.y
        if hasattr(arch, "x_binary"):    
                del arch.x_binary
        if hasattr(arch, "g"):
                del arch.g
        if hasattr(arch, "z"):
                del arch.z
        if hasattr(arch, "scores"):
                del arch.scores

    return possible_candidates, validity

def eval_data(random_cell, test_data, normal = True):
    candidates = [[] for _ in range(len(random_cell))]
    for idx in range(len(random_cell)):
        if normal:
            normal_cells = [random_cell[idx] for _ in range(len(test_data))]
            reduction_cells = test_data
        else:
            normal_cells = test_data
            reduction_cells = [random_cell[idx] for _ in range(len(test_data))]

        for g_n,g_r in zip(normal_cells, reduction_cells):
            d =g_n
            d.edge_index_normal = g_n.edge_index_normal
            d.x_normal = g_n.x_normal
            d.x_binary_normal = g_n.x_binary_normal
            d.y_normal = g_n.y_normal
            d.scores_normal = g_n.scores_normal
            d.edge_index_reduce = g_r.edge_index_reduce
            d.x_reduce = g_r.x_reduce
            d.x_binary_reduce = g_r.x_binary_reduce
            d.y_reduce = g_r.y_reduce
            d.scores_reduce = g_r.scores_reduce
            d.num_nodes=torch.tensor(d.x_normal.shape[0])
            if hasattr(d, "x"):
                del d.x
            if hasattr(d, "edge_index"):
                del d.edge_index
            if hasattr(d, "y"):
                del d.y
            if hasattr(d, "x_binary"):    
                del d.x_binary
            if hasattr(d, "g"):
                del d.g
            if hasattr(d, "z"):
                del d.z
            if hasattr(d, "scores"):
                del d.scores
            try:
                genotype = Dataset.get_genotype(d)
                d.genotype = genotype
                ntk = 0
                lr = 0
                for _ in range(3):
                    ntk_i, lr_i = Dataset.get_nb301_ntk_lr(tenas,genotype)
                    ntk += ntk_i
                    lr += lr_i
                ntk = ntk/3
                lr = lr/3
                d.ntk = torch.tensor([ntk])
                d.lr = torch.tensor([lr])
                candidates[idx].append(d)
            except:
                continue
    
    return candidates

def get_rank_weights(outputs, weight_factor):
    outputs_argsort = np.argsort(-np.asarray(outputs))
    ranks = np.argsort(outputs_argsort)
    return 1 / (weight_factor * len(outputs) + ranks)

def get_rank_weights(outputs, weight_factor, highest_best=True):
    if highest_best:
        outputs_argsort = np.argsort(-np.asarray(outputs))
    else:
        outputs_argsort = np.argsort(np.asarray(outputs))
    ranks = np.argsort(outputs_argsort)
    return 1 / (weight_factor * len(outputs) + ranks)

def w_dataloader(train_data, weight_factor, batch_size, weighted_retraining=True):
    b_size = batch_size
    if weighted_retraining:
        weight_factor = weight_factor
        outputs_lr = np.array([graph.lr.item() for graph in train_data])
        weights_lr = torch.tensor(get_rank_weights(outputs_lr, weight_factor, highest_best=True))
        outputs_ntk = np.array([graph.ntk.item() for graph in train_data])
        weights_ntk = torch.tensor(get_rank_weights(outputs_ntk, weight_factor, highest_best=False))
        weights = weights_lr+weights_ntk
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
    normal_cell, 
):  
    optimizer.zero_grad()

    if normal_cell:
        #loss for normal cell
        noise = torch.randn(
            b_size, 32,
            device = real.x_normal.device
            )

        nodes, edges = G.Decoder(noise)

        ln = G.node_criterion(nodes.view(-1,G.num_node_atts),  torch.argmax(real.x_binary_normal, dim=1).view(b_size,-1).flatten())
        le = G.edge_criterion(edges, real.scores_normal.view(b_size, -1))

        ln = torch.mean(ln.view(b_size, -1),1)
        le = torch.mean(le,1)
        recon_loss= 2*(ln+ 0.5*le)

    else:
        #loss for reduce cell
        noise = torch.randn(
            b_size, 32,
            device = real.x_normal.device
            )

        nodes, edges = G.Decoder(noise)

        ln = G.node_criterion(nodes.view(-1,G.num_node_atts),  torch.argmax(real.x_binary_reduce, dim=1).view(b_size,-1).flatten())
        le = G.edge_criterion(edges, real.scores_reduce.view(b_size, -1))

        ln = torch.mean(ln.view(b_size, -1),1)
        le = torch.mean(le,1)
        recon_loss = 2*(ln+ 0.5*le)

    err = torch.mean(recon_loss*weights.to(real.x_normal.device))
    err.backward()
    # optimize
    optimizer.step()
    # return stats

    return err.item()

def save_data(Dataset, train_data, path_measures, verbose=False):
    torch.save(
        np.array([i for i in train_data], dtype=object),
        path_measures.format("all")
        )
    return train_data


def get_summed_ranking_single(data):
    lr = np.argsort(np.argsort([-d.lr for d in data]))
    ntk = np.argsort(np.argsort([d.ntk for d in data]))

    ranking = lr + ntk
    idx = np.argsort(ranking)
    return [(ranking[i], data[i]) for i in idx]

def get_summed_ranking(data):
    # outputs sum of ranking for ntk and lr for each arch and the arch
    # the first is the best
    lr = []
    ntk = []

    for i in range(len(data)):
        lr_i = []
        ntk_i = []
        for arch in data[i]:
            lr_i.append(arch.lr)
            ntk_i.append(arch.ntk)
        ntk.append(ntk_i)
        lr.append(lr_i)
    if len(lr[0]):
        mean_lr = np.mean(lr,0)
        mean_ntk = np.mean(ntk,0)
    else:
        mean_lr = np.mean(lr,1)
        mean_ntk = np.mean(ntk,1)

    lr = np.argsort(np.argsort(-mean_lr))
    ntk = np.argsort(np.argsort(mean_ntk))

    ranking = ntk + lr
    idx = np.argsort(ranking)

    return [(ranking[i], data[0][i]) for i in idx]

##############################################################################
def training_loop():
    start = time.time()

    search_data = args.search_data
    print('Amount of to be searched data: {}'.format(search_data))

    print(f"G on device: {next(G.parameters()).device}")

    print("Creating Dataset.")

    path_measures_reduction = os.path.join(
                trial_runfolder, "{}_reduction.data"
                )

    path_measures_normal= os.path.join(
                trial_runfolder, "{}_normal.data"
                )

    instances = 0
    tick_size = args.tick_size
    instances_total = args.ticks * tick_size

    # sample first reduction cell at random:
    reduction_cell,_ = sample_one(G, measurements, args.device, args.dataset, num=3, normal=False)

    # Search for normal_cell  to this reduction_cell for 300 queries
    with tqdm.tqdm(total=search_data, desc="Instances", unit="") as pbar:
        while True:
            if args.verbose:
                pbar.write("Starting Training Loop...")
            for arch in conditional_train_data:
                arch.num_nodes = torch.tensor([arch.x_normal.shape[0]])
            weighted_dataloader = w_dataloader(conditional_train_data, args.weight_factor, args.batch_size)
            upd = len(conditional_train_data)-pbar.n
            if upd > 0:
                pbar.update(upd)
            for batch, w in weighted_dataloader:

                G.train()

                batch = batch.to(args.device)

                b_size = batch.batch.max().item() + 1

                err =  train(
                    real = batch,
                    b_size = b_size,
                    G = G,
                    weights = w,
                    optimizer = optimizerG,
                    normal_cell = True
                )
                # measurements for saving
                measurements.add_measure("train_loss",      err,      instances)

                instances += b_size
                

            if instances >= instances_total:
                if args.verbose:
                    pbar.write("Starting TENAS search of conditional generator for data size {}...".format(len(conditional_train_data)))

                visited = {}
                for d in conditional_train_data:
                    h = str(d.y_normal.detach().tolist()+d.y_reduce.detach().tolist())
                    visited[h] = 1

                # Finished Training, now evaluate KN and NLR for next samples
                test_data,_,_ = sample_data(G, reduction_cell, visited, measurements, args.device, args.dataset, num=args.num_test, normal=False)                
                if test_data == []:
                    break
                test_data = eval_data(reduction_cell, test_data, normal = False)
                torch.save(np.array([i[0] for i in test_data], dtype=object),
                    path_measures_normal.format("sampled_all_test_"+str(len(conditional_train_data)))
                    )

                ranking_sum = get_summed_ranking(test_data)
                n = len(conditional_train_data)
                for ranking, arch in ranking_sum[:args.k]:
                    arch.to('cpu')
                    conditional_train_data.append(arch)

                instances = 0
                tick_size += len(conditional_train_data) - n 
                instances_total = args.ticks * tick_size

                save_data(Dataset, conditional_train_data, path_measures_normal, verbose=False)
                

                
            if len(conditional_train_data) > search_data//2:
                output_name = 'round'
                trial = i
                print("Finished Grid Search with for normal cell...", flush=True)

                print('\n* Trial summary: results')

                results = []
                ranking_sum = get_summed_ranking_single(conditional_train_data)

                best_arch = ranking_sum[0][1] # in terms of ntk and lr
                results.append(best_arch)

                path = os.path.join(runfolder, '{}_{}_normal_cell.pkl'.format(output_name, trial))
                print('\n* Trial summary: results')
                print(results)
                print('\n* Saving to file {}'.format(path))
                with open(path, 'wb') as f:
                    pickle.dump([results, conditional_train_data], f )
                    f.close()

                break

    # Start search for reduction cell given best normal cell
    normal_cell = best_arch

    instances = 0
    tick_size = len(conditional_train_data)
    instances_total = args.ticks * tick_size

    with tqdm.tqdm(total=search_data, desc="Instances", unit="") as pbar:
        while True:
            if args.verbose:
                pbar.write("Starting Training Loop...")
            for arch in conditional_train_data:
                arch.num_nodes = torch.tensor([arch.x_normal.shape[0]])
            weighted_dataloader = w_dataloader(conditional_train_data, args.weight_factor, args.batch_size)
            upd = len(conditional_train_data)-pbar.n
            if upd > 0:
                pbar.update(upd)
            for batch, w in weighted_dataloader:

                G.train()

                batch = batch.to(args.device)

                b_size = batch.batch.max().item() + 1

                err =  train(
                    real = batch,
                    b_size = b_size,
                    G = G,
                    weights = w,
                    optimizer = optimizerG,
                    normal_cell = False
                )
                # measurements for saving
                measurements.add_measure("train_loss",      err,      instances)

                instances += b_size
                

            if instances >= instances_total:
                if args.verbose:
                    pbar.write("Starting TENAS search of conditional generator for data size {}...".format(len(conditional_train_data)))

                visited = {}
                for d in conditional_train_data:
                    h = str(d.y_normal.detach().tolist()+d.y_reduce.detach().tolist())
                    visited[h] = 1

                # Finished Training, now evaluate KN and NLR for next samples
                test_data,_,_ = sample_data(G, [normal_cell], visited, measurements, args.device, args.dataset, num=args.num_test, normal=True)                
                if test_data == []:
                    break

                test_data = eval_data([normal_cell], test_data, normal = True)
                torch.save(np.array([i[0] for i in test_data], dtype=object),
                    path_measures_reduction.format("sampled_all_test_"+str(len(conditional_train_data)))
                    )

                ranking_sum = get_summed_ranking_single(test_data[0])
                n = len(conditional_train_data)
                for ranking, arch in ranking_sum[:args.k]:
                    arch.to('cpu')
                    conditional_train_data.append(arch)

                instances = 0
                tick_size += len(conditional_train_data) - n 
                instances_total = args.ticks * tick_size

                save_data(Dataset, conditional_train_data, path_measures_reduction, verbose=False)
                

                
            if len(conditional_train_data) > search_data:
                output_name = 'round'
                trial = i
                print("Finished Grid Search with for reduction cell...", flush=True)

                print('\n* Trial summary: results')

                results = []
                ranking_sum = get_summed_ranking_single(conditional_train_data)

                best_arch = ranking_sum[0][1] # in terms of ntk and lr
                best_arch.num_nodes = torch.tensor([best_arch.x_reduce.shape[0]])
                results.append(best_arch)

                path = os.path.join(runfolder, '{}_{}_reduction_cell.pkl'.format(output_name, trial))
                print('\n* Trial summary: results')
                print(results)
                print('\n* Saving to file {}'.format(path))
                with open(path, 'wb') as f:
                    pickle.dump([np.array(results, dtype=object), np.array([i for i in conditional_train_data], dtype=object)], f )
                    f.close()

                stop = time.time()
                print(f"Training time: {stop - start}s")

                # save trainint time line input
                with open(os.path.join(runfolder, 'time_input.txt'), 'a') as f:
                    f.write(str(stop - start)+' seconds')

                genotype = Dataset.get_genotype(best_arch)
                with open(os.path.join(runfolder, 'genotype.txt'), 'a') as f:
                    f.write(str(genotype))
                break

##############################################################################

# torch.autograd.set_detect_anomaly(True)
if __name__ == "__main__":
    for i in range(args.trials):
        if args.trials > 1:
            args.seed = i
        args.seed = random.randint(1, 100000)
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
        m["nets"]["G"]["pars"]["acc_prediction"] = False
        m["nets"]["G"]["pars"]["list_all_lost"] = True
        G = Generator_Decoder(**m["nets"]["G"]["pars"]).to(args.device)
        state_dict = m["nets"]["G"]["state"]
        G.load_state_dict(state_dict)

        ##############################################################################
        #
        #                              Losses
        #
        ##############################################################################
        print("Initialize optimizers.")
        optimizerG = Optimizer(G, 'prediction').optimizer 
        ##############################################################################
        #
        #                              Dataset
        #
        ##############################################################################
        print("Creating Dataset.")
        NASBench = NASBench301
        Dataset = NASBench301.Dataset
        dataset =  NASBench301.Dataset(batch_size=args.batch_size, sample_size=args.num_init, only_prediction=True)
        tenas = TENAS(dataset=args.image_data, data_path=args.cifar_data_path, config_path=args.config_data_path, seed=seed)

        conditional_train_data = dataset.train_data
        print('Get NTK and LR')
        for arch in tqdm.tqdm(conditional_train_data):
            genotype = Dataset.get_genotype(arch)
            arch.genotype = genotype
            ntk = 0
            lr = 0
            for _ in range(3):
                ntk_i, lr_i = Dataset.get_nb301_ntk_lr(tenas,genotype)
                ntk += ntk_i
                lr += lr_i
            ntk = ntk/3
            lr = lr/3
            arch.ntk = torch.tensor([ntk])
            arch.lr = torch.tensor([lr])



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