import torch
import numpy as np
import sys, os, random, argparse, pickle, tqdm
from datetime import datetime
import xgboost as xgb

from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import DataLoader

from Generator import Generator_Decoder
from Measurements import Measurements
from Optimizer import Optimizer
from Checkpoint import Checkpoint
import Settings
import datasets.NASBench301 as NASBench301

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
parser.add_argument('--dataset',                type=str, default='NB301')
parser.add_argument('--image_data',             type=str, default='cifar10', help='Only for NB201 relevant, choices between [cifar10_valid_converged, cifar100, ImageNet16-120]')
parser.add_argument("--name",                   type=str, default="Surrogate_Search_XGB")
parser.add_argument("--weight_factor",          type=float, default=10e-3)
parser.add_argument("--num_init",               type=int, default=16)
parser.add_argument("--k",                      type=int, default=16)
parser.add_argument("--num_test",               type=int, default=1_00)
parser.add_argument("--ticks",                  type=int, default=1)
parser.add_argument("--tick_size",              type=int, default=16)
parser.add_argument("--batch_size",             type=int, default=16)
parser.add_argument("--search_data",            type=int, default=304)
parser.add_argument("--saved_path",             type=str, help="Load pretrained Generator", default="state_dicts/NASBench301")
parser.add_argument("--saved_iteration",        type=str, default="best", help="Which iteration to load of pretrained Generator")
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
runfolder = f"NAS_Search_XGB_{args.dataset}/surrogate_search/{args.search_data}/reduce/{runfolder}_reduce_{args.name}_{args.dataset}_{args.seed}"
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
    normal=True,
    ):
    i = 0
    validity = 0
    possible_candidates = []
    with torch.no_grad():
        G.eval()
        # generate up to  4500 possible graphs
        for j in range(1_000,5_000,1_000):
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
                    del d.edge_index
                    del d.y
                    del d.x_binary
                    del d.g
                    del d.z
                    del d.scores
                sampled_data.append(d)

            for sample in sampled_data:
                if str(sample.y_normal.detach().tolist()+sample.y_reduce.detach().tolist()) not in visited:
                    sample.acc = sample.val_acc
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
            noise = (-6) * torch.rand(j, latent_dim) + 3
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
            del arch.edge_index
            del arch.y
            del arch.x_binary
            del arch.g
            del arch.z
            del arch.scores

    return possible_candidates, validity

def get_rank_weights(outputs, weight_factor):
    outputs_argsort = np.argsort(-np.asarray(outputs))
    ranks = np.argsort(outputs_argsort)
    return 1 / (weight_factor * len(outputs) + ranks)

def w_dataloader(train_data, weight_factor, batch_size, weighted_retraining=True):
    b_size = batch_size
    if weighted_retraining:
        weight_factor = weight_factor
        outputs = np.array([graph.acc.item() for graph in train_data])
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
    normal_cell,
):

    noise = torch.randn(
        b_size, 32,
        device = real.x_normal.device
        )

    nodes, edges = G.Decoder(noise)
    if normal_cell:
        ln = G.node_criterion(nodes.view(-1,G.num_node_atts),  torch.argmax(real.x_binary_normal, dim=1).view(b_size,-1).flatten())
        le = G.edge_criterion(edges, real.scores_normal.view(b_size, -1))

    else:
        ln = G.node_criterion(nodes.view(-1,G.num_node_atts),  torch.argmax(real.x_binary_reduce, dim=1).view(b_size,-1).flatten())
        le = G.edge_criterion(edges, real.scores_reduce.view(b_size, -1))


    ln = torch.mean(ln.view(b_size, -1),1)
    le = torch.mean(le,1)
    recon_loss= 2*(ln+ 0.5*le)

    err = torch.mean((1-alpha)*recon_loss *weights.to(recon_loss.device))
    optimizer.zero_grad()

    err.backward()
    # optimize
    optimizer.step()
    # return stats

    return (err.item(),
            recon_loss.mean().item()
            )
def train_tree(train_data, normal_cell):
    
    params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "booster": "gbtree",
            "max_depth": 6,
            "min_child_weight": 1,
            "colsample_bytree": 1,
            "learning_rate": 0.3,
            "colsample_bylevel": 1,
        }

    if normal_cell:    
    # encode train data to adajcency one hot:
        encodings = np.array([arch.y_normal.numpy() for arch in train_data])
    else:    
        encodings = np.array([arch.y_reduce.numpy() for arch in train_data])

    ytrain = np.array([arch.acc.numpy() for arch in train_data])
    # normalize accuracies
    mean = np.mean(ytrain)
    std = np.std(ytrain)
    train_data = xgb.DMatrix(encodings, label=((ytrain - mean) / std))
    bst = xgb.train(params, train_data, num_boost_round=500)    
    # predict
    train_pred = np.squeeze(bst.predict(train_data))
    train_error = np.mean(abs(train_pred - ytrain))
    print("RMSE: %f" % (train_error))
    return bst

def eval_tree(bst, test_data, normal_cell):

    if normal_cell:    
        encodings = np.array([arch.y_normal.numpy() for arch in test_data])
    else:    
        encodings = np.array([arch.y_reduce.numpy() for arch in test_data])

    tree_test_data = xgb.DMatrix(encodings)
    preds = bst.predict(tree_test_data)    

    for i, arch in enumerate(test_data):
        arch.acc = torch.FloatTensor([preds[i]])

    return test_data

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

    path_measures_normal = os.path.join(
                trial_runfolder, "{}_normal.data"
                )

    instances = 0
    tick_size = args.tick_size
    instances_total = args.ticks * tick_size

    reduce_cell,_ = sample_one(G, measurements, args.device, args.dataset, num=1, normal=False)

    with tqdm.tqdm(total=search_data//2, desc="Instances", unit="") as pbar:
        while True:
            weighted_dataloader = w_dataloader(conditional_train_data, args.weight_factor, args.batch_size)
            upd = len(conditional_train_data)-pbar.n
            if upd > 0:
                pbar.update(upd)
            for batch, w in weighted_dataloader:

                G.train()

                batch = batch.to(args.device)

                b_size = batch.batch.max().item() + 1

                err, recon_loss =  train(
                    real = batch,
                    b_size = b_size,
                    G = G,
                    weights = w,
                    optimizer = optimizerG,
                    alpha = args.alpha,
                    normal_cell = True

                )
                # measurements for saving
                measurements.add_measure("train_loss",      err,      instances)
                measurements.add_measure("recon_loss",      recon_loss,      instances)

                instances += b_size


            if instances >= instances_total:
                if args.verbose:
                    pbar.write("Starting Training Tree Method for data size {}...".format(len(conditional_train_data)))
                # Train Tree Method
                bst = train_tree(conditional_train_data, normal_cell=True)

                pbar.write("Starting evaluation of conditional generator for data size {}...".format(len(conditional_train_data)))

                visited = {}
                for d in conditional_train_data:
                    h = str(d.y_normal.detach().tolist()+d.y_reduce.detach().tolist())
                    visited[h] = 1

                test_data,_,_ = sample_data(G, reduce_cell, visited, measurements, args.device, args.dataset, num=args.num_test, normal=False)

                torch.save(test_data,
                    path_measures_normal.format("sampled_all_test_"+str(len(conditional_train_data)))
                    )

                test_data =  eval_tree(bst, test_data,normal_cell=True)


                # Sort given the surrogate model
                sort = sorted(test_data, key=lambda i:i.acc, reverse=True)
                n = len(conditional_train_data)
                k = 0
                for arch in sort:
                    try:
                        arch = Dataset.get_info_generated_graph(arch, args.image_data)
                        arch.to('cpu')
                        if hasattr(arch, "val_acc"):
                            del arch.val_acc
                        conditional_train_data.append(arch)
                        k +=1
                    except:
                        continue
                    if k == args.k:
                        break

                instances = 0
                tick_size += len(conditional_train_data) - n
                instances_total = args.ticks * tick_size

                save_data(Dataset, conditional_train_data, path_measures_normal, verbose=False)


                if args.verbose:
                    top_5_acc = sorted([np.round(d.acc.item(),4) for d in conditional_train_data])[-5:]
                    print('grid search, top 5 val acc {}'.format(top_5_acc))


            if len(conditional_train_data) > args.search_data/2:
                output_name = 'round'
                trial = i
                print("Finished Grid Search with conditional prediction...", flush=True)

                print('\n* Trial summary: results')

                results = []
                for query in range(args.num_init, len(conditional_train_data), args.k):
                    best_arch = sorted(conditional_train_data[:query], key=lambda i:i.acc)[-1]
                    test_acc = best_arch.acc.item()
                    results.append((query, test_acc))

                path = os.path.join(runfolder, '{}_{}_normal_cell.pkl'.format(output_name, trial))
                print('\n* Trial summary: results')
                print(results)
                print('\n* Saving to file {}'.format(path))
                with open(path, 'wb') as f:
                    pickle.dump([results, conditional_train_data], f )
                    f.close()

                break

    normal_cell = best_arch

    path_measures_reduce= os.path.join(
                trial_runfolder, "{}_reduction.data"
                )
    instances = 0
    tick_size = len(conditional_train_data)
    instances_total = args.ticks * tick_size
    with tqdm.tqdm(total=search_data, desc="Instances", unit="") as pbar:
        while True:
            weighted_dataloader = w_dataloader(conditional_train_data, args.weight_factor, args.batch_size)
            upd = len(conditional_train_data)-pbar.n
            if upd > 0:
                pbar.update(upd)
            for batch, w in weighted_dataloader:

                G.train()

                batch = batch.to(args.device)

                b_size = batch.batch.max().item() + 1

                err, recon_loss =  train(
                    real = batch,
                    b_size = b_size,
                    G = G,
                    weights = w,
                    optimizer = optimizerG,
                    alpha = args.alpha,
                    normal_cell = False

                )
                # measurements for saving
                measurements.add_measure("train_loss",      err,      instances)
                measurements.add_measure("recon_loss",      recon_loss,      instances)

                instances += b_size


            if instances >= instances_total:
                if args.verbose:
                    pbar.write("Starting Training Tree Method for data size {}...".format(len(conditional_train_data)))
                # Train Tree Method
                bst = train_tree(conditional_train_data, normal_cell=False)
                
                pbar.write("Starting evaluation of conditional generator for data size {}...".format(len(conditional_train_data)))

                visited = {}
                for d in conditional_train_data:
                    h = str(d.y_normal.detach().tolist()+d.y_reduce.detach().tolist())
                    visited[h] = 1

                test_data,_,_ = sample_data(G, [normal_cell], visited, measurements, args.device, args.dataset, num=args.num_test, normal=True)

                torch.save(test_data,
                    path_measures_reduce.format("sampled_all_test_"+str(len(conditional_train_data)))
                    )

                test_data =  eval_tree(bst, test_data,normal_cell=False)

                # Sort given the surrogate model
                sort = sorted(test_data, key=lambda i:i.acc, reverse=True)

                n = len(conditional_train_data)
                k = 0
                for arch in sort:
                    try:
                        arch = Dataset.get_info_generated_graph(arch, args.image_data)
                        arch.to('cpu')
                        if hasattr(arch, "val_acc"):
                            del arch.val_acc
                        conditional_train_data.append(arch)
                        k +=1
                    except:
                        continue
                    if k == args.k:
                        break

                instances = 0
                tick_size += len(conditional_train_data) - n
                instances_total = args.ticks * tick_size

                save_data(Dataset, conditional_train_data, path_measures_reduce, verbose=False)


                if args.verbose:
                    top_5_acc = sorted([np.round(d.acc.item(),4) for d in conditional_train_data])[-5:]
                    print('grid search, top 5 val acc {}'.format(top_5_acc))


            if len(conditional_train_data) > args.search_data:
                output_name = 'round'
                trial = i
                print("Finished Grid Search with conditional prediction...", flush=True)

                print('\n* Trial summary: results')

                results = []
                for query in range(args.num_init, len(conditional_train_data), args.k):
                    best_arch = sorted(conditional_train_data[:query], key=lambda i:i.acc)[-1]
                    test_acc = best_arch.acc.item()
                    results.append((query, test_acc))

                path = os.path.join(runfolder, '{}_{}_reduction_cell.pkl'.format(output_name, trial))
                print('\n* Trial summary: results')
                print(results)
                print('\n* Saving to file {}'.format(path))
                with open(path, 'wb') as f:
                    pickle.dump([results, conditional_train_data], f )
                    f.close()

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
        m["nets"]["G"]["pars"]["data_config"]["regression_input"] = 176
        m["nets"]["G"]["pars"]["data_config"]["regression_hidden"] = 176
        m["nets"]["G"]["pars"]["data_config"]["regression_output"] = 1
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

        conditional_train_data = dataset.train_data
        for arch in conditional_train_data:
            arch = Dataset.get_info_generated_graph(arch)

        ##############################################################################
        #
        #                              Checkpoint
        #
        ##############################################################################
        # Load Measurements
        measurements = Measurements(
                        G = G,
                        batch_size = args.batch_size,
                        NASBench= NASBench
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