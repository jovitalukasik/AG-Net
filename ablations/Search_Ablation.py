######################################################################################
# Based on
# Copyright (c) Colin White, Naszilla, 
# https://github.com/naszilla/naszilla
# modified
######################################################################################

import torch
import numpy as np
import sys, os, random, argparse, copy, pickle, tqdm, shutil
from datetime import datetime

from acquisition import acquisition_fct
from pybnn import DNGO
from pybnn.bohamiann import Bohamiann
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization

from params import *

from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import DataLoader

sys.path.insert(1, os.path.join(os.getcwd()))
from Generator import Generator_Decoder, MLP_predictor
from Measurements import Measurements
from Optimizer import Optimizer, Loss
import Settings
import datasets.NASBench101 as NASBench101 
import datasets.NASBench201 as NASBench201


def sample_data(
    G,
    visited,
    measurements, 
    dataset,
    device,
    search_space, 
    num = 100,
    patience_factor = 5,
    latent_dim = 32,
    local = False
    ):
    candidates = []
    for _ in range(num*patience_factor):
        if len(candidates) >= num:
            break
        try:
            noise = (-6) * torch.rand(j, latent_dim) + 3
            graph = G(noise.to(device))
            valid_sampled_data = measurements._compute_validity_score(graph, search_space,  return_valid_spec=True)
            if local:
                for sampled_data in valid_sampled_data:
                    if str(sampled_data.z.detach().tolist())  not in visited:
                        candidates.append(sampled_data)
                        visited[str(sampled_data.z.detach().tolist())] = 1 
            else:
                sampled_y = torch.stack([g.y for g in valid_sampled_data])
                sampled_hash_idx = dataset.query(sampled_y)  
                for i, idx in enumerate(sampled_hash_idx):
                    if idx not in visited:
                        visited.append(idx)
                        candidates.append(valid_sampled_data[i])

        except:
            continue
    
    return candidates, visited
        

def dngo_search(G,  
                measurements,
                dataset,
                device, 
                Dataset,
                acq_type= 'ei',
                num_init= 16, 
                k= 16,
                total_queries= 150, 
                test_data = 1_000, 
                verbose = True,
                latent_dim = 32
    ):
    data = []
    while len(data) < num_init:
        init_data = G.generate(instances=1, device=device)
        true_data = Dataset.get_info_generated_graph(init_data,  args.image_data)
        if true_data != []:
            data.append(true_data[0])

    if args.dataset == 'NB201':
        test_data = 1_00

    visited = dataset.query(torch.stack([g.y for g in data]))  

    query = num_init + k 
    while query <= total_queries:
        
        x = np.array([graph.z.numpy() for graph in data])
        y = np.array([graph.val_acc.item() for graph in data])
        best = y.max()
        
        # get set of test data for BO
        test_candidates, visited = sample_data(G, visited, measurements, dataset, device,  args.dataset, num=test_data, latent_dim = latent_dim)

        x_test = np.array([graph.z.numpy() for graph in test_candidates])


        # 1) Train regression model on seletected datapoints
        model = DNGO(num_epochs=100, n_units_1=128,n_units_2 = 128, n_units_3=128, do_mcmc=False, normalize_input=False, normalize_output=False, rng=args.seed)

        model.train(x,y, do_optimize=True)
        print(model.network)

        # Predict on sampled test_data
        mu, v = model.predict(x_test)
        
        # 2) Optimize GP acquisition function to query next datapoints 
        acq_candidates = acquisition_fct(mu, np.sqrt(v), best, acq_type)  
        
        # 3) add the k architectures with highest acquisition function value 
        for i in acq_candidates[-k:]:

            # get true acc for graph+hp
            arch = test_candidates[i]
            arch = Dataset.get_info_generated_graph(arch, args.image_data)
        
            # 4) Append new data to GP train data
            data.append(arch)
        
        if verbose == True:
            top_5_acc = sorted([np.round(d.val_acc.item(),4) for d in data])[-5:]
            print('dngo search, query {},  top 5 val_acc {}'.format(query, top_5_acc))

        query += k

    return data 


def random_search(G, measurements,dataset, device, Dataset, total_queries = 150, latent_dim=32, patience_factor = 5, verbose=True):
    data = []
    visited = []

    tries_left = total_queries * patience_factor
    while len(data) < total_queries:
        tries_left -=1
        if tries_left <=0:
            break
        
        candidates, visited = sample_data(G, visited, measurements, dataset, device, args.dataset, num=1, latent_dim=latent_dim)
        if candidates != []:
            arch = Dataset.get_info_generated_graph(candidates, args.image_data)[0] 
            data.append(arch)

    if verbose == True:
        top_5_acc = sorted([np.round(d.val_acc.item(),4) for d in data])[-5:]
        print('random search, query {},  top 5 acc {}'.format(total_queries, top_5_acc))

    return data


def local_search(G,
                 measurements, 
                 dataset,
                 device,
                 Dataset,
                 num_init=16,
                 k=16,
                 query_full_nbhd=True,
                 stop_at_maximum=True, 
                 total_queries=500,
                 verbose = 1,
                 epsilon = 0.5, 
                 latent_dim = 32):
    ## local Search 
    query_dict = {}
    iter_dict = {}
    data = []
    query = 0

    while True:
        # loop over full runs of local search until we hit total_queries
        arch_dicts = []
        while len(arch_dicts) < num_init:  
            graph, query_dict = sample_data(G,query_dict, measurements, dataset,device,  args.dataset, num=1, latent_dim=latent_dim, local=True)
            if graph != []:
                arch_dict = Dataset.get_info_generated_graph(graph, args.image_data)[0] #####???? [0]
                data.append(arch_dict)
                arch_dicts.append(arch_dict)
                query += 1
            if query >= total_queries:
                return data

        sorted_arches = sorted([(arch, arch.val_acc.item()) for arch in arch_dicts], key = lambda i:i[1])
        arch_dict = sorted_arches[-1][0]

        while True:
            # loop over iterations of local search until we hit a local maximum
            if verbose:
                print('starting iteration, query', query)
            iter_dict[arch_dict.y] = 1

            # Create Uniform epsilon ball around best query arch_dict
            s = (-epsilon - epsilon) * torch.rand(total_queries, arch_dict.z.shape[0]) + epsilon
            nbhd_z = arch_dict.z + s

            nbhd = G(nbhd_z)

            improvement = False
            nbhd_dicts = []
            for nbr in nbhd:
                if nbr.z not in query_dict:
                    if measurements._compute_validity_score([nbr], args.dataset, return_list=True) == [False]:
                        continue
                    else:
                        query_dict[nbr.z] = 1
                        nbr_dict = Dataset.get_info_generated_graph(nbr, args.image_data)
                        data.append(nbr_dict)
                        nbhd_dicts.append(nbr_dict)
                        query +=1
                        if query >=total_queries:  
                            return data
                        if nbr_dict.val_acc > arch_dict.val_acc:
                            improvement = True
                            if not query_full_nbhd:
                                arch_dict = nbr_dict
                                break

            if not stop_at_maximum:
                sorted_data = sorted([(arch, arch.val_acc.item()) for arch in data], key = lambda i:i[1])
                index = len(sorted_data)-1
                while sorted_data[index][0].y in iter_dict:
                    index -= 1

                arch_dict = sorted_data[index][0]

            elif not improvement:
                break
            else:
                sorted_nbhd = sorted([(nbr, nbr.val_acc.item()) for nbr in nbhd_dicts], key = lambda i:i[1])
                arch_dict = sorted_nbhd[-1][0]

        if verbose:
            top_5_acc = sorted([np.round(d.val_acc.item(),4) for d in data])[-5:]
            print('random search, query {},  top 5 acc {}'.format(query, top_5_acc))


##############################################################################
#
#                             Run Experiments
#
##############################################################################
  
def run_nas(G,measurements, dataset, device, Dataset, algo_params):

    ps = copy.deepcopy(algo_params)
    algo_name = ps.pop('algo_name')
    
    if algo_name == 'random':
        data = random_search(G, measurements,dataset, device, Dataset, **ps)
    elif algo_name == 'local_search':
        data = local_search(G, measurements,dataset, device, Dataset, **ps)
    elif algo_name == 'dngo':
        data = dngo_search(G, measurements,dataset, device, Dataset, **ps)
    else:
        print('invalid search algoritm')
        sys.exit()

    # k = 10
    k = 16
    if 'k' in ps:
        k = ps['k']
    total_queries = 300
    if 'total_queries' in ps:
        total_queries = ps['total_queries']

    return compute_best_test_acc(data, total_queries, k), data

def compute_best_test_acc(data, total_queries, k):
    """
    Given completed nas algorithm, we output the test accuracy of the arch with the best found validation acc after every multiple of k:
    """
    results = []
    for query in range(k, total_queries+k, k):
        best_arch = sorted(data[:query], key=lambda i:i.val_acc)[-1]
        test_acc = best_arch.acc.item()
        val_acc = best_arch.val_acc.item()
        results.append((query, val_acc, test_acc))
    
    return results

def run_search_experiment(args, runfolder, trials):

    output_name = 'round'

    # load Checkpoint for trained Generator
    print("Load pretrained Generatpr")

    m = torch.load(os.path.join(args.saved_path, f"{args.saved_iteration}.model"), map_location=args.device) #pretrained_dict
    m["nets"]["G"]["pars"]["list_all_lost"] = True
    m["nets"]["G"]["pars"]["acc_prediction"] = False
    G = Generator_Decoder(**m["nets"]["G"]["pars"]).to(args.device)
    state_dict = m["nets"]["G"]["state"]
    G.load_state_dict(state_dict)

    print("G parameters:")
    print(G.pars)
    print(G)
    print()

    
    # Set random seed for reproducibility
    print("Search deterministically.")
    seed = args.seed
    print(f"Random Seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    latent_dim = m["nets"]["G"]["pars"]["model_config"]["graph_embedding_dim"]

    algo = args.search_method
    algorithm_params = algo_params(algo)
    num_algos = len(algorithm_params)

    if args.dataset=='NB101':
        NASBench = NASBench101
        Dataset = NASBench101.Dataset
        dataset =  NASBench101.Dataset(batch_size=16, sample_size=16, only_prediction=True)
    elif args.dataset=='NB201':
        NASBench = NASBench201
        Dataset = NASBench.Dataset
        dataset =  NASBench.Dataset(batch_size=16, sample_size=16, only_prediction=True, dataset=args.image_data)
    else:
        raise TypeError("Unknow Seach Space : {:}".format(args.dataset))

    dataset.load_hashes()
        # Load Measurements
    measurements = Measurements(
                    G = G, 
                    batch_size = 1,
                    NASBench = NASBench
                )

    # for i in range(args.trials):
    results = []
    run_data = []
    for j in range(num_algos):
        print('\n Running algorithm and trial: {}_{}'.format(algo, trials))
        algo_result, data = run_nas(G, measurements, dataset, args.device, Dataset, algorithm_params[j])
        algo_result = np.round(algo_result, 5)

        results.append(algo_result)
        run_data.append(data)

    runfolder = os.path.join(runfolder, '{}_{}.pkl'.format(output_name, trials))
    print('\n* Trial summary: results')
    print(results)
    print(algorithm_params)
    print('\n* Saving to file {}'.format(runfolder))
    with open(runfolder, 'wb') as f:
        pickle.dump([algorithm_params, results, run_data], f )
        f.close()

def main(args):
    now = datetime.now()
    runfolder = now.strftime("%Y_%m_%d_%H_%M_%S")
    if args.dataset == 'NB201':
        runfolder = f"{args.name}_{args.dataset}/{args.image_data}/{args.search_method}/{runfolder}"
    else:
        runfolder = f"{args.name}_{args.dataset}/{args.search_method}/{runfolder}"
    runfolder = os.path.join(Settings.FOLDER_EXPERIMENTS, runfolder)
    if not os.path.exists(runfolder):
        os.makedirs(runfolder)

    # save command line input
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    with open(os.path.join(runfolder, 'cmd_input.txt'), 'a') as f:
        f.write(cmd_input)
    print('Command line input: ' + cmd_input + ' is saved.')
    
    for i in range(args.trials):
        if args.trials > 1:
            args.seed = i
        run_search_experiment(args, runfolder, i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for NAS latent space search experiments')
    parser.add_argument("--device",             type=str, default="cpu")
    parser.add_argument('--trials',             type=int, default=1, help='Number of trials')
    parser.add_argument('--dataset',            type=str, default='NB201')
    parser.add_argument('--image_data',         type=str, default='cifar10_valid_converged', help='Only for NB201 relevant, choices between [cifar10_valid_converged, cifar100, ImageNet16-120]')
    parser.add_argument("--name",               type=str, default="Search_Ablation")
    parser.add_argument("--search_method",      type=str, default="local_search")
    parser.add_argument("--saved_path",         type=str, help="Load pretrained Generator", default="state_dicts/NASBench201")
    parser.add_argument("--saved_iteration",    type=str,  default='best', help="Which iteration to load of pretrained Generator")
    parser.add_argument("--seed",               type=int, default=1)

    args = parser.parse_args()
    main(args)

