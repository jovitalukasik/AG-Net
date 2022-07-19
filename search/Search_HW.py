import torch
import numpy as np
import sys, os, random, argparse, copy, pickle, tqdm, shutil
from datetime import datetime
from sklearn.metrics import mean_squared_error
from scipy import stats

import matplotlib.pyplot as plt

from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import DataLoader

sys.path.insert(1, os.path.join(os.getcwd()))
from Generator import Generator_Decoder, MLP_predictor
import datasets.NASBenchHW as NASBenchHW
NASBenchHW.nasbench #= NASBench201.nasbench
from Measurements import Measurements
from Optimizer import Optimizer, Loss
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

devices_NB201 = [
    'edgegpu',
    'raspi4',
    'edgetpu',
    'pixel3',
    'eyeriss',
    'fpga',
]

parser = argparse.ArgumentParser(description='Args for NAS latent space search experiments')
parser.add_argument("--device",             type=str, default="cpu")
parser.add_argument('--trials',             type=int, default=1, help='Number of trials')
parser.add_argument('--dataset',            type=str, default='NB201')
parser.add_argument('--image_data',         type=str, default='cifar10_valid_converged', help='Only for NB201 relevant, choices between [cifar10_valid_converged, cifar100, ImageNet16-120]')
parser.add_argument('--target_device',      type=str, default='edgegpu')
parser.add_argument('--max_latency',        type=float, default=5)
parser.add_argument("--name",               type=str, default="Search_HW")
parser.add_argument("--weight_factor",      type=float, default=10e-3)
parser.add_argument("--num_init",           type=int, default=16)
parser.add_argument("--k",                  type=int, default=16)
parser.add_argument("--num_test",           type=int, default=100)
parser.add_argument("--ticks",              type=int, default=30)
parser.add_argument("--tick_size",          type=int, default=16)  
parser.add_argument("--batch_size",         type=int, default=16)
parser.add_argument("--search_data",        type=int, default=310)
parser.add_argument("--saved_path",         type=str, help="Load pretrained Generator", default="state_dicts/NASBench201")
parser.add_argument("--saved_iteration",    type=str,  default='best' , help="Which iteration to load of pretrained Generator")
parser.add_argument("--seed",               type=int, default=1)
parser.add_argument("--alpha",              type=float, default=0.95)
parser.add_argument("--verbose",            type=int, default=0)
parser.add_argument("--joint",              type=int, default=1)

args = parser.parse_args()
##############################################################################
#
#                              Runfolder
#
##############################################################################
now = datetime.now()
runfolder = now.strftime("%Y_%m_%d_%H_%M_%S")
runfolder = f"NAS_Search_{args.dataset}/joint={args.joint}/{args.image_data}/{args.target_device}/{args.max_latency}/{runfolder}_{args.seed}_{args.name}"
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
    patience_factor = 1,
    latent_dim = 32, ):
    i = 0
    validity = 0
    possible_candidates = []
    with torch.no_grad():
        G.eval()
        # generate up to 10_000 possible graphs
        for j in range(5_00,3_000,5_00):
            noise = (-6) * torch.rand(j, latent_dim) + 3
            graphs = G(noise.to(device), mlp_condition=target_device_distribution.to(device))
            sampled_data_idx = measurements._compute_validity_score(graphs, search_space,  return_list=True)
            valid_sampled_data = [graphs[i] for i,v in enumerate(sampled_data_idx) if v == True]
            validity += sum(sampled_data_idx)
            for sampled_data in valid_sampled_data:
                if str(sampled_data.y.detach().tolist())  not in visited:
                    possible_candidates.append(sampled_data)
                    visited[str(sampled_data.y.detach().tolist())] = 1 
            i += j
        
            if len(possible_candidates) > num:
                random_shuffle = np.random.permutation(range(len(possible_candidates)))
                possible_candidates = [possible_candidates[i] for i in random_shuffle[:num]]
                break

    print('In total generated graphs: {}'.format(i))
    validity = validity/i
    return possible_candidates, visited, validity

def get_rank_weights(outputs, weight_factor, highest_best=True):
    if highest_best:
        outputs_argsort = np.argsort(-np.asarray(outputs))
    else:
        outputs_argsort = np.argsort(np.asarray(outputs))
    ranks = np.argsort(outputs_argsort)
    return 1 / (weight_factor * len(outputs) + ranks)

def w_dataloader_old(train_data, args):
    b_size = args.batch_size
    if args.weighted_retraining:
        outputs_acc = np.array([graph.val_acc.item() for graph in train_data])
        weights_acc = torch.tensor(get_rank_weights(outputs_acc, args.weight_factor, highest_best=True))
        outputs_device = np.array([graph.devices_latency[devices_NB201.index(args.target_device)][-1].item() for graph in train_data])
        if args.joint:
            weights_device = torch.tensor(get_rank_weights(outputs_device, args.weight_factor, highest_best=False))
            penalty = torch.tensor([0.005 if latency > args.max_latency else 1 for latency in outputs_device])
            weights = (weights_acc+weights_device)* penalty

        else:
            penalty = torch.tensor([0.005 if latency > args.max_latency else 1 for latency in outputs_device])
            weights = weights_acc * penalty

    else:
        weights = torch.ones(b_size)
    
    sampler = WeightedRandomSampler(
            weights, len(train_data))
    weighted_train_data = [(train_data[i],weights[i]) for i,w in enumerate(weights)]
    weighted_dataloader = DataLoader(weighted_train_data, sampler = sampler, batch_size = b_size, num_workers = 0, pin_memory = True)
    
    return weighted_dataloader

def w_dataloader(train_data, args):
    b_size = args.batch_size
    acc = np.array([graph.val_acc.item() for graph in train_data])
    acc_rank = np.argsort(np.argsort(-acc))+1
    lat = np.array([graph.devices_latency[devices_NB201.index(args.target_device)][-1].item() for graph in train_data])
    penalty = np.array([1000 if latency > args.max_latency else 1 for latency in lat])
    if args.joint:
        lat_rank = np.argsort(np.argsort(lat))+1
    else:
        lat_rank = np.zeros_like(acc_rank)
    rank = (acc_rank + lat_rank) * penalty
    weights = torch.tensor(get_rank_weights(rank, args.weight_factor, highest_best=False))

    
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
    args
):  
    optimizer.zero_grad()

    device_data = torch.stack([real.devices_latency.reshape(b_size, len(devices_NB201), len(devices_NB201)+1)[i][devices_NB201.index(args.target_device)] for i in range(b_size)])
    generated, rec_loss, mse = G.loss(real, b_size, device_data)

    mse_a = mse[0]
    mse_a_divisor = torch.tensor([max(a.detach().item(),1) for a in mse_a]).to(mse_a.device)
    mse_a = mse_a / mse_a_divisor

    mse_l = mse[1]
    mse_l_divisor = torch.tensor([max(l.detach().item(),1) for l in mse_l]).to(mse_l.device)
    mse_l = mse_l / mse_l_divisor

    recon_loss_divisor = torch.tensor([max(r.detach().item(),1) for r in rec_loss]).to(rec_loss.device)
    recon_loss = rec_loss/recon_loss_divisor

     
    err = (1-args.alpha)*(recon_loss) + args.alpha*(0.5*mse_l + 0.5*mse_a)
    err = torch.mean(err*weights.to(real.x.device))
        

    err.backward()

    # optimize
    optimizer.step()

    # return stats    
    return (err.item(), 
            rec_loss.mean().item(), 
            mse_a.mean().item(),
            mse_l.mean().item(),
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

def get_summed_ranking(data,evaluated=False):
    # outputs sum of ranking for ntk and lr for each arch and the arch
    # the first is the best
    if evaluated:
        sort_device = sorted([(idx, arch) for idx, arch in enumerate(data)], key = lambda i:i[1].devices_latency[devices_NB201.index(args.target_device)][-1])
    else:
        sort_device = sorted([(idx, arch) for idx, arch in enumerate(data)], key = lambda i:i[1].latency)
    sorted_devices=[] # ranking, idx, arch
    for i,tup in enumerate(sort_device): 
        sorted_devices.append((i,*tup))

    sort_acc = sorted([(idx, arch) for idx, arch in enumerate(data)], key = lambda i:i[1].val_acc, reverse=True)
    sorted_acc=[] # ranking, idx, arch
    for i,tup in enumerate(sort_acc): 
        sorted_acc.append((i,*tup))

    ranking_sum = []
    ranking_device = [item[0] for i in range(len(data))  for item in sorted_devices if item[1] == i ]
    ranking_acc = [item[0] for i in range(len(data))  for item in sorted_acc if item[1] == i ]
    ranking_sum = sorted([(ranking_device[i]+ranking_acc[i], arch) for i,arch in enumerate(data)], key = lambda i:i[0])
    return ranking_sum


def rank_generated(data, rank_latency=True, penalty=True):
    accuracies = np.argsort(np.argsort([-d.val_acc for d in data]))
    if rank_latency:
        latencies = np.argsort(np.argsort([d.latency for d in data]))
    else:
        latencies = np.zeros_like(accuracies)
    if penalty:
        penalty = np.array([1000 if d.latency > args.max_latency else 1 for d in data])
    else:
        penalty = np.ones_like(accuracies)
    ranking = (latencies + accuracies) * penalty
    idx = np.argsort(ranking)
    return [(ranking[i], data[i]) for i in idx]

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


    with tqdm.tqdm(total=search_data, desc="Data", unit="") as pbar:
        while True:
            if args.verbose:
                pbar.write("Starting Training Loop...")
            weighted_dataloader = w_dataloader(conditional_train_data, args)
            upd = len(conditional_train_data)-pbar.n
            if upd > 0:
                pbar.update(upd)

            for batch, w in weighted_dataloader:

                G.train()

                batch = batch.to(args.device)

                b_size = batch.batch.max().item() + 1
                ### Training step for G ###
                err, recon_loss, pred_loss, device_loss =  train(
                    real = batch,
                    b_size = b_size,
                    G = G,
                    weights = w,
                    optimizer = optimizerG,
                    args = args
                )

                if args.verbose:
                    pbar.write("recon_loss : {}".format(recon_loss))
                    pbar.write("acc_loss : {}".format(pred_loss))
                    pbar.write("device_loss : {}".format(device_loss))
                # measurements for saving
                measurements.add_measure("train_loss",      err,      instances)
                measurements.add_measure("recon_loss",      recon_loss,      instances)
                measurements.add_measure("acc_loss",      pred_loss,      instances)
                measurements.add_measure("device_loss",      device_loss,      instances)
                if args.verbose:
                    pbar.write("recon_loss : {}".format(recon_loss))
                    pbar.write("acc_loss : {}".format(pred_loss))
                    pbar.write("device_loss : {}".format(device_loss))

                instances += b_size
                

            if instances >= instances_total:
                if args.verbose:
                    pbar.write("Starting evaluation of conditional generator for data size {}...".format(len(conditional_train_data)))
                visited = {}
                for d in conditional_train_data:
                    h = str(d.y.detach().tolist()) 
                    visited[h] = 1


                # Finished Training, now evaluate trained surrogate model for next samples
                test_data,_ , _ = sample_data(G, visited, measurements, args.device, args.dataset, num=args.num_test)
                n = len(conditional_train_data)

                if test_data == []:
                    pbar.write('No new valid data found, stop search with {} data'.format(n))   
                    search_data = len(conditional_train_data)-1
                else:
                    pbar.write(f"Sampled {len(test_data)} architectures.")
                    torch.save(test_data, 
                        path_measures.format("sampled_all_test_"+str(len(conditional_train_data)))
                        )

                    # Sort given the surrogate model
                    ranking = rank_generated(test_data, rank_latency=args.joint, penalty=True)

                    for ranking, arch in ranking[:args.k]:
                        # get true acc for graph+hp
                        arch = Dataset.get_info_generated_graph(arch, args.image_data)#####???? [0]

                        # 4) Append new data to GP train data
                        arch.to('cpu')
                        if hasattr(arch, "z"):
                            del arch.z    
                            del arch.g
                            del arch.latency
                        conditional_train_data.append(arch)

                    checkpoint.save(len(conditional_train_data), only_model=True)            
                    instances = 0
                    tick_size += len(conditional_train_data)-n

                    instances_total = args.ticks * tick_size

                    save_data(Dataset, conditional_train_data, path_measures, verbose=False)
                    
                    
                    if args.verbose:
                        if args.joint:
                            ranking_sum = get_summed_ranking(conditional_train_data, evaluated=True)
                            top5 = ranking_sum[:5]
                            top_5_latency = [np.round(i.devices_latency[devices_NB201.index(args.target_device)][-1].item(),4) for r,i in top5]
                            top_5_acc = [np.round(i.val_acc.item(),4) for r,i in top5]
                        else:
                            sorted_arch =  sorted(conditional_train_data, key=lambda i:i.val_acc, reverse=True)
                            sorted_latency = [np.round(d.devices_latency[devices_NB201.index(args.target_device)][-1].item(),4) for d in sorted_arch]
                            latency_ind_condition = np.where(np.array(sorted_latency)<args.max_latency)[0][:5]
                            if latency_ind_condition == []:
                                top_5_acc = [np.round(sorted_arch[i].val_acc.item(),4) for i in sorted_arch]
                                top_5_latency = [sorted_latency[i] for i in sorted_arch]
                            else:
                                top_5_latency = [sorted_latency[i] for i in latency_ind_condition]
                                top_5_acc = [np.round(sorted_arch[i].val_acc.item(),4) for i in latency_ind_condition]

                        pbar.write('grid search, top 5 val acc, that fulfill latency condition for {} {},{}'.format(args.target_device, top_5_acc,top_5_latency ))

                    
            if len(conditional_train_data) > search_data:
                output_name = 'round'
                # i = 0
                trial = i
                pbar.write("Finished Grid Search with conditional latency...") 

                pbar.write('\n* Trial summary: results')

                results = []
                for query in range(args.num_init, len(conditional_train_data), args.k):
                    if args.joint:
                        ranking_sum = get_summed_ranking(conditional_train_data[:query],evaluated=True)
                        best_arch = ranking_sum[0][1]
                    else:
                        best_archs = sorted(conditional_train_data[:query], key=lambda i:i.val_acc,  reverse=True)
                        sorted_latency =  [np.round(d.devices_latency[devices_NB201.index(args.target_device)][-1].item(),4) for d in best_archs]
                        latency_ind_condition = np.where(np.array(sorted_latency)<args.max_latency)[0][0]
                        if latency_ind_condition == []:
                            best_arch = best_archs[0]
                        else:
                            best_arch = best_archs[latency_ind_condition]

                    test_acc = best_arch.acc.item()
                    val_acc = best_arch.val_acc.item()
                    latency = best_arch.devices_latency[devices_NB201.index(args.target_device)][-1].item()
                    results.append((query, val_acc, test_acc, latency))

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
        torch.cuda.empty_cache()
        
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
        regression_conditions = 6
        m["nets"]["G"]["pars"]["data_config"]["regression_input"] = 84 + regression_conditions
        m["nets"]["G"]["pars"]["regression_conditions"]=regression_conditions
        m["nets"]["G"]["pars"]["data_config"]["regression_output"] = 2
        m["nets"]["G"]["pars"]["acc_prediction"]=True
        m["nets"]["G"]["pars"]["list_all_lost"]=True
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

        # G_dict.update(state_dict) 
        G_dict.update(new_state_dict) 
        G.load_state_dict(G_dict)

        # Load Measurements
        measurements = Measurements(
                        G = G, 
                        batch_size = args.batch_size,
                        NASBench = NASBenchHW#NASBench201
                    )
        ##############################################################################
        #
        #                              Losses
        #
        ##############################################################################

        print("Initialize optimizers.")
        optimizerG = Optimizer(G, 'latency_prediction').optimizer 
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
            folder = trial_runfolder,
            nets = chkpt_nets,
            optimizers = chkpt_optimizers,
            measurements = measurements
        )

        ##############################################################################
        #
        #                              Dataset
        #
        ##############################################################################
        print("Creating Dataset.")
        Dataset = NASBenchHW.Dataset
        dataset =  NASBenchHW.Dataset(batch_size=args.batch_size, sample_size=args.num_init, only_prediction=True,  dataset=args.image_data)

        conditional_train_data = dataset.train_data
        target_device_distribution = torch.FloatTensor([NASBenchHW.devices_ONEHOT_HWNB_by_device[args.target_device]])
        
        print('Start Search for device {} with maximal latency {}'.format(args.target_device, args.max_latency))
        training_loop()