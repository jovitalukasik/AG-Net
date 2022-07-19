import sys, pathlib
import os, glob, json, tqdm
import torch
import numpy as np 
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj, subgraph
import itertools
import copy
import Settings



if __name__ == "__main__":
    
    # Set package for relative imports
    if __package__ is None or len(__package__) == 0:                  
        DIR = pathlib.Path(__file__).resolve().parent.parent
        print(DIR)
        sys.path.insert(0, str(DIR.parent))
        __package__ = DIR.name


if __name__ == "__main__":
    sys.path.insert(1, os.path.join(Settings.FOLDER, '/datasets'))
    from .Generator import scores_to_adj, adj_to_scores
else:
    sys.path.insert(1, os.path.join(Settings.FOLDER))
    from Generator import scores_to_adj, adj_to_scores

OP_PRIMITIVES = [
    'identity',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

NODE_OP_PRIMITIVES = [
    'output', # very last output of reduction cell 0
    'input', #very frist input of normal cell 1
    'input_1', #second input of normal cell c_k-1 2
    'identity', #3
    'max_pool_3x3', #4
    'avg_pool_3x3', #5
    'skip_connect', #6
    'sep_conv_3x3', #7
    'sep_conv_5x5', #8
    'dil_conv_3x3', #9 
    'dil_conv_5x5' #10
]



OP_ONEHOT = {i:np.eye(8)[i] for i in range(8)}
OP_ONEHOT_BY_PRIMITIVE ={i:OP_ONEHOT[OP_PRIMITIVES.index(i)] for i in OP_PRIMITIVES}

def sort_edge_index(edge_index):
    num_nodes = np.max(edge_index)+1
    idx = edge_index[1] * num_nodes + edge_index[0]
    perm = idx.argsort()
    return edge_index[:, perm]

adj = 1-np.tri(7)
adj[:,6] =adj[:,1]=0
EDGE_LIST_ALL = np.array(np.nonzero(adj))
EDGE_LIST_ALL=  sort_edge_index(EDGE_LIST_ALL)

L = [(x,y) for x,y in EDGE_LIST_ALL.T] 
L_inverse = {(x,y): i for i,(x,y) in enumerate(L)}


class Dataset:
            
    ##########################################################################
    def __init__(
            self,
        ):

        file_cache = os.path.join(os.getcwd(), "random_sampled_cache")
        file = 'normal_cell_topologies.json'
        d = json.load(open(file,"r"))
        data = list(d.values())
        flat_list = [item for sublist in data for item in sublist]
        self.data = []
        for arch in tqdm.tqdm(flat_list):##change to go through each folder!
            self.data.append(Dataset.map_network(arch)[0])
            # self.data[-1].scores = Dataset.map_network(arch)[1]
        
        print(f"Check data for duplicates: {file_cache}")
        ys = [graph.y.detach().numpy() for graph in self.data]
        ys_np = np.array(ys)
        u, ind = np.unique(ys_np, axis=0, return_index=True)
        self.data = [self.data[i] for i in ind]
        print(f"Length data : {len(self.data)}")
        print(f"Saving data from cache: {file_cache}")
        torch.save(self.data, file_cache)


    ##########################################################################
    @staticmethod
    def map_network_cell(item, cell="normal"):

        edge_u = []
        edge_v = []
        edge_attr = []
        for i in range(14):
            idx = f"NetworkSelectorDatasetInfo:darts:edge_{cell}_{i}"
            if idx in item:
                u, v = L[i]
                edge_u += [u]
                edge_v += [v]
                edge_attr += [NODE_OP_PRIMITIVES.index(item[idx])]

        edge_u    += [2, 3, 4, 5]
        edge_v    += [6]*4
        edge_attr += [NODE_OP_PRIMITIVES.index("identity")]*4
        
        edge_index = torch.tensor([edge_u, edge_v])
        edge_attr  = torch.tensor(edge_attr)
        
        return edge_index, edge_attr

    ##########################################################################
    @staticmethod
    def map_network_cell_like(normal_cell, reduction_cell):

        def make_adj(edge_index):
            adj = torch.zeros(11, 11)

            adj[0][2]=1
            adj[1][3]=1

            adj[2][-1] = 1
            adj[3][-1] = 1
            adj[4][-1] = 1
            adj[5][-1] = 1
            adj[6][-1] = 1
            adj[7][-1] = 1
            adj[8][-1] = 1
            adj[9][-1] = 1

            ##Edge 3 : Node  4,5
            subgraph=[]
            #subg=subgraph(list(range(4)), edge_index)[0]
            for i in range(len(edge_index[0])):
                if edge_index[1][i]==3:
                    subgraph.append((edge_index[0][i].item(), edge_index[1][i].item()))    

            if subgraph[0][0]>=2:
                if subgraph[0][0]%2!=0:
                    adj[subgraph[0][0]+1][4]=1
                    adj[subgraph[0][0]+2][4]=1
                else:
                    adj[subgraph[0][0]][4]=1
                    adj[subgraph[0][0]+1][4]=1
            else:
                adj[subgraph[0][0]][4]=1
            if subgraph[1][0]>=2:
                if subgraph[1][0]%2!=0:
                    adj[subgraph[1][0]+1][5]=1
                    adj[subgraph[1][0]+2][5]=1
                else:
                    adj[subgraph[1][0]][5]=1
                    adj[subgraph[1][0]+1][5]=1
            else:
                adj[subgraph[1][0]][5]=1

            ##Edge 4 : Node  6,7
            subgraph=[]
            for i in range(len(edge_index[0])):
                if edge_index[1][i]==4:
                    subgraph.append((edge_index[0][i].item(), edge_index[1][i].item()))    


            if subgraph[0][0]>=2:
                if subgraph[0][0]%2!=0:
                    adj[subgraph[0][0]+1][6]=1
                    adj[subgraph[0][0]+2][6]=1
                else:
                    adj[subgraph[0][0]][6]=1
                    adj[subgraph[0][0]+1][6]=1
            else:
                adj[subgraph[0][0]][6]=1
            if subgraph[1][0]>=2:
                if subgraph[1][0]%2!=0:
                    adj[subgraph[1][0]+1][7]=1
                    adj[subgraph[1][0]+2][7]=1
                else:
                    adj[subgraph[1][0]][7]=1
                    adj[subgraph[1][0]+1][7]=1
            else:
                adj[subgraph[1][0]][7]=1

            ##Edge 5 : Node  8,9
            subgraph=[]
            for i in range(len(edge_index[0])):
                if edge_index[1][i]==5:
                    subgraph.append((edge_index[0][i].item(), edge_index[1][i].item()))    

            if subgraph[0][0]>=2:
                if subgraph[0][0]%2!=0:
                    adj[subgraph[0][0]+1][8]=1
                    adj[subgraph[0][0]+2][8]=1
                else:
                    adj[subgraph[0][0]][8]=1
                    adj[subgraph[0][0]+1][8]=1
            else:
                adj[subgraph[0][0]][8]=1
            if subgraph[1][0]>=2:
                if subgraph[1][0]%2!=0:
                    adj[subgraph[1][0]+1][9]=1
                    adj[subgraph[1][0]+2][9]=1
                else:
                    adj[subgraph[1][0]+2][9]=1
                    adj[subgraph[1][0]+3][9]=1
            else:
                adj[subgraph[1][0]][9]=1

            return adj

        normal_adj = make_adj(normal_cell[0])            
        edge_index_normal = torch.nonzero(normal_adj, as_tuple=False).T

        node_attr = [NODE_OP_PRIMITIVES.index('input'), NODE_OP_PRIMITIVES.index('input_1')]
        node_attr.extend(normal_cell[1][:8])
        node_attr.extend([NODE_OP_PRIMITIVES.index('output')])
        num_nodes_in_cell = len(node_attr)
        
        reduce_adj = make_adj(reduction_cell[0])     
        edge_index_reduce = torch.nonzero(reduce_adj, as_tuple=False).T

        edge_index =  torch.LongTensor([[], []])

        edge_index = torch.cat((edge_index, edge_index_normal, torch.LongTensor([[10,10], [11,12]]), edge_index_reduce+num_nodes_in_cell),1)


        node_attr = [NODE_OP_PRIMITIVES.index('input'), NODE_OP_PRIMITIVES.index('input_1')]
        node_attr.extend(normal_cell[1][:8])
        node_attr.extend([NODE_OP_PRIMITIVES.index('output')])

        return edge_index, torch.tensor(node_attr)

    @staticmethod
    def map_network(item):
        edge_index_normal, edge_attr_normal = Dataset.map_network_cell(item, cell="normal")
        edge_index_reduce, edge_attr_reduce = Dataset.map_network_cell(item, cell="reduce")
        edge_index, node_attr = Dataset.map_network_cell_like((edge_index_normal, edge_attr_normal), (edge_index_reduce,edge_attr_reduce ))


        adj = to_dense_adj(edge_index)[0]
        scores = adj_to_scores(adj, l=231)
        x_binary = torch.nn.functional.one_hot(node_attr, num_classes=14)
        y = torch.cat((x_binary.reshape(-1).float(), scores.float()))
        
        # return edge_index, edge_attr
        return Data(edge_index=edge_index.long(), x=node_attr, num_nodes=node_attr.shape[0], x_binary=x_binary, y=y, true_normal_adj=edge_index_normal, true_reduce_adj=edge_index_reduce), scores 
    ##########################################################################
    @staticmethod
    def transform_node_atts_to_darts_cell(matrix):
        r'input already normal matrix or reduced!!'

        darts_adjacency_matrix = np.zeros((7, 7)) 

        darts_adjacency_matrix[:,0] = matrix[:7,0]
        darts_adjacency_matrix[:,1] = matrix[:7,1]
        d_c = 2
        for i in range(2,10,2):
            a = np.zeros((7))
            a[:2] = np.sum(matrix[:2,i:i+2],1)
            column_conc = np.sum(matrix[2:-1, i:i+2],1)
            row = [np.sum(column_conc[i:i+2]) for i in range(0,8,2)]
            a[2:-1] = row
            a[-1] = np.sum(matrix[-1,i:i+2])
            darts_adjacency_matrix[:,d_c] = a
            d_c += 1
        a = np.zeros((7))
        a[:2] = matrix[:2,-1]
        row = [np.sum(matrix[i:i+2,-1]) for i in range(2,10,2)]
        a[2:-1] = row
        a[-1] = matrix[-1,-1]
        darts_adjacency_matrix[:,-1] = a

        return darts_adjacency_matrix
    ##########################################################################
    @staticmethod
    def map_hyperparameters(item):
        item         = item["optimized_hyperparamater_config"]
        
        lr           = item["OptimizerSelector:sgd:learning_rate"]
        weight_decay = item["OptimizerSelector:sgd:weight_decay"]
        
        return torch.tensor([lr, weight_decay])

    ##########################################################################
    @staticmethod
    def sample(dataset,
        sample_size = 50,
        only_prediction = False,
        seed = 999         
    ):
        random_shuffle = np.random.permutation(range(len(dataset)))

        test_data = copy.deepcopy([dataset[i] for i in random_shuffle[:-sample_size]])
        
        if only_prediction:
            train_data = copy.deepcopy([dataset[i] for i in random_shuffle[-sample_size:]])
            # sampled_acc = torch.stack([g.val_acc for g in train_data])

        else:
            train_data = copy.deepcopy(dataset)
            for i in random_shuffle[:-sample_size]:
                train_data[i].val_acc *= -1

            # sampled_acc = torch.stack([train_data[i].val_acc for i in random_shuffle[-sample_size:]])
        
        return train_data, test_data #, sampled_acc

    ##########################################################################
    @staticmethod
    def generate_genotype(matrix, ops):
        r'already normal or reduced matrix and ops'

        cell = []
        i = 0
        for ind in range(2,6):
            edge_0 = np.where(matrix[:,ind]==1)[0][0]
            edge_1 = np.where(matrix[:,ind]==1)[0][1]
            cell.append((ops[i],edge_0))
            cell.append((ops[i+1],edge_1))
            i += 1
        return cell

    ##########################################################################
    @staticmethod
    def get_info_generated_graph(item, dataset=None):

        Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

        if isinstance(item , list):
            data = []
            for graph in item:
                adjacency_matrix = to_dense_adj(graph.edge_index)[0].cpu().numpy().astype(int)
                ops =[NODE_OP_PRIMITIVES[attr] for attr in graph.x.cpu().numpy()]
                try:
                    normal_darts_adj = Dataset.transform_node_atts_to_darts_cell(adjacency_matrix[:11,:11]) 
                    normal_darts_adj[normal_darts_adj == 2] = 1

                    reduce_darts_adj = Dataset.transform_node_atts_to_darts_cell(adjacency_matrix[11:,11:]) 
                    reduce_darts_adj[reduce_darts_adj == 2] = 1

                    ops_normal = ops[2:10]
                    ops_reduce =  ops[13:21]

                    normal_cell = Dataset.generate_genotype(normal_darts_adj, ops_normal)
                    reduce_cell = Dataset.generate_genotype(reduce_darts_adj, ops_reduce)

                    genotype = Genotype(normal=normal_cell, normal_concat=[2, 3, 4, 5], reduce=reduce_cell, reduce_concat=[2, 3, 4, 5])
                    # Predict
                    prediction_genotype = performance_model.predict(config=genotype_config, representation="genotype", with_noise=False)
                    runtime_genotype = runtime_model.predict(config=genotype_config, representation="genotype")

                    graph.acc = torch.FloatTensor([prediction_genotype/100.0])
                    graph.training_time = torch.FloatTensor([runtime_genotype])
                    data.append(graph)  
                except:
                    continue

        else:
            adjacency_matrix = to_dense_adj(item.edge_index)[0].cpu().numpy().astype(int)  
            try:

                normal_darts_adj = Dataset.transform_node_atts_to_darts_cell(adjacency_matrix[:11,:11]) ##dataset
                normal_darts_adj[normal_darts_adj == 2] = 1

                reduce_darts_adj = Dataset.transform_node_atts_to_darts_cell(adjacency_matrix[11:,11:]) ##dataset
                reduce_darts_adj[reduce_darts_adj == 2] = 1

                ops = [NODE_OP_PRIMITIVES[attr] for attr in item.x.cpu().numpy()]
                ops_normal = ops[2:10]
                ops_reduce =  ops[13:21]

                normal_cell = Dataset.generate_genotype(normal_darts_adj, ops_normal)
                reduce_cell = Dataset.generate_genotype(reduce_darts_adj, ops_reduce)

                genotype = Genotype(normal=normal_cell, normal_concat=[2, 3, 4, 5], reduce=reduce_cell, reduce_concat=[2, 3, 4, 5])
                
                # Predict
                prediction_genotype = performance_model.predict(config=genotype_config, representation="genotype", with_noise=False)
                runtime_genotype = runtime_model.predict(config=genotype_config, representation="genotype")

                item.acc = torch.FloatTensor([prediction_genotype/100.0])
                item.training_time = torch.FloatTensor([runtime_genotype])
                data = item

            except:
                pass
        return data


##############################################################################
#
#                              Debugging
#
##############################################################################

if __name__ == "__main__":
    
    def print_keys(d, k=None, lvl=0):
        if k is not None:
            print(f"{'---'*(lvl)}{k}")
        if type(d) == list and len(d) == 1:
            d = d[0]
        if type(d) == dict:
            for k in d:
                print_keys(d[k], k, lvl+1)
                
    # print_keys(json.load(open(os.path.join(".","NASBench301","results_0.json"), "r")))
    ds = Dataset(5)
    print(ds.data)
    for batch in ds.dataloader:
        print(batch)
        break