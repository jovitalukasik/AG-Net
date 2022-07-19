from jmespath import search
import torch
from sklearn.metrics import mean_squared_error

from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data ,DataLoader
from torch_geometric.data import DataLoader

import numpy as np
from scipy import linalg

################################################################
class Measurements:
    def __init__(
            self,
            G, 
            batch_size, 
            NASBench
    ):
        
        self.G = G
        self.batch_size = batch_size
        
        # measurements

        self.m_i = {}
        self.m_other = {}

        self.others = {}
        self.others_instances = {}

        self.NASBench = NASBench
    ############################################################
    def measure(
            self,
            instances,
            search_space,
            device, 
            acc_prediction = False,

        ):

        with torch.no_grad():

            # Eval Generator
            self.G.eval()
            
            fakes = self.G.generate(
                instances = 10_000,
                device = device,

            )

            self.fid_scores = self.compute_fid_from_data(
                    np.array([g.y.cpu().numpy() for g in fakes])) 
            self.validity_scores = self._compute_validity_score(fakes, dataset=search_space)

            graph_properties = self.graph_properties(fakes)
            self.graph_properties_min = graph_properties[0]
            self.graph_properties_max = graph_properties[1]
            self.graph_properties_mean = graph_properties[2]
            self.graph_properties_std = graph_properties[3]

            # Keeps the predicted/generated graph, only include true accuracies
            if search_space == 'NB101':
                dataset = self.NASBench.Dataset.get_info_generated_graph(fakes)
            elif search_space == 'NB201':
                dataset = self.NASBench.Dataset.get_info_generated_graph(fakes, dataset='cifar10_valid_converged')
            elif search_space =='NBNLP' or search_space == 'NB301':
                pass
            else:
                raise NotImplementedError()
            
            if acc_prediction == True:
                if dataset != [] :
                    dataloader = DataLoader(dataset, shuffle=False, num_workers=0, pin_memory=False, batch_size=128)
                    self.mse_test_loss = self._compute_prediction_test_mse(dataloader, device)
                else: 
                    self.mse_test_loss = torch.ones(1)
            else:
                self.mse_test_loss = torch.ones(1)
                
            self.G.train()
    
    ############################################################
    def add_measure(self, name, val, instances):
        try:
            self.others[name].append(val)
            self.others_instances[name].append(instances)
        except KeyError:
            self.others[name] = [val]
            self.others_instances[name] = [instances]
            
    ############################################################
    def get_measurements(self, instances):

        i = instances // self.batch_size - 1

        try:
            self.m_i["Measures.fid_scores"].append(self.fid_scores)
            self.m_i["Measures.validity_scores"].append(self.validity_scores)
            self.m_i["Measures.instances"].append(instances)
            self.m_i["Measures.min_edges"].append(self.graph_properties_min)
            self.m_i["Measures.max_edges"].append(self.graph_properties_max)
            self.m_i["Measures.mean_edges"].append(self.graph_properties_mean)
            self.m_i["Measures.std_edges"].append(self.graph_properties_std)
            self.m_i["Measures.mse_test_loss"].append(self.mse_test_loss)


        except KeyError: 
            self.m_i["Measures.fid_scores"] = [self.fid_scores]
            self.m_i["Measures.validity_scores"] = [self.validity_scores]
            self.m_i["Measures.instances"] = [instances]
            self.m_i["Measures.min_edges"] = [self.graph_properties_min]
            self.m_i["Measures.max_edges"] = [self.graph_properties_max]
            self.m_i["Measures.mean_edges"] = [self.graph_properties_mean]
            self.m_i["Measures.std_edges"] = [self.graph_properties_std]
            self.m_i["Measures.mse_test_loss"] = [self.mse_test_loss]


        for k, v in self.others.items():
            self.m_other[f"Measures.{k}"] = [v]

        for k, v in self.others_instances.items():
            self.m_other[f"Measures.instances_{k}"] = [v]

        
        return self.m_i, self.m_other
            
    ############################################################
    def best_models(self):

        models = set()

        if len(self.fid_scores) > 0:
            models.add(
                self.fid_scores.index(min(self.fid_scores))
            )
        if len(self.validity_scores) > 0 :
            models.add(
                self.validity_scores.index(max(self.validity_scores))
            )
        if len(self.mse_test_loss) > 0 :
            models.add(
                self.mse_test_loss.index(min(self.mse_test_loss))
            )

        return models
    ############################################################
    # Based on https://github.com/google-research/nasbench :
    @staticmethod
    def _is_valid_dag(matrix, num_vertices=7, max_edges=9, search_space='NB101', zero_colums=1,  zero_rows=0):
        NUM_VERTICES = num_vertices
        MAX_EDGES = max_edges

        shape = np.shape(matrix)
        num_edges = np.sum(matrix)
        num_vertices = shape[0]

        if len(shape) != 2 or shape[0] != shape[1]:
            return False
        if not Measurements.is_upper_triangular(matrix):
            return False
        
        if not Measurements.is_full_dag(matrix, search_space=search_space,  zero_colums=zero_colums, zero_rows=zero_rows):
            return False

        if num_vertices > NUM_VERTICES:
            return False

        if num_edges > MAX_EDGES:
            return False

        return True
    
    @staticmethod
    def is_upper_triangular(matrix):
        """True if matrix is 0 on diagonal and below."""
        for src in range(np.shape(matrix)[0]):
            for dst in range(0, src + 1):
                if matrix[src, dst] != 0:
                    return False
        return True

    @staticmethod
    def is_full_dag(matrix, search_space='NB101', zero_colums=1, zero_rows=0):
        """Full DAG == all vertices on a path from vert 0 to (V-1).

        i.e. no disconnected or "hanging" vertices.

        It is sufficient to check for:
            1) no rows of 0 except for row V-1 (only output vertex has no out-edges)
            2) no cols of 0 except for col 0 (only input vertex has no in-edges)

        Args:
            matrix: V x V upper-triangular adjacency matrix

        Returns:
            True if the there are no dangling vertices.
        """
        shape = np.shape(matrix)
        if search_space == 'NB301':
            rows = matrix[:shape[0]-1, :] == 0
        elif search_space == 'NBNLP':
            rows = matrix[zero_rows:shape[0]-1, :]
        else:
            rows = matrix[:shape[0]-1, :] == 0
        rows = np.all(rows, axis=1)     # Any row with all 0 will be True
        rows_bad = np.any(rows)
        if search_space == 'NBNLP':
            cols = matrix[:, zero_colums:]
        elif search_space == 'NB301':
            cols = matrix[:, 2:] == 0
        else:
            cols = matrix[:, 1:] == 0
        cols = np.all(cols, axis=0)     # Any col with all 0 will be True
        cols_bad = np.any(cols)

        return (not rows_bad) and (not cols_bad)

    ############################################################
    def _compute_matrix_validity(self, fake_data, dataset='NB101'):
        n_valid = 0 
        for graph in fake_data:
            adjacency_matrix = to_dense_adj(graph.edge_index)[0].numpy().astype(int)
            if Measurements._is_valid_dag(adjacency_matrix):
                n_valid += 1

        return n_valid / (len(fake_data))

    ############################################################
    def _compute_validity_score(self, fake_data, dataset='NB101', return_list=False, return_valid_spec=False):

        n_valid = 0 
        valid_list = []
        valid_specs = []
        if dataset == 'NB101':
            for graph in fake_data:
                try:
                    ops = []
                    for op in graph.x_binary:
                        op= op.argmax().item()
                        ops.append(self.NASBench.OPS_by_IDX_NB101[op])
                        if op == 0 :
                            break
                except:
                    ops = [self.NASBench.OPS_by_IDX_NB101[attr.item()] for attr in graph.x]
                try:
                    adjacency_matrix = to_dense_adj(graph.edge_index)[0].cpu().numpy().astype(int)            
                    spec = self.NASBench.api.ModelSpec(matrix = adjacency_matrix, ops= ops)
                    if self.NASBench.nasbench.is_valid(spec):
                        n_valid += 1
                        fixed_metrics, computed_metrics = self.NASBench.nasbench.get_metrics_from_spec(spec)
                        valid_graph = self.NASBench.Dataset.map_network(fixed_metrics)[0]
                        graph.edge_index = valid_graph.edge_index
                        graph.x = valid_graph.x
                        graph.x_binary = valid_graph.x_binary
                        graph.y = valid_graph.y
                        graph.y_generated = graph.y
                        valid_specs.append(graph)
                    valid_list.append(self.NASBench.nasbench.is_valid(spec))
                except:
                    valid_list.append(False)
                    continue
        elif dataset == 'NB201':
            for graph in fake_data:
                ops = [self.NASBench.OPS_by_IDX_NB201[i] for i in graph.x.detach().cpu().numpy()]
                matrix = to_dense_adj(graph.edge_index)[0].cpu().numpy().astype(int)
                if not matrix.shape[0] == 8:
                    valid_list.append(False)
                    continue
                if not (matrix == self.NASBench.ADJACENCY_NB201).all():
                    valid_list.append(False)
                    continue
                if ops[0] != 'input' or ops[-1] !='output':
                    valid_list.append(False)
                    continue
                flag = True
                for i in range(1, len(ops)-1):
                    if ops[i] == 'input':
                        valid_list.append(False)
                        flag = False
                        print(flag)
                        break
                if flag == False:
                    continue
                else:
                    valid_specs.append(graph)
                    n_valid += 1
                    valid_list.append(True)  
        elif dataset == 'NB301':
            # only one cell for validity check either normal or reduce not important
            for graph in fake_data:
                ops = [self.NASBench.NODE_OP_PRIMITIVES[i] for i in graph.x.cpu().numpy()]
                flag = True
                for i in range(2, len(ops)-1):
                    if ops[i] not in ['identity','max_pool_3x3','avg_pool_3x3','skip_connect','sep_conv_3x3','sep_conv_5x5','dil_conv_3x3','dil_conv_5x5']:
                        valid_list.append(False)
                        flag = False
                        break
                if flag == False:
                    continue
                try:
                    matrix = to_dense_adj(graph.edge_index)[0].cpu().numpy().astype(int)

                except:
                    continue
                if matrix.shape[0] != 11:
                    valid_list.append(False)
                    continue

                normal_darts = self.NASBench.Dataset.transform_node_atts_to_darts_cell(matrix)
                log_sig = torch.nn.LogSigmoid() 
                edges = log_sig(graph.g[-55:])
                score_matrix = self.NASBench.scores_to_adj(edges, num_nodes=11).cpu().numpy()
                score_matrix_darts = self.NASBench.Dataset.transform_node_atts_to_darts_cell(score_matrix)

                normal_darts[normal_darts >= 2] = 1  

                # prune edges if too many edges:
                for i,column in enumerate(normal_darts[:,2:6].T):
                    column[2+i:] = 0
                    if sum(column) > 2:
                        score_column = score_matrix_darts[:,2+i]
                        ind = np.argsort(score_column[np.where(column==1)[0]])[:-2]
                        column[np.where(column==1)[0][ind]]= 0


                if not normal_darts[:,2:6].sum()==8 or not (normal_darts[:,2:6].sum(0) == 2).all(): 
                    valid_list.append(False)
                    continue  
                if not Measurements.is_upper_triangular(normal_darts):
                    valid_list.append(False)
                    continue
                else:
                    n_valid += 1
                    valid_list.append(True)  
                    valid_specs.append(graph) 
        elif dataset == 'NBNLP':
            for graph in fake_data:
                ops = [self.NASBench.NB_NLP_ops[i] for i in graph.x.cpu().numpy()]
                if graph.num_nodes > 27:
                    valid_list.append(False)
                    continue
                if ops[0] != 'input' or ops[-1] !='output':
                    valid_list.append(False)
                    continue
                flag = True
                for i in range(4):
                    if ops[i] != 'input':
                        valid_list.append(False)
                        flag = False
                        break
                if flag == False:
                    continue
                for i in range(4, len(ops)-1):
                    if ops[i] not in ['linear', 'blend', 'elementwise_sum',  'elementwise_prod', 'activation_tanh', 'activation_sigm', 'activation_leaky_relu']:
                        valid_list.append(False)
                        flag = False
                        break
                if flag == False:
                    continue

                # prune hidden states if too many:
                matrix = to_dense_adj(graph.edge_index)[0].cpu().numpy().astype(int) 
                column = matrix[:,-1]
                if column[-2]==0:
                    valid_list.append(False)
                    continue
                if len(graph.x) <14:
                    if sum(column)>3:
                        log_sig = torch.nn.LogSigmoid() 
                        edges = log_sig(graph.g[-325:])
                        score_column = self.NASBench.scores_to_adj(edges, num_nodes=26).cpu().numpy()[:len(graph.x)-2,-1]
                        ind = np.argsort(score_column[np.where(column[:-2]==1)[0]])[:-2]
                        column[np.where(column==1)[0][ind]]= 0
                        graph.edge_index = torch.tensor(np.nonzero(matrix))

                if not Measurements._is_valid_dag(matrix, num_vertices=26, max_edges=42, search_space='NBNLP', zero_colums=4, zero_rows=4):
                    valid_list.append(False)
                    continue
                if matrix.shape[0] != graph.num_nodes:
                    valid_list.append(False)
                    continue
                else:
                    n_valid += 1
                    valid_list.append(True)  
                    valid_specs.append(graph)      
        else:
            raise NotImplementedError()

        if return_list :
            return valid_list
        elif return_valid_spec:
            return valid_specs

        return n_valid / (len(fake_data))
    
    ############################################################

    def _compute_prediction_test_mse(self, dataset, device):
        preds = []
        targets = []
        # loop trough test data batches 
        for batch in dataset:
            if isinstance(batch, list):
                batch = batch[0]
            batch = batch.to(device)
            batch = batch.clone().apply(lambda x: x.detach())
            batch_size = batch.batch.max().item() + 1
            output = self.G(batch.z.reshape(batch_size, -1).float())
            pred = torch.stack([g.val_acc for g in output])
            preds.extend((pred.detach().cpu().numpy()))
            targets.extend((batch.val_acc.detach().cpu().numpy()))

        test_loss = mean_squared_error(targets, preds)
        
        return test_loss

    ############################################################
    def graph_properties(self, fake_data):
        
        num_nodes = [i.edge_index.shape[1] for i in fake_data]
        min_edges = min(num_nodes)
        max_edges = max(num_nodes)
        mean_edges = np.mean(num_nodes)
        std_edges = np.std(num_nodes)
        return min_edges, max_edges, mean_edges, std_edges  

    ############################################################
    def fit_fid(self, data):
        mu = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)
        
        return mu, cov
    
    ############################################################
    def set_fid_real_stats(self, data):
        self.fid_mu, self.fid_cov = self.fit_fid(data)
        
    ############################################################
    def compute_fid(
            self,
            mu2,
            cov2,
            eps = 1E-6
        ):
        
        mu1, cov1 = self.fid_mu, self.fid_cov

        diff = mu1 - mu2
    
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(cov1.shape[0]) * eps
            covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))
    
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1E-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
    
        tr_covmean = np.trace(covmean)
    
        return ( diff.dot(diff) +
                 np.trace(cov1) +
                 np.trace(cov2) -
                 2 * tr_covmean )
    
    ############################################################
    def compute_fid_from_data(
            self,
            data
        ):
        
        mu2, cov2 = self.fit_fid(data)
        return self.compute_fid(mu2, cov2)
