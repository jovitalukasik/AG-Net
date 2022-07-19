import json
import os, tqdm
import numpy as np
from collections import defaultdict

from tqdm import tqdm
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.read_and_write import json as config_space_json_r_w

# from surrogate_models.utils import ConfigLoader



class ConfigLoader:
    def __init__(self, config_space_path):
        self.config_space = self.load_config_space(config_space_path)

        # The exponent to scale the fidelity with.
        # Used to move architectures across the fidelity budgets
        # Default at None, hence the fidelity values are not changed
        self.fidelity_exponent = None

        # The number of skip connections to have in the cell
        # If this set to None (default) No skip connections will be added to the cell
        # Maximum is the maximum number of operations.
        self.parameter_free_op_increase_type = None
        self.ratio_parameter_free_op_in_cell = None

        # Manually adjust a certain set of hyperparameters
        self.parameter_change_dict = None

        # Save predefined fidelity multiplier
        self.fidelity_multiplier = {
            'SimpleLearningrateSchedulerSelector:cosine_annealing:T_max': 1.762734383267615,
            'NetworkSelectorDatasetInfo:darts:init_channels': 1.3572088082974532,
            'NetworkSelectorDatasetInfo:darts:layers': 1.2599210498948732
        }
        self.fidelity_starts = {
            'SimpleLearningrateSchedulerSelector:cosine_annealing:T_max': 50,
            'NetworkSelectorDatasetInfo:darts:init_channels': 8,
            'NetworkSelectorDatasetInfo:darts:layers': 5
        }

    def __getitem__(self, path):
        """
        Load the results from results.json
        :param path: Path to results.json
        :return:
        """
        json_file = json.load(open(path, 'r'))
        config_dict = json_file['optimized_hyperparamater_config']

        config_space_instance = self.query_config_dict(config_dict)
        val_accuracy = json_file['info'][0]['val_accuracy']
        test_accuracy = json_file['test_accuracy']
        return config_space_instance, val_accuracy, test_accuracy, json_file

    def get_runtime(self, path):
        """
        Load the runtime from results.json
        :param path: Path to results.json
        return:
        """
        json_file = json.load(open(path, 'r'))
        config_dict = json_file['optimized_hyperparamater_config']

        config_space_instance = self.query_config_dict(config_dict)
        runtime = json_file['runtime']
        return config_space_instance, runtime

    def query_config_dict(self, config_dict):
        # Evaluation methods
        # Scale the hyperparameters if needed
        if self.fidelity_exponent is not None:
            config_dict = self.scale_fidelity(config_dict)

        # Add selected parameter free op
        if self.ratio_parameter_free_op_in_cell is not None:
            config_dict = self.add_selected_parameter_free_op(config_dict)

        # Change a selection of parameters
        if self.parameter_change_dict is not None:
            config_dict = self.change_parameter(config_dict)

        # Create the config space instance based on the config space
        config_space_instance = \
            self.convert_config_dict_to_configspace_instance(self.config_space, config_dict=config_dict)

        return config_space_instance

    def add_selected_parameter_free_op(self, config_dict):
        """
        Add selected parameter free operation to the config dict
        :param config_dict:
        :return:
        """
        assert self.parameter_free_op_increase_type in ['max_pool_3x3',
                                                        'avg_pool_3x3',
                                                        'skip_connect'], 'Unknown parameter-free op was selected.'
        # Dictionary containing operations
        cell_op_dict_sel_param_free = {'normal': {}, 'reduce': {}}
        cell_op_dict_non_sel_param_free = {'normal': {}, 'reduce': {}}

        for cell_type in ['normal']:
            for edge in range(0, 14):
                key = 'NetworkSelectorDatasetInfo:darts:edge_{}_{}'.format(cell_type, edge)
                op = config_dict.get(key, None)
                if op is not None:
                    if op == self.parameter_free_op_increase_type:
                        cell_op_dict_sel_param_free[cell_type][key] = op
                    else:
                        cell_op_dict_non_sel_param_free[cell_type][key] = op

        # Select random subset of operations which to turn to selected parameter-free op
        for cell_type in ['normal', 'reduce']:
            num_sel_param_free_ops = len(cell_op_dict_sel_param_free[cell_type].values())
            num_non_sel_param_free_ops = len(cell_op_dict_non_sel_param_free[cell_type].values())

            num_ops = num_sel_param_free_ops + num_non_sel_param_free_ops
            desired_num_sel_param_free_ops = np.round(num_ops * self.ratio_parameter_free_op_in_cell).astype(np.int)
            remaining_num_sel_param_free_op = desired_num_sel_param_free_ops - num_sel_param_free_ops

            if remaining_num_sel_param_free_op > 0:
                # There are still more selected parameter free operations to add to satisfy the ratio of
                # sel param free op. Therefore override some of the other operations to be parameter free op.
                sel_param_free_idx = np.random.choice(num_non_sel_param_free_ops, remaining_num_sel_param_free_op,
                                                      replace=False)
                for idx, (key, value) in enumerate(cell_op_dict_non_sel_param_free[cell_type].items()):
                    if idx in sel_param_free_idx:
                        config_dict[key] = self.parameter_free_op_increase_type
        return config_dict

    def scale_fidelity(self, config_dict):
        """
        Scale the fidelity of the current sample
        :param config_dict:
        :return:
        """
        for name, value in self.fidelity_multiplier.items():
            config_dict[name] = int(config_dict[name] * value ** self.fidelity_exponent)
        return config_dict

    def change_parameter(self, config_dict):
        for name, value in self.parameter_change_dict.items():
            config_dict[name] = value
        return config_dict

    def convert_config_dict_to_configspace_instance(self, config_space, config_dict):
        """
        Convert a config dictionary to configspace instace
        :param config_space:
        :param config_dict:
        :return:
        """

        def _replace_str_bool_with_python_bool(input_dict):
            for key, value in input_dict.items():
                if value == 'True':
                    input_dict[key] = True
                elif value == 'False':
                    input_dict[key] = False
                else:
                    pass
            return input_dict

        # Replace the str true with python boolean type
        config_dict = _replace_str_bool_with_python_bool(config_dict)
        config_instance = CS.Configuration(config_space, values=config_dict)
        return config_instance

    @staticmethod
    def load_config_space(path):
        """
        Load ConfigSpace object
        As certain hyperparameters are not denoted as optimizable but overriden later,
        they are manually overriden here too.
        :param path:
        :return:
        """
        with open(os.path.join(path), 'r') as fh:
            json_string = fh.read()
            config_space = config_space_json_r_w.read(json_string)

        # Override the constant hyperparameters for num_layers, init_channels and
        config_space._hyperparameters.pop('NetworkSelectorDatasetInfo:darts:layers', None)
        num_layers = CSH.UniformIntegerHyperparameter(name='NetworkSelectorDatasetInfo:darts:layers', lower=1,
                                                      upper=10000)
        config_space._hyperparameters.pop('SimpleLearningrateSchedulerSelector:cosine_annealing:T_max', None)
        t_max = CSH.UniformIntegerHyperparameter(name='SimpleLearningrateSchedulerSelector:cosine_annealing:T_max',
                                                 lower=1, upper=10000)
        config_space._hyperparameters.pop('NetworkSelectorDatasetInfo:darts:init_channels', None)
        init_channels = CSH.UniformIntegerHyperparameter(name='NetworkSelectorDatasetInfo:darts:init_channels', lower=1,
                                                         upper=10000)
        config_space._hyperparameters.pop('SimpleLearningrateSchedulerSelector:cosine_annealing:eta_min', None)
        eta_min_cosine = CSH.UniformFloatHyperparameter(
            name='SimpleLearningrateSchedulerSelector:cosine_annealing:eta_min', lower=0, upper=10000)

        config_space.add_hyperparameters([num_layers, t_max, init_channels, eta_min_cosine])
        return config_space

    def get_config_without_architecture(self, config_instance):
        """
        Remove the architecture parameters from the config.
        Currently this function retrieves the 5 parameters which are actually changed throughout the results:
        num_epochs, num_layers, num_init_channels (3 fidelities) + learning_rate, weight_decay
        :param config_instance:
        :return:
        """
        non_arch_hyperparameters_list = [
            config_instance._values['SimpleLearningrateSchedulerSelector:cosine_annealing:T_max'],
            config_instance._values['NetworkSelectorDatasetInfo:darts:init_channels'],
            config_instance._values['NetworkSelectorDatasetInfo:darts:layers'],
            config_instance._values['OptimizerSelector:sgd:learning_rate'],
            config_instance._values['OptimizerSelector:sgd:weight_decay']]

        return non_arch_hyperparameters_list


"""
SCRIPT TO CREATE THE GROUNDTRUTH DATA FOR THE NORMAL/REDUCTION CELL TOPOLOGY ANALYSIS.
"""

def get_graph_topologies():
    config_space = ConfigLoader('configspace.json').config_space

    # Sample architectures from search space
    sample_archs = [config_space.sample_configuration() for i in range(500_000)]

    # Extract the normal cell topologies
    normal_cell_topologies = defaultdict(list)
    for arch in tqdm(sample_archs):
        normal_cell_topology = {
            'NetworkSelectorDatasetInfo:darts:inputs_node_normal_{}'.format(idx): arch[
                'NetworkSelectorDatasetInfo:darts:inputs_node_normal_{}'.format(idx)] for idx in range(3, 6)
        }
        arch_hash = hash(frozenset(normal_cell_topology.items()))
        # if len(normal_cell_topologies[arch_hash]) < 10:
        normal_cell_topologies[arch_hash].append(arch.get_dictionary())

    #assert len(normal_cell_topologies) == 180, 'Not all connectivity patterns were sampled.'
    # assert all([len(archs) == 10 for normal_cell, archs in
                # normal_cell_topologies.items()]), 'The number of configs for each normal wasnt fulfilled'
    json.dump(normal_cell_topologies, open('normal_cell_topologies.json', 'w'))


def replace_normal_cell():
    normal_cell_topologies = json.load(open('normal_cell_topologies.json', 'r'))
    normal_cell_topologies_new = defaultdict(list)

    for normal_cell_topology, archs in normal_cell_topologies.items():
        for arch in archs:
            # Replace topology in reduction cell with normal cell's
            for inter_node in range(3, 6):
                arch['NetworkSelectorDatasetInfo:darts:inputs_node_reduce_{}'.format(inter_node)] = \
                    arch['NetworkSelectorDatasetInfo:darts:inputs_node_normal_{}'.format(inter_node)]
            # Replace operations in reduction cell with normal cell's
            for op_idx in range(14):
                # First remove the existing operation in reduction cell if it exists.
                arch.pop('NetworkSelectorDatasetInfo:darts:edge_reduce_{}'.format(op_idx), None)
                if 'NetworkSelectorDatasetInfo:darts:edge_normal_{}'.format(op_idx) in arch:
                    arch['NetworkSelectorDatasetInfo:darts:edge_reduce_{}'.format(op_idx)] = \
                        arch['NetworkSelectorDatasetInfo:darts:edge_normal_{}'.format(op_idx)]
        normal_cell_topologies_new[normal_cell_topology].append(archs)

    json.dump(normal_cell_topologies, open('normal_cell_topologies_replicated_normal_and_reduction_cell.json', 'w'))


if __name__ == "__main__":
    get_graph_topologies()
    # replace_normal_cell()