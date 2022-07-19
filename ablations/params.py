######################################################################################
# Copyright (c) Colin White, Naszilla, 
# https://github.com/naszilla/naszilla
######################################################################################import sys


def algo_params(method_str):
    """
      Return params list based on param_str.
      These are the parameters used to produce the figures in the paper
      For AlphaX and Reinforcement Learning, we used the corresponding github repos:
      https://github.com/linnanwang/AlphaX-NASBench101
      https://github.com/automl/nas_benchmarks
    """
    params = []

    if method_str == 'local_search': 
        params.append({'algo_name':'local_search', 'total_queries':300})
        params.append({'algo_name':'local_search', 'total_queries':300, 'stop_at_maximum':False})
        params.append({'algo_name':'local_search', 'total_queries':300, 'query_full_nbhd':False})

    elif method_str == 'random':
        params.append({'algo_name':'random', 'total_queries':300})

    elif method_str == 'dngo':
        params.append({'algo_name':'dngo', 'total_queries':300})

    elif method_str == 'bo_gp':
        params.append({'algo_name':'bo_gp', 'total_queries':300})        
        print('Not implemented yet: {}'.format(param_str))
        sys.exit()

    elif method_str == 'bananas':
        params.append({'algo_name':'bananas', 'total_queries':300})
    
    elif method_str == 'weighted_bo':
        params.append({'algo_name':'weighted_bo', 'retrain_sample_size':100, 'total_queries':300, 'weight_factor':10e-3, 'M':10, 'r':5})
   
    elif method_str == 'gradient':
        print('Not implemented yet: {}'.format(param_str))
        sys.exit()
   
    else:
        print('invalid algorithm params: {}'.format(param_str))
        sys.exit()

    print('\n* Running experiment: ' + method_str)
    return params


def meta_neuralnet_params(param_str):

    if param_str == 'standard':
        metanet_params = {'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}
        params = {'ensemble_params':[metanet_params for _ in range(5)]}

    elif param_str == 'diverse':
        metanet_params = {'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}
        ensemble_params = [
            {'loss':'mae', 'num_layers':10, 'layer_width':20, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0},
            {'loss':'mae', 'num_layers':5, 'layer_width':5, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0},
            {'loss':'mae', 'num_layers':5, 'layer_width':30, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0},
            {'loss':'mae', 'num_layers':30, 'layer_width':5, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0},
            {'loss':'mae', 'num_layers':30, 'layer_width':30, \
            'epochs':150, 'batch_size':32, 'lr':.01, 'regularization':0, 'verbose':0}
        ]
        # TODO: this can be returned without a dictionary (update the algorithms that use metann_params)
        params = {'ensemble_params':ensemble_params}

    else:
        print('Invalid meta neural net params: {}'.format(param_str))
        raise NotImplementedError()

    return params