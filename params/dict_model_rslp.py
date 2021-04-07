separate_ei = True
N_num = 1024
main_loss = 'MSE'
act_coeff = 0.0
weight_coeff = 0.0
noise_coeff = 0.0
bias = True
time_const = 0.1
act_func = 'tanh'
init_method = 'mlp'
if init_method in ['mlp']:
    init_method_dict = {
        'type': 'mlp',
        'act_func': decoder_act_func,
        'bias': True,
        'lr': lr_decoder,
        'batch_norm': decoder_batch_norm, # batch norm automatically alterate scale of weights and biases are to appropriate scale.
        'act_func_on_last_layer': False,
        'bias_on_last_layer': False,
    },

if separate_ei:
    E_ratio = 0.8
    cons_weight = ['r', 'f']
    time_const_e = time_const_i = time_const
    act_func_e = act_func_i = act_func

# init weight params
if separate_ei:
    coeff = 1.0e-1
else:
    coeff = 1.0
init_weight = {
    'i': ['glorot', coeff],
    'o': ['glorot', coeff],
    'r': ['glorot', coeff],
}

#input_mode = 'v_xy'

dynamic_weight_coeff = {
    'enable': True,
    'ratio_to_main_loss': 0.10,
    'ratio_to_main_loss_min': 0.05,
    'ratio_to_main_loss_max': 0.20,
    'target':['r']
}

disable_connection = []

dict_ = {
    'name': 'Navi',
    'type': 'rnn', # 'rnn' for RNN_Navi, 'lstm' for LSTM_Navi, 'linear' for Linear_Navi.
    #'task': None, # to be determined
    'input_num': None, # to be determined,
    'output_num': None, # to be determined
    'N_num': N_num,
    'init_weight': init_weight,
    'init_method': 'mlp', # method to init network state at t=0.
    'separate_ei': separate_ei,
    'cons_method': 'abs',
    'input_mode': None, # to be determined
    'bias': bias,
    'loss':{
        'main':{
            'type': main_loss,
            'coeff': 1.0
        },
        'act': act_coeff,
        'weight': weight_coeff,
        'dynamic_weight_coeff': dynamic_weight_coeff,
    },
    'mask': [],
    'bias': bias,
    'input_num': N_num,
    'output_num': None,
    'no_self': True,
    'init_weight': init_weight, # method to init weight.
    'N_num': N_num,
    'cons_weight': cons_weight,
    #'separate_ei': separate_ei,
    'mask': [],
    'separate_ei': separate_ei,
    'disable_connection': disable_connection,
}

if separate_ei:
    dict_['E_num'] = int(N_num * E_ratio)
    dict_['I_num'] = N_num - dict_['E_num']
    dict_['cons_weight'] = dict_['cons_weight'] = cons_weight
    dict_['cons_method'] = 'abs'
    dict_['time_const_e'] = time_const_e
    dict_['time_const_i'] = time_const_i
    dict_['act_func_e'] = act_func_e
    dict_['act_func_i'] = act_func_i
else:
    dict_['time_const'] = time_const
    dict_['act_func'] = act_func

def interact(env_info):
    agent_dict = env_info['agent_dict']
    task = agent_dict['task']
    dict_['device'] = env_info['device']
    '''
    if task in ['pc', 'pc_coords']:
    '''
    return