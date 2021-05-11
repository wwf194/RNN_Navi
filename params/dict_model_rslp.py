separate_ei = False
N_num = 1024 * 2

noise_coeff = 0.0
r_bias = True # whether to add bias when cal h from u through recurrent weight.
time_const = 1.0
act_func = 'relu'
init_method = 'linear'
if init_method in ['linear']:
    init_method_dict = {
        'type': 'linear',
        #'act_func': decoder_act_func,
        'bias': False,
        #'batch_norm': decoder_batch_norm, # batch norm automatically alterate scale of weights and biases are to appropriate scale.
        'act_func_on_last_layer': False,
        'bias_on_last_layer': False,
    }

input_method = 'linear'
if input_method in ['linear']:
    input_method_dict = {
        'type': 'linear',
        'bias': True,
        'bias_on_last_layer': True,
        'act_func_on_last_layer': False,
    }

if separate_ei:
    E_ratio = 0.8
    weight_Dale = ['r', 'f']
    time_const_e = time_const_i = time_const
    act_func_e = act_func_i = act_func

# init weight params
if separate_ei:
    coeff = 1.0e-1
else:
    coeff = 1.0e-1
init_weight = {
    'i': ['glorot', coeff],
    'o': ['glorot', coeff],
    'r': ['glorot', coeff],
}

#input_mode = 'v_xy'
disable_connection = []

dict_ = {
    'name': 'Navi',
    'type': 'rslp', # 'rslp' for RSLP_Navi, 'lstm' for LSTM_Navi, 'linear' for Linear_Navi.
    #'task': None, # to be determined
    'input_num': None, # to be determined,
    'input_init_num': None, # to be determined,
    'output_num': None, # to be determined,
    'N_num': N_num,
    'init_weight': init_weight,
    'init_method': init_method_dict, # method to init network state at t=0.
    'input_method': input_method_dict,
    'separate_ei': separate_ei,
    'cons_method': 'abs',
    'input_mode': None, # to be determined
    'r_bias': r_bias,
    'loss': None, # to be set
    'mask': [],
    'no_self': True,
    'init_weight': init_weight, # method to init weight.
    #'separate_ei': separate_ei,
    'mask': [],
    'separate_ei': separate_ei,
    'disable_connection': disable_connection,
}

if separate_ei:
    dict_['E_num'] = int(N_num * E_ratio)
    dict_['I_num'] = N_num - dict_['E_num']
    dict_['weight_Dale'] = weight_Dale
    dict_['cons_method'] = 'abs'
    if time_const_e == time_const_i:
        dict_['time_const'] = time_const_e
    else:
        dict_['time_const_e'] = time_const_e
        dict_['time_const_i'] = time_const_i
    if act_func_e == act_func_i:
        dict_['act_func'] = act_func_e
    else:
        dict_['act_func_e'] = act_func_e
        dict_['act_func_i'] = act_func_i
else:
    dict_['time_const'] = time_const
    dict_['act_func'] = act_func

def interact(env_info):
    agent_dict = env_info['agent']
    task = agent_dict['task']
    dict_['device'] = env_info['device']
    dict_['loss'] = agent_dict['loss']
    '''
    if task in ['pc', 'pc_coord']:
    '''
    return