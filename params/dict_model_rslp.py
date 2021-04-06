separate_ei = True
N_num = 
main_loss = 'MSE'
act_coeff = 0.0
weight_coeff = 0.0
noise_coeff = 0.0
bias = True

if separate_ei:
    coeff = 1.0e-1
else:
    coeff = 1.0
init_weight = {
    'i': ['glorot', coeff],
    'o': ['glorot', coeff],
    'r': ['glorot', coeff],
}

input_mode = 'v_xy'

dict_ = {
    'name': 'Navi',
    'type': 'rnn', # 'rnn' for RNN_Navi, 'lstm' for LSTM_Navi, 'linear' for Linear_Navi.
    'task': None, # to be determined
    'input_num': None,# to be determined,
    'output_num': None, # to be determined
    'N_num': N_num,
    'init_weight': init_weight,
    'init_method':'zero',
    'separate_ei': separate_ei,
    'cons_method': 'abs',
    'input_mode': None, # to be determined
    'bias': bias,
    'loss':{
        'main':{
            'type': main_loss,
            'coeff':1.0
        },
        'act': act_coeff,
        'weight': weight_coeff,
        'dynamic_weight_coeff': dynamic_weight_coeff,
    },
    'mask': [],
}
dict_N={
    'bias': bias,
    'input_num': N_num,
    'output_num': output_num,
    'no_self': no_self,
    'init_method': init_method, # method to init network state at t=0.
    'init_weight': init_weight, # method to init weight.
    'N_num': N_num,
    'Dale': Dale,
    'separate_ei': separate_ei,
    'mask': [],
    'cons_method': cons_method,
    'separate_ei': separate_ei,
    'disable_connection': disable_connection,
}
dict_N = dict_N

if separate_ei:
    dict_N['E_num'] = int(N_num * e_ratio)
    dict_N['I_num'] = N_num - dict_N['E_num']
    dict_['Dale'] = Dale
    dict_N['Dale'] = Dale
    if locals().get('time_const_e') is None:
        dict_N['time_const'] = dict_N['time_const_e'] = dict_N['time_const_i'] = time_const
    else:
        dict_N['time_const_e'] = time_const_e
        dict_N['time_const_i'] = time_const_i
    if locals().get('act_func_e') is None:
        dict_N['act_func'] = dict_N['act_func_e'] = dict_N['act_func_i'] = act_func
    else:
        dict_N['act_func_e'] = act_func_e
        dict_N['act_func_i'] = act_func_i
else:
    dict_N['time_const'] = time_const
    dict_N['act_func'] = act_func

if task in ['pc', 'pc_coords']:
    dict_place_cells = {
        'type': pc_type,
        'N_num': pc_num,
        'act_center': act_center, # peak activation.
        'act_decay': act_decay, # distance when place cells activation decays to exp{-1}.
        'norm_local': norm_local,
        'arena_index': place_cells_arena_index,
    }
    if pc_type in ['diff_gaussian']:
        dict_place_cells.update({
            'act_positive': act_positive, # for diff_gaussian mode
            'act_ratio': act_ratio # for diff_gaussian mode
        })
    dict_['place_cells'] = dict_place_cells 


def interact(env_info):
    agent_dict = env_info['agent_dict']
    task = agent_dict['task']
    if task in ['pc', 'pc_coords']:
        
    


    return