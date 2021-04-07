task = 'pc'
input_mode = 'v_xy'
step_num = 50


pc_num = 512
pc_type = 'diff_gaussian'

act_center = 1.0
act_decay = 0.15
norm_local = True # make all pc fire rate at one location sum to 1, so that it can be interpreted as a probability distribution.

if pc_type in ['diff_gaussian']:
    act_ratio = 2.0
    act_positive = True # raise pc act level so that minimum activation is above zero.

# set up place cells dict

dict_ = {
    'step_num': step_num,
    'input_mode': input_mode,
    'task': task
}

pc_dict = { # place cells dict
    'type': pc_type,
    'N_num': pc_num,
    'act_center': act_center, # peak activation.
    'act_decay': act_decay, # distance when place cells activation decays to exp{-1}.
    'norm_local': norm_local,
    'arena_index': 0, # default: place cells uniforms fill the first arena
    'device': None # to be set
}
if pc_type in ['diff_gaussian']:
    pc_dict['act_ratio'] = act_ratio
    pc_dict['act_positive'] = act_positive

dict_['place_cells'] = pc_dict

def interact(env_info):
    device = env_info['device']
    model_dict = env_info['model_dict']
    pc_dict['device'] = device
    if input_mode in ['v_xy']:
        model_dict['input_num'] = 2 # (v_x, v_y)
    elif input_mode in ['v_hd']:
        model_dict['input_num'] = 3 # (cos, sin, v)
    else:
        raise Exception('Invalid input mode: %s'%input_mode)

    if task in ['pc']:
        model_dict['input_init_num'] = pc_num
        model_dict['output_num'] = pc_num # pc_act
    elif task in ['coords']:
        model_dict['input_init_num'] = 2 # (x, y)
        model_dict['output_num'] = 2 # (x, y)
    else:
        raise Exception('Invalid task: %s'%task)
    return