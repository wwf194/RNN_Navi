dict_ = {  # arena 0
    'type':'rec_max',
    'width': None,
    'height': None, 
}

def interact(env_info):
    #print('interacting')
    arenas_dict = env_info['arenas_dict']
    dict_['width'] = arenas_dict['width']
    dict_['height'] = arenas_dict['height']

