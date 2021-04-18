width = 2.20
height = 2.20

import os
import sys
import importlib

from utils import path_to_module, remove_suffix, import_file

dict_ = {
    'width': width,
    'height': height,
    'arena_dicts':[
    ]
}

arena_files = {
    './dict_arena_rec_max.py'
}

Arena_Params = []

for arena_file in arena_files:
    Arena_Param = import_file(arena_file, start_path=__file__, main_path=sys.path[0])
    Arena_Params.append(Arena_Param)
    arena_dict = Arena_Param.dict_
    '''
    for key, value in arena_dict.items():
        print('%s:%s'%(key, value))
    #arena_type = arena_dict['type']
    '''
    dict_['arena_dicts'].append(arena_dict)

    '''
    if arena_type in ['rec', 'rec_max', 'square']:
        dict_arenas['arena_dicts'].append({  # arena 0
            'type':'rec_max',
            'width': arena_width,
            'height': arena_height, 
        })
    elif arena_type in ['polygon']:
        # to be implemented
        pass
    elif arena_type in ['c', 'circle']:
        dict_arenas['arena_dicts'].append({
            'type':'circle',
            'width': arena_width,
            'height': arena_height,
            'center_coord':[0.0, 0.0],
            'radius': arena_width / 2.0,                   
        })
    else:
        raise Exception('Invalid arena type: %s'%str(arena_type))
    '''
def interact(env_info):
    for Arena_Param in Arena_Params:
        Arena_Param.interact(env_info)
    return
