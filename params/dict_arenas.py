import os
import sys
import importlib

from utils import path_to_module, remove_suffix, import_file

#print(__file__) # absolute path of this script.

print(os.path.relpath(__file__, sys.path[0]))

rel_path = os.path.relpath(__file__, sys.path[0])

arena_width = 1.10
arena_height = 1.10

dict_ = {
    'width': 1.10,
    'height': 1.10,
    'arena_dicts':[
    ]
}

arena_files = {
    './dict_arena_square.py'
}

Arena_Params = []

for arena_file in arena_files:
    #Arena_Param = importlib.import_module(remove_suffix(path_to_module(rel_path), '.py'))
    Arena_Param = import_file(arena_file, start_path=__file__, main_path=sys.path[0])
    Arena_Params.append(Arena_Param)
    arena_dict = Arena_Param.dict_
    arena_type = arena_dict['type']
    dict_['arena_dicts'].append(arena_dict)
    '''
    if arena_type in ['rec', 'rec_max', 'square']:
        dict_['arena_dicts'].append({  # arena 0
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
