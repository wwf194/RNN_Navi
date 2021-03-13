
dict_arenas = {
    'arena_dicts':[
    ]
}

arena_files = {
    
}

for arena_type in arena_types:
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

def interact(env_info):
    return
