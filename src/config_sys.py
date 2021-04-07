import sys
import re

Libs_path = {
    'WWF-PC':'A:/Software_Projects/Libs/',
    'srthu2':'/data4/wangweifan/Libs/',
}

sys_paths = ['./src/Models/', './src/Optimizers/', './src/']

def set_sys_path():
    sys_type = get_sys_type()
    if sys_type in ['windows']:
        sys_paths.append(Libs_path['WWF-PC'])
    elif sys_type in ['linux']:
        sys_paths.append(Libs_path['srthu2'])
    else:
        raise Exception('Cannot add Libs path. Unknown system type.')

    for path in sys_paths:
        sys.path.append(path)

def get_sys_type():
    if re.match(r'win',sys.platform) is not None:
        sys_type = 'windows'
    elif re.match(r'linux',sys.platform) is not None:
        sys_type = 'linux'
    else:
        sys_type = 'unknown'
    return sys_type

set_sys_path()