import sys
import re

tasks = [
    {
        'name': 'add_sys_path',
        'path': '/home/limoufan/wangweifan/Libs/' # img-15
        #'path': '/data4/wangweifan/Libs/' # img-16
    },
    {
        'name': 'add_sys_path',
        'path': [
            './src/Models/', 
            './src/Optimizers/', 
            './src/'
        ],
    },
]

def main():
    for task in tasks:
        name = task['name']
        if name in ['add_sys_path']:
            if isinstance(task['path'], str):
                sys.path.append(task['path'])
            elif isinstance(task['path'], list):
                for path in task['path']:
                    sys.path.append(path)
            else:
                raise Exception('add_sys_path: Invalid path type: %s'%type(task['path']))
        elif name in ['get_sys_type']:
            global sys_type
            sys_type = get_sys_type()
        else:
            raise Exception('sys_config: Invalid task name: %s'%task['name'])

Libs_path = {
    'wwf-pc':'A:/Software_Projects/Libs/',
    'srthu2':'/data4/wangweifan/Libs/',
}

sys_paths = []

def set_sys_path():
    sys_type = get_sys_type()
    if sys_type in ['windows']:
        sys_paths.append(Libs_path['wwf-pc'])
    elif sys_type in ['linux']:
        sys_paths.append(Libs_path['srthu2'])
    else:
        raise Exception('Unable to add Libs path. Unknown system type.')

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

main()