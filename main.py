import json
import args


def get_config_sys_from_json():
    with open('./config_sys.jsonc', 'r') as f:
        config_sys = json.load(f)
    #systype = config_sys['os.type']
    #print(systype)
    return config_sys

def build_model(args):
    model_type = args['name']
    
    if model_type in ['rnn']:
        build_rnn(args)
def build_rnn(args):
    from Models import rnn
    return rnn(args)

do_task = {
    'build_model': build_model
}

def main():
    with open('./config_tasks.json') as f:
        config_tasks = json.load(f)
    
    tasks = config_tasks['tasks']
    for task in tasks:
        do_task[task['name']](task['arg'])        




if __name__=="__main__":
    get_config_sys_from_json