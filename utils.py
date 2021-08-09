from utils_ import *
from utils_ import get_best_gpu

def build_model(dict_, load=False):
    import Models
    type_ = dict_['type']
    print('build_model: model_type: %s'%type_)
    if type_ in ['rnn', 'rslp']:
        return Models.RSLP_LIF(dict_=dict_, load=load)
    else:
        raise Exception('Options: Invalid model type: %s'%str(type_))

def build_agent(dict_, load=False):
    import Agent
    return Agent.Agent(dict_=dict_, load=load)

def build_arenas(dict_, load=False):
    import Arenas
    return Arenas.Arenas(dict_ = dict_, load=load)

def build_Optimizer(dict_, load=False, params=None, model=None):
    import Optimizers
    type_ = dict_['type']
    if type_ in ['BP', 'bp']:
        return Optimizers.Optimizer_BP(dict_, load=load, params=params, model=model)
    elif type_ in ['CHL', 'chl']:
        return Optimizers.Optimizer_CHL(dict_, load=load, params=params, model=model)
    elif type_ in ['TP', 'tp']:
        return Optimizers.Optimizer_TP(dict_, load=load, params=params, model=model)
    else:
        raise Exception('Invalid optimizer type: %s'%str(type_))

def build_trainer(dict_, load=False):
    import Trainers
    return Trainers.Trainer(dict_, load=load)

def get_device(args):
    if hasattr(args, 'device'):
        if args.device not in [None, 'None']:
            print(args.device)
            return args.device
    return get_best_gpu()

def get_required_file(file_start):
    if isinstance(file_start, str):
        file_start = [file_start]
    elif isinstance(file_start, tuple) or isinstance(file_start, set):
        file_start = list(file_start)

    file_list = file_start
    for file in file_start:
        get_required_file_recur(file, file_list)
    return file_list

def get_required_file_recur(file, file_list):
    # file_list: stores required files in format of relative path to ./
    if not file.startswith('/') and not file.startswith('./'):
        file = './' + file
    if not file in file_list:
        file_list.append(file)
    File = import_file(file)

    if 'file_required' in dir(File):
        file_required = File.file_required
        #print(type(file_required))
        if isinstance(file_required, list):
            pass
        elif isinstance(file_required, dict):
            file_required = file_required.values()
        elif isinstance(file_required, set) or isinstance(file_required, tuple):
            file_required = list(file_required)
        else:
            raise Exception('get_required_file_recur: Unknown file_required type: %s'%type(file_required))
        
        for file_rel in file_required:
            file_rel_main = cal_path_rel_main(path_rel=file_rel, path_start=File.__file__, path_main=__file__)
            #print('file_rel_main: %s'%file_rel_main)
            get_required_file_recur(file_rel_main, file_list)