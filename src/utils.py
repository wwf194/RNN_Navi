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

def build_optimizer(dict_, load=False):
    import Optimizers
    type_ = dict_['type']
    if type_ in ['BP', 'bp']:
        return Optimizers.Optimizer_BP(dict_, load=load)
    elif type_ in ['CHL', 'chl']:
        return Optimizers.Optimizer_CHL(dict_, load=load)
    elif type_ in ['TP', 'tp']:
        return Optimizers.Optimizer_TP(dict_, load=load)
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