# run this script to do different tasks

# python main.py --task copy --path ./Instances/TP-1/  //move necessary files for training and analysis to path.


# Every component, such as model, optimizer, trainer, agent, is initialized according to a dict in a .py file.

# run this script to do different tasks

# python main.py --task copy --path ./Instances/TP-1/  //move necessary files for training and analysis to path.

# Every component, such as model, optimizer, trainer, agent, is initialized according to a dict in a .py file.

import os
import sys
import re
import argparse
import warnings
import importlib
import shutil
from inspect import getmembers, isfunction

sys.path.append('./src/')
import ConfigSystem
#print(sys.path)
from utils import build_model, build_agent, build_arenas, build_Optimizer, build_trainer, copy_folder, cal_path_rel_main, get_items_from_dict
from utils import scan_files, copy_files, path_to_module, remove_suffix, select_file, EnsurePath, get_device, import_file, join_path
#from config import Options
from utils_anal import compare_traj, get_input_output

from Trainers import Trainer
import Models
import Optimizers
from Analyzer import *

parser = argparse.ArgumentParser(description='Parse args.')
parser.add_argument('-d', '--device', type=str, dest='device', default=None, help='device')
parser.add_argument('-t', '--task', type=str, dest='task', default=None, help='task to do')
parser.add_argument('-p', '--path', type=str, dest='path', default=None, help='a path to current directory. required in some tasks.')
parser.add_argument('-o', '--optimizer', dest='optimizer', type=str, default=None, help='optimizer type. BP, TP, CHL, etc.')
parser.add_argument('-tr', '--trainer', dest='trainer', type=str, default=None, help='trainer type.')
parser.add_argument('-m', '--model', dest='model', type=str, default=None, help='model type. RSLP, RMLP, RSLCNN, RMLCNN, etc.')
parser.add_argument('-a', '-arenas', dest='arenas', type=str, default='rec', help='arenas param file')
parser.add_argument('-dl', '--agent', dest='agent', type=str, default=None, help='data loader type.')
parser.add_argument('-cf', '--config', dest='config', type=str, default=None, help='name of config file')
#parser.add_argument('-ei', '-separate_ei', dest='separate_ei', type=str, default=None)
parser.add_argument('-nabt', '-no_anal_before_train', dest='no_anal_before_train', action='store_true')
parser.add_argument('-pp', '-param_path', dest='param_path', type=str, default=None, help='path to folder that stores params files.')
args = parser.parse_args()

def main():
    #test_act_map()
    train()
    #test_model()
    #test_agent()

def test_place_cells():
    options = Options(mode='init', args=args)
    options.dict_model['place_cells']['arena_index'] = 1
    options.build()
    model = options.model
    model.place_cells.plot_place_cells(save=True, save_path='../anal/place_cells/', save_name='place_cells_plot_circle.png')
    model.place_cells.plot_place_cells_coords(save=True, arena=options.arenas.get_arena(1), save_path='../anal/place_cells/', save_name='place_cells_coords_circle.png')     

def test_plot_weight():
    options = Options(mode='init', args=args)
    options.build()
    options.model.plot_weight(save=True, save_path='../anal/model/', save_name='weight_plot.png') 

def test_act_map():
    options = Options(mode='init', args=args)
    options.build()
    model = options.model
    model.place_cells.plot_place_cells(save=True, save_path='../anal/', save_name='place_cells_plot.png')
    options.agent.plot_walk_random(save=True, save_path='../anal/beforeTrain/')
    options.agent.anal_act(save=True, save_path='../anal/beforeTrain/act_map/',
                                save_name='act_map.png', plot_num='all',
                                model=options.model, trainer=options, arena=options.arenas.current_arena(),
                                separate_ei=options.model.separate_ei)
def test_anal_act():
    options = Options(mode='init', items=['arena', 'agent', 'trainer', 'optimizer', 'analyzer'], args=args)
    #options.build_model(load=True, dir_='../saved_models/Navi_epoch=afterTrain')
    save_path = '../anal/afterLoad/'
    options.build()
    options.build_model(load=True, dir_='../saved_models/Navi_epoch=afterTrain')
    options.model.receive_options(options)
    options.model.to(options.model.device)
    model, trainer, optimizer = options.model, options.trainer, options.optimizer
    model.plot_weight(save_path=save_path+'model/', save_name='model_weight_plot.png')
    #trainer.bind_model(model)
    #trainer.bind_agent(agent)
    #optimizer.bind_model(model)
    optimizer.bind_agent(agent)
    print('test_load_model: plotting heat map after load.')
    options.agent.plot_path(save=True, save_path=save_path, save_name='path_plot.png', model=options.model)
    options.agent.anal_act(save_path=save_path,
                            model=options.model, trainer=options.trainer, arena=options.arenas.current_arena(),
                            separate_ei=options.model.separate_ei)

def test_load_model():
    options = Options(mode='init', items=['arena', 'agent', 'trainer', 'optimizer', 'analyzer', 'model'], args=args)
    #options.build_model(load=True, dir_='../saved_models/Navi_epoch=afterTrain')
    options.build()
    print('test_model: options.model:%s'%str(options.model))
    options.model.place_cells.plot_place_cells(save=True, save_path='../anal/', save_name='place_cells_plot.png')

    print('test_load_model: plotting heat map before load.')
    options.agent.plot_path(save=True, save_path='../anal/beforeLoad/', save_name='path_plot.png', model=options.model)
    options.agent.anal_act(save_path='../anal/beforeLoad/act_map/',
                                save_name='act_map.png', plot_num='all',
                                model=options.model, trainer=options, arena=options.arenas.current_arena(),
                                separate_ei=options.model.separate_ei)

    options.model.save(save_path='../saved_models/', save_name='beforeTrain')
    options.build_model(load=True, dir_='../saved_models/beforeTrain')
    options.model.receive_options(options)
    options.model.to(options.model.device)
    print('test_load_model: plotting heat map after load.')
    options.agent.plot_path(save=True, save_path='../anal/afterLoad/', save_name='path_plot.png', model=options.model)
    options.agent.anal_act(save_path='../anal/afterLoad/act_map/',
                                save_name='act_map.png', plot_num='all',
                                model=options.model, trainer=options.trainer, arena=options.arenas.current_arena(),
                                separate_ei=options.model.separate_ei)



def test_model():
    options = Options(mode='init', items=['arena', 'agent', 'trainer', 'optimizer', 'analyzer'], args=args)
    options.build_model(load=True, dir_='../saved_models/Navi_epoch=afterTrain')
    print('test_model: options.model:%s'%str(options.model))
    options.build()
    options.model.place_cells.plot_place_cells(save=True, save_path='../anal/', save_name='place_cells_plot.png')

    options.agent.anal_act(save=True, save_path='../anal/beforeTrain/act_map/',
                                save_name='act_map.png', plot_num='all',
                                model=options.model, trainer=options.trainer, arena=options.arenas.current_arena(),
                                separate_ei=options.model.separate_ei)

def test_plot_path():
    '''
    options = Options(mode='init', items=['arena', 'agent', 'trainer', 'optimizer', 'analyzer', 'model'], args=args)
    options.build()
    options.agent.plot_path(save_path='../anal/', save_name='path_plot.png')
    '''

    options = Options(mode='init', items=['arena', 'agent', 'trainer', 'optimizer', 'analyzer'], args=args)
    options.build_model(load=True, dir_='../saved_models/Navi_epoch=afterTrain')
    print('test_model: options.model:%s'%str(options.model))
    options.build()
    options.agent.plot_path(save_path='../anal/', save_name='path_plot_afterTrain.png')

def test_agent():
    options = Options(mode='init', items=['arena', 'agent', 'model'], args=args)
    options.build()
    options.agent.plot_walk_random(save=True, save_path='../anal/agent/')

def test_arena():
    #options = Options(mode='init', items=['arena', 'agent', 'model'], args=args)
    options = Options(mode='init', items=['arena'], args=args)
    options.build()
    arenas = options.arenas
    arenas.plot_random_xy(save=True, save_path='../anal/arena/')

def test_arena_circle():
    #options = Options(mode='init', items=['arena', 'agent', 'model'], args=args)
    options = Options(mode='init', items=['arena', 'agent', 'model'], args=args)
    options.build()
    arenas = options.arenas
    arenas.plot_arenas(save=True, save_path='../anal/arena/')
    arenas.plot_random_xy(save=True, save_path='../anal/arena/')
    
    arenas.set_current_arena(1)
    options.agent.plot_walk_random(save=True, save_path='../anal/arena/')

def train_():
    options = Options(mode='init', args=args)
    options.build() # build DataLoader, Agent, Arena, Optimizer, etc.    
    model, trainer, optimizer = options.model, options.trainer, options.optimizer
    trainer.bind_model(model)
    optimizer.bind_model(model)
    optimizer.scheduler.verbose = True
    
    if options.task in ['pc', 'pc_coord']:
        options.model.place_cells.plot_place_cells(save=True, save_path='../anal/', save_name='place_cells_plot.png')
    '''
    options.agent.plot_walk_random(save=True, save_path='../anal/beforeTrain/')
    options.agent.anal_act(save=True, save_path='../anal/beforeTrain/',
                                save_name='act_map.png', plot_num='all',
                                model=options.model, trainer=options, arena=options.arenas.current_arena(),
                                separate_ei=options.model.separate_ei)

    '''
    options.save(save_path='./config/beforeTrain/')
    options.trainer.train()
    options.save(save_path='./config/afterTrain/')

def train(args=None, param_path=None, **kw):
    if args is None:
        args = kw.get('args')
    
    if param_path is None:
        if args.param_path is not None:
            param_path = args.param_path
        else:
            param_path = './params/'
        #sys.path.append(param_path)

    param_dict = get_param_dict(args)
    model_dict = param_dict['model']
    arenas_dict = param_dict['arenas']
    agent_dict = param_dict['agent']
    optimizer_dict = param_dict['optimizer']
    task_dict = param_dict['task']
    #trainer_dict = param_dict['trainer_dict']

    # overwrite dict items from args    
    if args.no_anal_before_train:
        if task_dict.get('train') is not None:
            task_dict['train']['anal_before_train'] = False

    #trainer = build_trainer(trainer_dict)
    agent = build_agent(agent_dict)
    arenas = build_arenas(arenas_dict)
    model = build_model(model_dict)
    #optimizer = build_Optimizer(optimizer_dict)
    #optimizer.bind_model(model)
    #optimizer.bind_trainer(trainer)

    #trainer.bind_arenas(arenas)
    #trainer.bind_model(model)
    #trainer.bind_optimizer(optimizer)
    #trainer.bind_agent(agent)
    #agent.bind_optimizer(optimizer)
    agent.bind_arenas(arenas)
    agent.bind_model(model)
    #trainer.train() # the model needs some data from agent to get response properties.
    agent.train(task_dict['train'])

def load():
    loaded_items = {}

    return loaded_items

def train_simplified():
    options = Options(mode='init', args=args)
    options.build() # build DataLoader, Agent, Arena, Optimizer, etc.    

    options.save(save_path='./config/beforeTrain/')
    options.trainer.train()
    options.save(save_path='./config/afterTrain/')

def scan_param_file(path, file_info):
    # file_info [(file_name, file_pattern)]
    if not path.endswith('/'):
        path.append('/')
    param_file = {}
    for file_name, file_pattern in file_info:
        param_file[file_name] = scan_files(path, file_pattern, raise_not_found_error=False)
    return param_file

def get_param_file(args, verbose=True):
    param_path = args.param_path
    if param_path is None:
        param_path = './params/'
    if not param_path.endswith('/'):
        param_path += '/'
    param_file = scan_param_file(param_path, [('config', r'config(.*)\.py')])
    config_files = param_file['config']

    #if use_config_file: # get param files according to a config file.
    if args.config is None:
        if len(config_files)==1:
            config_file = config_files[0]
        elif len(config_files)==0:
            raise Exception('Missing config file in %s.'%param_path)
        else:
            raise Exception('Multiple config files found in %s. Please specify one:\n  %s'%(param_path, config_files))
    else:
        config_file = select_file(args.config, config_files, default_file='config_rnn_ei_pc', 
            match_prefix='config_', match_suffix='.py', file_type='config')
    if verbose:
        print('Setting params according to config file: %s.'%config_file)
    Config_Param = import_file(param_path + config_file)
    #Config_Param = importlib.ImportModule(path_to_module(param_path) + remove_suffix(config_file))
    #print(Config_Param.dict_)
    #print('config_file: %s'%config_file)
    if config_file.startswith('./'):
        config_file.lstrip('./')
    param_file = Config_Param.dict_.update({
        'config': './' + config_file, # includa config_file itself as param_file
    })
    return {
        'param_file': Config_Param.dict_,
        'param_path': param_path,
    }
def get_param_dict(args, verbose=True):
    param_file, param_path = get_items_from_dict(get_param_file(args), ['param_file', 'param_path'])
    #print(param_file.keys())

    '''
    model_file = param_file['model'] # model_file are in form of relevant path to ./
    agent_file = param_file['agent']
    arenas_file = param_file['arenas']
    optimizer_file = param_file['optimizer']
    trainer_file = param_file['trainer']
    '''

    param_dict = {}

    Params = []
    for param_name, file_path in param_file.items():
        print(join_path(param_path, file_path))
        Param = import_file(join_path(param_path, file_path))
        param_dict[param_name] = Param.dict_
        Params.append(Param)

    device = get_device(args)
    if verbose:
        print('Using device: %s'%str(device))

    env_info = {
        'device': device,
        'args': args,
    }
    env_info.update(param_dict)

    for Param in Params:
        if 'interact' in list(map(lambda x:x[0], getmembers(Param, isfunction))): # check whether Param module has interact function.
            Param.interact(env_info)
            '''
            Model_Param.interact(env_info)
            Optimizer_Param.interact(env_info)
            Trainer_Param.interact(env_info)
            Agent_Param.interact(env_info)
            Arenas_Param.interact(env_info)
            '''
    return param_dict
def copy_project_files(args, verbose=True):
    path = args.path
    if path is None:
        raise Exception('copy_project_files: args.path must not be none. please give path to copy files to')
    EnsurePath(args.path)
    if args.param_path is None:
        param_path = './params/'
    #print(path)
    if not path.endswith('/'):
        path += '/'
    file_list = [ # necessary files
        #'cmd.py',
        'Models',
        'Agent.py',
        'Arenas.py',
        'Trainers.py',
        'Optimizers',
        'Analyzer.py',
        'anal_grid.py',
        'utils.py',
        'utils_agent.py',
        'utils_arena.py',
        'utils_plot.py',
        'utils_anal.py',
        'model.py',
        'utils_train.py',
    ]
    copy_files(file_list, path_from='./src/', path_to=path + 'src/', verbose=verbose)
    
    file_list = [
        'main.py',
        'ConfigSystem.py',
    ]
    copy_files(file_list, path_from='./', path_to=path, verbose=verbose)
    
    # select and copy param files
    param_file, param_path = get_items_from_dict(get_param_file(args), ['param_file', 'param_path'])
    #print('param_file: %s'%param_file)
    if isinstance(param_file, list):
        param_file = list(map(lambda file:join_path(param_path, file), param_file))
    elif isinstance(param_file, dict):
        param_file = list(map(lambda file:join_path(param_path, file), param_file.values()))
    else:
        raise Exception('Invalid param file type:%s'%type(param_file))
    #print('param_file: %s'%param_file)
    #input()
    param_file = get_required_file(param_file)

    #copy_files(param_file, path_from=os.path.abspath('./'), path_to=path, verbose=verbose)
    copy_files(param_file, path_from='./', path_to=path, verbose=verbose)

def backup(args):
    if args.path is None:
        path_to = '/data4/wangweifan/.backup/Navi-v3/' # default backup path
    else:
        path_to = args.path
    copy_folder(path_from=os.path.abspath('../'), path_to=path_to, exceptions=['../Instances/'])

if __name__=='__main__':
    main()
    #print(os.path.abspath('../'))
    #task = 'train' if args.task is None else args.task
    if args.task is None:
        task = 'train'
        warnings.warn('Task is not given from args. Using default task: train.')
    else:
        task = args.task
    if task in ['copy', 'copy_files', 'copy_file']: # copy necessary files for training and 
        copy_project_files(args)
    elif task in ['train']:
        train(args)
    elif task in ['backup']:
        backup(args)
    elif task in ['train_sim']:
        train_simplified()
    elif task in ['test_model']:
        test_model()
    elif task in ['test_load_model']:
        test_load_model()
    elif task in ['test_arena']:
        test_arena()
    elif task in ['test_agent', 'walk_random']:
        test_agent()
    elif task in ['test_plot_path', 'plot_path']:
        test_plot_path()
    elif task in ['test_arena_circle']:
        test_arena_circle()
    elif task in ['test_place_cells']:
        test_place_cells()
    elif task in ['test_plot_weight']:
        test_plot_weight()
    elif task in ['test_score', 'test_anal_act']:
        test_anal_act()
    else:
        raise Exception('Invalid task: %s'%task)