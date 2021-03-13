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

sys.path.append('./src/')

import config_sys
from utils import build_model, build_agent, build_arenas, build_optimizer, build_trainer
from utils import scan_files, copy_files, path_to_module, remove_suffix, select_file, ensure_path, get_device
from config import Options
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
parser.add_argument('-pp', '--param_path', dest='param_path', type=str, default=None, help='path to folder that stores param dict files.')
parser.add_argument('-cf', '--config', dest='config', type=str, default=None, help='name of config file')
args = parser.parse_args()
parser.add_argument('-d', '-device', dest='device', type=str, default='None', help='device')
#parser.add_argument('-ei', '-separate_ei', dest='separate_ei', type=str, default=None)
parser.add_argument('-nabt', '-no_anal_before_train', dest='no_anal_before_train', action='store_true')
parser.add_argument('-pp', '-params_path', dest='params_path', type=str, default=None, help='path to folder that stores params files.')
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
    trainer.bind_model(model)
    optimizer.bind_model(model)
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
    
    if options.task in ['pc', 'pc_coords']:
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

    component_dicts = get_param_dicts(args)
    model_dict = component_dicts['model_dict']
    arenas_dict = component_dicts['arenas_dict']
    agent_dict = component_dicts['agent_dict']
    optimizer_dict = component_dicts['model_dict']
    trainer_dict = component_dicts['trainer_dict']
    
    trainer = build_trainer(trainer_dict)
    agent = build_agent(agent_dict)
    arenas = build_arenas(arenas_dict)
    # model can be RSLP, RMLP, RCNN ...
    model = build_model(model_dict)
    # optimizer can be BP, TP or CHL optimizer.
    optimizer = build_optimizer(optimizer_dict)
    optimizer.bind_model(model)
    optimizer.bind_trainer(trainer)

    trainer.bind_arenas(arenas)
    trainer.bind_model(model)
    trainer.bind_optimizer(optimizer)
    trainer.bind_agent(agent)
    trainer.train() # the model needs some data from agent to get response properties.

def load():
    loaded_items = {}

    return loaded_items

def train_simplified():
    options = Options(mode='init', args=args)
    options.build() # build DataLoader, Agent, Arena, Optimizer, etc.    

    options.save(save_path='./config/beforeTrain/')
    options.trainer.train()
    options.save(save_path='./config/afterTrain/')

def scan_param_files(path):
    if not path.endswith('/'):
        path.append('/')
    model_files = scan_files(path, r'dict_model(.*)\.py', raise_not_found_error=False)
    optimizer_files = scan_files(path, r'dict_optimizer(.*)\.py', raise_not_found_error=False)
    trainer_files = scan_files(path, r'dict_trainer(.*)\.py', raise_not_found_error=False)
    agent_files = scan_files(path, r'dict_agent(.*)\.py', raise_not_found_error=False)
    config_files = scan_files(path, r'config(.*)\.py', raise_not_found_error=False)

    '''
    if raise_not_found_error: # raise error if did not find any param dict
        if len(model_files)==0:
            raise Exception('No available model param dict in %s'%str(path))
        if len(optimizer_files)==0:
            raise Exception('No available optimizer param dict in %s'%str(path))
        if len(trainer_files)==0:
            raise Exception('No available trainer param dict in %s'%str(path))
        if len(agent_files)==0:
            raise Exception('No available agent param dict in %s'%str(path)) 
    '''
    return {
        'model_files': model_files,
        'optimizer_files': optimizer_files,
        'trainer_files': trainer_files,
        'agent_files': agent_files,
        'config_files': config_files
    }
    '''
    files_path = os.listdir(path)
    pattern_model = re.compile(r'dict_model(.*)\.py')
    pattern_optimizer = re.compile(r'dict_optimizer(.*)\.py')
    pattern_trainer = re.compile(r'dict_trainer(.*)\.py')
    patern_agent = re.compile(r'dict_agent(.*)\.py')
    model_files, optimizer_files, trainer_files, agent_files = [], [], [], []
    for file_name in files_path:
        #print(file_name)
        if pattern_model.match(file_name) is not None:
            model_files.append(file_name)
        elif pattern_optimizer.match(file_name) is not None:
            optimizer_files.append(file_name)
        elif pattern_trainer.match(file_name) is not None:
            trainer_files.append(file_name)
        elif patern_agent.match(file_name) is not None:
            agent_files.append(file_name)
        else:
            #warnings.warn('Unidentifiable param dict: %s'%str(file_name))
            pass

    # remove folders
    for files in [model_files, optimizer_files, trainer_files, agent_files]:
        for file in files:
            if os.path.isdir(file):
                warnings.warn('%s is a folder, and will be ignored.'%(path + file))
                files.remove(file)

    return model_files, optimizer_files, trainer_files, agent_files
    '''

def get_param_files(args, verbose=True):
    path = args.param_path
    if path is None:
        path = './params/'
    if not path.endswith('/'):
        path += '/'
    files = scan_param_files(path)
    
    model_files = files['model_files']
    agent_files = files['agent_files']
    arenas_files = files['arenas_files']
    optimizer_files = files['optimizer_files']
    trainer_files = files['trainer_files']
    config_files = files['config_files']

    #print(model_files)
    #print(optimizer_files)
    #print(trainer_files)
    #print(agent_files)

    model_str, = args.model
    agent_str = args.agent
    arenas_str = args.arenas
    optimizer_str = args.optimizer
    trainer_str = args.trainer
    
    files_str = [model_str, optimizer_str, trainer_str, agent_str, arenas_str]
    component_files = [model_files, agent_files, arenas_files, optimizer_files, trainer_files]
    
    use_config_file = False
    if len(config_files)==1:
        sig = True
        for files in component_files:
            if len(files)>1 or len(files)==0:
                sig = False
        if sig:
            use_config_file = True
    if args.config is not None:
        use_config_file = True

    if use_config_file: # get param files according to a config file.
        if verbose:
            print('Setting params according to config file.')
        config_file = select_file(args.config, config_files, default_file=None, 
            match_prefix='config_', match_suffix='.py', file_type='config')
        print(config_file)
        Config_Param = importlib.import_module(path_to_module(path) + remove_suffix(config_file))
        try:
            model_file = Config_Param.model_file
            optimizer_file = Config_Param.optimizer_file
            trainer_file = Config_Param.trainer_file
            agent_file = Config_Param.agent_file
        except Exception:
            raise Exception('Cannot read file name from %s'%(path + config_file))
    
        for file in [model_file, optimizer_file, trainer_file, agent_file]:
            if not os.path.exists(path + file):
                raise Exception('FileNotFoundError: %s'%(path + file))
        return {
            'model_file': model_file,
            'optimizer_file': optimizer_file,
            'trainer_file': trainer_file,
            'agent_file': agent_file,
            'config_file': config_file,
            'files_path': path,
        }
    else: # get param files directly
        if verbose:
            print('Setting params according to model, optimzier, trainer, and agent param files.')
        if len(model_files)==0:
            raise Exception('No available model param file.')
        elif len(model_files)==1:
            model_file = model_files[0]
            if verbose:
                print('Using the only available model file: %s'%model_file)          
        else:
            model_file = select_file(model_str, model_files, default_file='dict_model_rslp.py', 
                match_prefix='dict_model_', match_suffix='.py', file_type='model')

        if len(agent_files)==0:
            raise Exception('No available agent param file.')
        elif len(agent_files)==1:
            agent_file = agent_files[0]
            if verbose:
                print('Using the only available agent file: %s'%agent_file)
        else:
            agent_file = select_file(agent_str, agent_files, default_file='dict_agent_cifar10.py', 
                match_prefix='dict_agent_', match_suffix='.py', file_type='data loader')

        if len(arenas_files)==0:
            raise Exception('No available agent param file.')
        elif len(arenas_files)==1:
            agent_file = arenas_files[0]
            if verbose:
                print('Using the only available agent file: %s'%agent_file)
        else:
            arenas_file = select_file(arenas_str, arenas_files, default_file='dict_arenas_squaqre.py', 
                match_prefix='dict_agent_', match_suffix='.py', file_type='arenas')
        
        if len(optimizer_files)==0:
            raise Exception('No available optimizer param file.')
        elif len(optimizer_files)==1:
            optimizer_file = optimizer_files[0]
            if verbose:
                print('Using the only available optimizer file: %s'%optimizer_file)        
        else:
            optimizer_file = select_file(optimizer_str, optimizer_files, default_file='dict_optimizer_bp.py', 
                match_prefix='dict_optimizer_', match_suffix='.py', file_type='optimizer')

        if len(trainer_files)==0:
            raise Exception('No available trainer param file.')
        elif len(trainer_files)==1:
            trainer_file = trainer_files[0]
            if verbose:
                print('Using the only available trainer file: %s'%trainer_file)
        else:
            trainer_file = select_file(trainer_str, trainer_files, default_file='dict_trainer.py', 
                match_prefix='dict_trainer_', match_suffix='.py', file_type='trainer')


        #print(model_file)
        #print(optimizer_file)
        #print(trainer_file)
        #print(agent_file)
        
        return {
            'model_file': model_file,
            'arenas_file': arenas_file,
            'agent_file': agent_file,
            'optimizer_file': optimizer_file,
            'trainer_file': trainer_file,
            'files_path': path,
        }
def get_param_dicts(args):
    component_files = get_param_files(args)
    model_file = component_files['model_file']
    agent_file = component_files['agent_file']
    arenas_file = component_files['arenas_file']
    optimizer_file = component_files['optimizer_file']
    trainer_file = component_files['trainer_file']
    files_path = component_files['files_path']
    
    #print(path_to_module(files_path) + remove_suffix(model_file))
    Model_Param = importlib.import_module(path_to_module(files_path) + remove_suffix(model_file))
    model_dict = Model_Param.dict_

    Agent_Param = importlib.import_module(path_to_module(files_path) + remove_suffix(agent_file))
    agent_dict = Agent_Param.dict_

    Arenas_Param = importlib.import_module(path_to_module(files_path) + remove_suffix(arenas_file))
    agent_dict = Arenas_Param.dict_

    Optimizer_Param = importlib.import_module(path_to_module(files_path) + remove_suffix(optimizer_file))
    optimizer_dict = Optimizer_Param.dict_

    Trainer_Param = importlib.import_module(path_to_module(files_path) + remove_suffix(trainer_file))
    trainer_dict = Trainer_Param.dict_

    device = get_device(args)
    print('Using device: %s'%str(device))

    env_info = {
        'model_dict': model_dict,
        'optimizer_dict': optimizer_dict,
        'trainer_dict': trainer_dict,
        'agent_dict': agent_dict,
        'device': device
    }

    Model_Param.interact(env_info)
    Optimizer_Param.interact(env_info)
    Trainer_Param.interact(env_info)
    Agent_Param.interact(env_info)

    return model_dict, optimizer_dict, trainer_dict, agent_dict

def copy_project_files(args):
    path = args.path
    if path is None:
        raise Exception('copy_project_files: args.path must not be none. please give path to copy files to')
    ensure_path(args.path)
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
        'Optimizers.py',
        'Analyzer.py',
        #'config.py',
        'main.py',
        'config.py',
        'anal_grid.py',
        'utils.py',
        'utils_agent.py',
        'utils_arena.py',
        'utils_plot.py',
        'utils_anal.py',
        'config_sys.py',
    ]
    copy_files(file_list, path_from='./src/', path_to=path + 'src/')
    file_list = [
        'main.py',
        'params/__init__.py'
    ]
    copy_files(file_list, path_from='./', path_to=path)
    param_files = get_param_files(args)
    #param_files = list(map(lambda file:param_path + file, param_files))
    model_file = param_files['model_file']
    agent_file = param_files['agent_file']
    arenas_file = param_files['arenas_file']
    optimizer_file = param_files['optimizer_file']
    trainer_file = param_files['trainer_file']

    component_files = [model_file, agent_file, arenas_file, optimizer_file, trainer_file, agent_file]
    #print(component_files)
    copy_files(component_files, path_from=param_path, path_to=path + param_path)

if __name__=='__main__':
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
    elif task in ['train']:
        train()
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