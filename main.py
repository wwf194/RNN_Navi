import os
import sys
import argparse
import json
import json5
import warnings
import logging
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("task", nargs="?", default="DoTasks")
args = parser.parse_args()

#sys.path.append("./utils/")

import utils
from utils import set_logger_global, get_logger_global, add_log, add_warning

set_logger_global()
logger = get_logger_global()

def do_tasks():

    try: # catch all unhandled exceptions
        do_tasks_pre()
        #import utils_torch
        #utils.args_global = utils.json.JsonObj2PyObj(utils.args_global)
        do_tasks_main()
    except Exception:
        logger.error(traceback.format_exc())

def do_tasks_pre():
    tasks = utils.json.JsonFile2PyObj('./task_pre.jsonc')
    for task in tasks:
        TaskImplementFunction[task.name](task.args)
    import utils_torch
    utils_torch.set_logger(get_logger_global())

def do_tasks_main():
    import utils_torch
    tasks = utils.json.JsonFile2PyObj('./task.jsonc')
    for task in tasks:
        #add_log("main: doing task: %s"%task['name'])
        #print(task)
        TaskImplementFunction[task.name](task.args)

def build_object(args):
    if isinstance(args, list):
        for arg in args:
            _build_object(arg)
    elif isinstance(args, dict):
        _build_object(args)

def _build_object(args):
    import utils_torch
    Class = utils_torch.import_module(args.ModulePath)
    obj = Class.init_from_param(getattr(utils.args_global.ParamDicts, args.ParamName))
    setattr(utils.args_global.Objects, args.name, obj)
def load_parameter_file(args):
    if isinstance(args, dict):
        _load_parameter_file(args)
    elif isinstance(args, list):
        for args_dict in args:
            _load_parameter_file(args_dict)
    else:
        raise Exception()

def _load_parameter_file(args):
    import utils_torch
    setattr(utils.args_global.ParamDicts, args.name, utils_torch.JsonFile2PyObj(args.path))
    add_log("Loading parameter file %s to parameter %s"%(args.path, args.name))

def load_configuration_file(args):
    if isinstance(args, list):
        for args_dict in args:
            _load_configuration_file(args_dict)
    else:
        _load_configuration_file(args)

def _load_configuration_file(args):
    path, name = args["path"], args["name"]
    utils.args_global["ConfigDicts"][name] = utils.JsonFile2JsonObj(path)
    add_log("Loaded configuration file from to %s config %s."%(path, name))

def add_library_path(args):
    if isinstance(args, dict):
        _add_library_path(args)
    elif isinstance(args, list):
        for args_dict in args:
            _add_library_path(args_dict)
    else:
        raise Exception()

def _add_library_path(args):
    # requires args to be a dict.
    lib_name = args['name']
    lib_path = args['path']
    if lib_path=="!get_from_config":
        success = False
        for config_name, config_dict in utils.args_global.ConfigDicts.__dict__.items():
            if config_dict.get("libs") is not None:
                libs = config_dict["libs"]
                if libs.get(lib_name) is not None:
                    lib_path = libs[lib_name]["path"]
                    success = True
                    break
        if not success:
            add_warning('add_lib failed: cannot find path to lib %s'%lib_name)
            return
    if os.path.exists(lib_path):
        if os.path.isdir(lib_path):
            sys.path.append(lib_path)
            add_log("Added library <%s> from path %s"%(lib_name, lib_path))
        else:
            add_warning('add_lib failed: path %s exists but is not a directory.'%lib_path)
    else:
        add_warning('add_lib: invalid lib_path: ', lib_path)

def parse_parameter(args):
    import utils_torch
    print(type(utils.args_global))
    ParamJsonObjParsed = utils_torch.utils.parse_param_py_obj(utils.args_global.ParamDicts)
    utils.args_global.ParamDictsOrigin = utils.args_global.ParamDicts
    utils.args_global.ParamDicts = utils.args_global.ParamDictsParsed = ParamJsonObjParsed

def train(args):
    if args.type in ["SupervisedLearning"]:
        train_supervised_learning(args)
    else:
        raise Exception()

def train_supervised_learning(args):
    return

TaskImplementFunction = {
    #"BuildModel": build_model,
    "AddLibraryPath": add_library_path,
    "LoadConfigurationFile": load_configuration_file,
    "LoadParameterFile": load_parameter_file,
    "ParseParameter": parse_parameter,
    "BuildObject": build_object,
    "Train": train,
}

def clean_log():
    do_tasks_pre()
    import utils_torch
    utils_torch.RemoveAllFiles("./log/")

if __name__=="__main__":
    if args.task in ["CleanLog", "clean_log", "cleanlog"]:
        clean_log()
    elif args.task in ["DoTasks"]:
        do_tasks()
    else:
        raise Exception()

