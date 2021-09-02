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

import utils
from utils import SetLoggerGlobal, GetLoggerGlobal, AddLog, AddWarning

SetLoggerGlobal()
logger = GetLoggerGlobal()

def do_tasks():
    try: # catch all unhandled exceptions
        do_tasks_pre()
        do_tasks_main()
    except Exception:
        logger.error(traceback.format_exc())

def do_tasks_pre():
    tasks = utils.json.JsonFile2PyObj('./task_pre.jsonc')
    for task in tasks:
        TaskImplementFunction[task.name](task.args)
    import utils_torch
    utils_torch.set_logger(GetLoggerGlobal())

def do_tasks_main():
    tasks = utils.json.JsonFile2PyObj('./task.jsonc')
    for task in tasks:
        TaskImplementFunction[task.name](task.args)

def BuildObject(args):
    if isinstance(args, list):
        for arg in args:
            _BuildObject(arg)
    elif isinstance(args, dict):
        _BuildObject(args)

def _BuildObject(args):
    import utils_torch
    Class = utils_torch.ImportModule(args.ModulePath)
    obj = Class.InitFromParam(getattr(utils.ArgsGlobal.ParamDicts, args.ParamName))
    setattr(utils.ArgsGlobal.ObjRoot, args.name, obj)
def LoadParameterFile(args):
    if isinstance(args, dict):
        _LoadParameterFile(args)
    elif isinstance(args, list):
        for args_dict in args:
            _LoadParameterFile(args_dict)
    else:
        raise Exception()

def _LoadParameterFile(args):
    import utils_torch
    ParamPyObj = utils_torch.JsonFile2PyObj(args.path)
    setattr(utils.ArgsGlobal.ParamDicts, args.name, ParamPyObj)
    AddLog("Loading parameter file %s to parameter %s"%(args.path, args.name))

def LoadConfigurationFile(args):
    if isinstance(args, list):
        for args_dict in args:
            _LoadConfigurationFile(args_dict)
    else:
        _LoadConfigurationFile(args)

def _LoadConfigurationFile(args):
    path, name = args["path"], args["name"]
    utils.ArgsGlobal["ConfigDicts"][name] = utils.JsonFile2JsonObj(path)
    AddLog("Loaded configuration file from to %s config %s."%(path, name))

def AddLibraryPath(args):
    if isinstance(args, dict):
        _AddLibraryPath(args)
    elif isinstance(args, list):
        for args_dict in args:
            _AddLibraryPath(args_dict)
    else:
        raise Exception()

def _AddLibraryPath(args):
    # requires args to be a dict.
    lib_name = args['name']
    lib_path = args['path']
    if lib_path=="!Getfrom_config":
        success = False
        for config_name, config_dict in utils.ArgsGlobal.ConfigDicts.__dict__.items():
            if config_dict.get("libs") is not None:
                libs = config_dict["libs"]
                if libs.get(lib_name) is not None:
                    lib_path = libs[lib_name]["path"]
                    success = True
                    break
        if not success:
            AddWarning('add_lib failed: cannot find path to lib %s'%lib_name)
            return
    if os.path.exists(lib_path):
        if os.path.isdir(lib_path):
            sys.path.append(lib_path)
            AddLog("Added library <%s> from path %s"%(lib_name, lib_path))
        else:
            AddWarning('add_lib failed: path %s exists but is not a directory.'%lib_path)
    else:
        AddWarning('add_lib: invalid lib_path: ', lib_path)

def parse_parameter(args):
    import utils_torch
    ParamJsonObjParsed = utils_torch.parse.ParseParamPyObj(utils.ArgsGlobal.ParamDicts)
    utils.ArgsGlobal.ParamDictsOrigin = utils.ArgsGlobal.ParamDicts
    utils.ArgsGlobal.ParamDicts = utils.ArgsGlobal.ParamDictsParsed = ParamJsonObjParsed

def train(args):
    if args.type in ["SupervisedLearning"]:
        train_supervised_learning(args)
    else:
        raise Exception()

def train_supervised_learning(args):
    return

TaskImplementFunction = {
    #"BuildModel": build_model,
    "AddLibraryPath": AddLibraryPath,
    "LoadConfigurationFile": LoadConfigurationFile,
    "LoadParameterFile": LoadParameterFile,
    "ParseParameter": parse_parameter,
    "BuildObject": BuildObject,
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

